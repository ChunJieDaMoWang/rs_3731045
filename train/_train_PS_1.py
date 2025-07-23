import os
import torch
import copy
import random
import pandas as pd
import numpy as np
import albumentations as A
from skimage import io
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model.UPerNet import UPerNet
from loss_functions.Combo import ComboLoss
from torchmetrics import F1Score, CohenKappa, Accuracy
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.amp import autocast, GradScaler


def set_random_seed(seed: int) -> None:
    """
    设置所有相关模块的随机种子，保证结果复现性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证每次结果一致，但可能会降低运行速度
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True  # 启用TF32加速
    torch.backends.cudnn.allow_tf32 = True


def seed_worker(worker_id: int) -> None:
    """
    DataLoader 中每个 worker 的随机种子初始化函数
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------
# 增强配置参数
# # --------------------
class Config:
    def __init__(self, args=None):
        if args is None:
            args = dict()
        elif not isinstance(args, dict):
            raise TypeError("`args` must be a dict")

        # 随机种子设置
        self.seed: int = args.setdefault("Seed", 42)

        # 高级配置
        self.use_amp = args.setdefault("Use AMP", False)
        self.grad_clip = args.setdefault('Grad Clip', 1.0)

        # 意外中断的补救
        self.resume_checkpoint = args.setdefault('Resume Checkpoint', None)

        # 系统设备配置
        self.device = args.setdefault("Device", 'cuda:1' if torch.cuda.is_available() else 'cpu')
        self.save_dir = args.setdefault("Save Dir", r'F:\_Dataset\Weight save')
        self.log_file = args.setdefault('Log File', 'training.log')

        # 数据集配置
        self.batch_size = args.setdefault("Batch Size", 16)
        self.num_classes = args.setdefault("Num Classes", 2)
        self.image_size = args.setdefault("Image Size", (256, 256))
        self.image_bands = args.setdefault("Image Bands", 4)
        self.mean_and_std_path = args.setdefault(
            'Mean And STD Path',
            r'F:\_Dataset\S1S2 Water\S2\256_256\train\S1S2Water_train_mean&std.xlsx'
        )
        self.train_txt_path = args.setdefault(
            "Train Txt Path",
            r'F:\_Dataset\S1S2 Water\S2\256_256\train\S1S2 Water Train.txt'
        )
        self.val_txt_path = args.setdefault(
            "Val Txt Path",
            r'F:\_Dataset\S1S2 Water\S2\256_256\val\S1S2 Water Val.txt'
        )
        self.channel_first = args.setdefault('Channel First', True)

        # 模型参数配置
        self.model_name = args.setdefault("Model Name", 'ResNet')
        self.decoder_channels = args.setdefault("Decoder Channels", 384)

        # 损失函数参数配置
        self.class_weight_xlsx_path = args.setdefault(
            'Class Weight Xlsx Path',
            r'F:\_Dataset\S1S2 Water\S2\256_256\train\S1S2Water_train_class_weight.xlsx'
        )
        self.tversky_alpha = args.setdefault('Tversky Alpha', 0.5)
        self.tversky_beta = args.setdefault('Tversky Beta', 0.5)
        self.tversky_gamma = args.setdefault('Tversky Gamma', 2.)
        self.ce_gamma = args.setdefault('Cross Entropy Gamma', 2.)
        self.ignore_index = args.setdefault('Ignore Index', None)
        self.label_smoothing = args.setdefault('Label Smoothing', 0.1)
        self.dynamic_weights = args.setdefault('Dynamic Weights', True)
        self.aux_weight = args.setdefault('Auxiliary Weight', None)

        # 优化器参数配置
        self.learning_rate = args.setdefault("Learning Rate", 3e-4)
        self.betas = args.setdefault("Betas", (0.8, 0.888) if self.use_amp else (0.9, 0.999))
        self.weight_decay = args.setdefault("Weight Decay", 1e-2)

        # 学习率策略配置
        self.use_scheduler = args.setdefault('Use Scheduler', True)
        self.warmup_epochs = args.setdefault('Warmup Epochs', 5)
        self.cos_t_0 = args.setdefault('COS T0', 10)
        self.cos_eta_min = args.setdefault('COS ETA MIN', 1e-5)

        # 训练参数配置
        self.max_epochs = args.setdefault('Max Epoches', 60)
        self.min_epochs = args.setdefault('Min Epoches', 30)
        self.early_stop_patience = args.setdefault('Early Stop Patience', 10)

        # 新增EMA参数
        self.ema_decay = args.setdefault('EMA Decay', None)
        self.ema_start_epoch = args.setdefault('EMA Start Epoch', 5)


# --------------------
# 增强数据集类
# --------------------
class SegmentationDataset(Dataset):
    def __init__(self, dataset_list, dataset_mean, dataset_std, data_bands, seed, dataset_type='train'):

        self.init_seed = seed
        # 定义初始周期
        self.current_epoch = 0

        try:
            self.dataset_list = np.loadtxt(dataset_list, dtype=str, delimiter=';')
        except Exception as e:
            raise ValueError(f"加载数据集列表时出错: {e}")

        self.dataset_mean = tuple([each_mean for each_mean in dataset_mean])
        self.dataset_std = tuple([each_std for each_std in dataset_std])

        self.data_bands, self.dataset_type = data_bands, dataset_type.lower()
        # self.transform = True
        self._get_transform(seed)

    def _read_tiff(self, tiff_path, tiff_type='image'):

        # 当图像为rgb和rgba时，形状为（H，W，C），其余均为（C，H，W）
        tiff_data = io.imread(tiff_path)
        if tiff_type == 'image':
            tiff_data = tiff_data.astype(np.float32)
            if self.data_bands not in (3, 4):
                return np.transpose(tiff_data, (1, 2, 0))
            else:
                return tiff_data

        elif tiff_type == 'label':
            if tiff_data.ndim > 2:
                raise ValueError('标签的维度数不对，请检查')
            else:
                self.image_size = tiff_data.shape[-1]
                return tiff_data

        else:
            raise TypeError('tiff_type类型有误')

    def _get_transform(self, seed):
        if self.dataset_type.lower() == 'train':
            self.transform = A.Compose([
                # -------------------- 几何变换（同步处理影像和标签）-------------------
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.4),
                    A.RandomRotate90(p=0.6),
                ], p=0.7),

                # -------------------- 空间域增强 --------------------
                # # 弹性形变（轻微扰动水体边界）
                # A.ElasticTransform(
                #     alpha=1, sigma=50,
                #     p=0.3
                # ),

                # # 随机遮挡（模拟云/阴影）
                # A.CoarseDropout(
                #     num_holes_range=(1, 5), hole_height_range=(0.1, 0.2), hole_width_range=(0.1, 0.2),
                #     fill=0, fill_mask=0,
                #     p=0.4
                # ),

                # -------------------- 标准化 --------------------
                A.Normalize(
                    mean=self.dataset_mean,
                    std=self.dataset_std,
                    max_pixel_value=1.0,
                ),

                # A.Transpose(p=0.5),
                # A.GridDropout(p=0.2),
                # A.RandomGridShuffle(grid=(4, 4)),
                # A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-30, 30), p=0.7),
            ], additional_targets={'mask': 'mask'}, seed=seed)
        elif self.dataset_type.lower() == 'val' or self.dataset_type.lower() == 'test':
            self.transform = A.Compose([
                A.Normalize(max_pixel_value=1.0, mean=self.dataset_mean, std=self.dataset_std),
            ])
        else:
            raise ValueError('数据集类型“dataset_type”有误，请检查')

    def set_epoch(self, epoch):
        self.current_epoch = epoch  # 新增方法，用于设置当前周期

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        image_path, label_path = self.dataset_list[idx]

        image = self._read_tiff(image_path, 'image')
        label = self._read_tiff(label_path, 'label')

        if self.transform:
            # 保存当前随机状态
            original_random_state = random.getstate()
            original_np_random_state = np.random.get_state()
            if self.dataset_type.lower() == 'train':
                # 获取当前 DataLoader worker 的 ID（多进程时）
                worker_info = torch.utils.data.get_worker_info()
                worker_id = worker_info.id if worker_info else 0

                # 生成唯一种子（结合初始种子、周期、worker ID 和样本索引）
                seed = int(int(self.init_seed) + self.current_epoch * 1000000 + worker_id * 100000 + idx)
                random.seed(seed)
                np.random.seed(seed)
                self._get_transform(seed)

            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

            # 恢复原有随机状态避免影响其他部分
            random.setstate(original_random_state)
            np.random.set_state(original_np_random_state)

        image = np.transpose(image, (2, 0, 1))
        label = label.astype(np.uint64)

        return torch.from_numpy(image), torch.from_numpy(label).long()


# --------------------
# 模型构建
# --------------------

def create_model(config):
    model = UPerNet(
        config.model_name, config.image_bands, config.decoder_channels, config.image_size[0], config.num_classes
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    print(f"不可训练参数量: {non_trainable_params / 1e6:.2f}M")

    return model


# --------------------
# 增强训练引擎
# --------------------
class SegmentationTrainer:
    def __init__(self, config):
        self.val_loader = None
        self.config = config
        self.ema_model = None
        self.best_ema_state = None
        self.step_counter = 0  # 新增步骤计数器

        self._init_components()
        self._init_dataloaders()
        self._init_scheduler()
        self._init_metrics()
        self._init_from_checkpoint()


    def _init_components(self):
        # 模型和混合精度
        self.model = create_model(self.config).to(self.config.device)
        self.autocast_kwargs = {'device_type': self.config.device, 'enabled': self.config.use_amp}
        self.scaler = GradScaler(init_scale=2 ** 16, growth_interval=2000, enabled=self.config.use_amp)

        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
            fused=True
        )

        # 初始化损失函数
        try:
            balance_class_weight = pd.read_excel(self.config.class_weight_xlsx_path, index_col=0).values.flatten()
        except Exception as e:
            raise ValueError(f"加载 class_weight 文件出错: {e}")

        balance_class_weight = balance_class_weight.sum() / (self.config.num_classes * balance_class_weight)
        self.loss_fn = ComboLoss(
            balance_class_weight, self.config.ce_gamma,
            self.config.tversky_alpha, self.config.tversky_beta, self.config.tversky_gamma,
            self.config.ignore_index, self.config.label_smoothing,
            self.config.num_classes, self.config.dynamic_weights,
            self.config.aux_weight, self.config.channel_first
        )

    def _init_metrics(self):

        # 评估指标
        self.train_metrics = {
            'acc': Accuracy(task="multiclass", num_classes=self.config.num_classes),
            'kappa': CohenKappa(task='multiclass', num_classes=self.config.num_classes),
            'f1 score': F1Score(task='multiclass', num_classes=self.config.num_classes),
            'miou': MeanIoU(num_classes=self.config.num_classes, input_format='index'),
            'g dice': GeneralizedDiceScore(num_classes=self.config.num_classes, input_format='index')
        }

        for metric in self.train_metrics.values():
            metric.to(self.config.device)

        self.val_metrics = copy.deepcopy(self.train_metrics)

    def _init_scheduler(self):
        """组合学习率调度器"""
        warmup = LinearLR(
            self.optimizer,
            start_factor=1e-2,
            total_iters=self.config.warmup_epochs
        )

        cosine = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_epochs - self.config.warmup_epochs,
            eta_min=self.config.cos_eta_min
        )

        self.lr_scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.config.warmup_epochs]
        )

    def _init_dataloaders(self):
        """
        初始化训练与验证数据加载器，设置 worker 的随机种子与生成器，确保每个 worker 随机行为可控
        """
        try:
            mean_and_std = pd.read_excel(self.config.mean_and_std_path, index_col=0)
        except Exception as e:
            raise ValueError(f"加载 mean&std 文件出错: {e}")
        data_mean, data_std = mean_and_std['mean'].values.tolist(), mean_and_std['std'].values.tolist()

        # 设置 DataLoader 的生成器，保证各 worker 的随机种子一致
        self.g = torch.Generator()
        self.g.manual_seed(self.config.seed)

        self.train_ds = SegmentationDataset(
            self.config.train_txt_path, data_mean, data_std,
            self.config.image_bands, self.config.seed, 'train'
        )

        self.val_ds = SegmentationDataset(
            self.config.val_txt_path, data_mean, data_std,
            self.config.image_bands, self.config.seed, 'val'
        )

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=16,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=self.g
        )

        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=self.g
        )

    def _init_from_checkpoint(self):
        self.resumed = False
        """从检查点恢复训练状态"""
        if self.config.resume_checkpoint and os.path.isfile(self.config.resume_checkpoint):
            checkpoint = torch.load(self.config.resume_checkpoint, map_location=self.config.device, weights_only=True)

            # 加载基础状态
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.g.set_state(torch.ByteTensor(checkpoint['generator_state'].cpu()))

            # 加载SWA相关状态
            self.ema_state = checkpoint['ema_state']
            self.best_ema_state = checkpoint.get('best_ema_state', None)
            self.step_counter = checkpoint.get('step_counter', self.step_counter)

            # 恢复训练进度
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_combined_score = checkpoint['best_combined_score']
            self.no_improve = checkpoint['no_improve']

            print(f"\n成功从检查点恢复训练 (epoch {self.start_epoch})")
            print(f"最佳组合分数: {self.best_combined_score:.4f}, 早停计数器: {self.no_improve}")
            self.resumed = True

    def _update_ema(self):
        if not self.ema_model:
            self.ema_model = torch.optim.swa_utils.AveragedModel(
                self.model, self.config.device,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(self.config.ema_decay)
            )
        self.ema_model.update_parameters(self.model)

    def train_step(self, epoch):
        self.train_ds.set_epoch(epoch)
        self.model.train()
        train_loss = torch.tensor(0., device=self.config.device)
        for each_metric in self.train_metrics.values():
            each_metric.reset()

        with tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}", leave=False) as pbar:
            for count, (images, labels) in enumerate(pbar):
                self.optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

                images = images.to(self.config.device, non_blocking=True)
                labels = labels.to(self.config.device, non_blocking=True)

                with autocast(**self.autocast_kwargs):
                    preds = self.model(images)
                    loss = self.loss_fn(preds, labels)

                # 反向传播优化
                self.scaler.scale(loss).backward()
                if self.config.grad_clip > 0. and not self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip,
                        error_if_nonfinite=True
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.config.ema_decay:
                    self.step_counter += 1
                    if self.step_counter >= len(self.train_loader) * self.config.ema_start_epoch:
                        self._update_ema()

                #
                with torch.inference_mode():
                    # 计算指标
                    preds = torch.argmax(preds.detach(), dim=1)
                    for metric in self.train_metrics.values():
                        metric.update(preds, labels)

                train_loss += loss
                pbar.set_postfix({
                    'each train loss': f"{loss.detach().item():.4f}",
                    'mean train loss': f"{(train_loss / (count + 1)).detach().item():.4f}",
                    **{k: f"{metric.compute().item():.4f}" for k, metric in self.train_metrics.items()},
                })

                del preds, loss

        return {
            'loss': (train_loss / len(self.train_loader)).detach().item(),
            **{k: metric.compute().item() for k, metric in self.train_metrics.items()}
        }

    def validate_step(self, epoch):
        torch.cuda.empty_cache()
        self.model.eval()

        val_loss = torch.tensor(0.0, device=self.config.device)
        for each_metric in self.val_metrics.values():
            each_metric.reset()

        with torch.inference_mode(), tqdm(self.val_loader, desc=f"Validating Epoch {epoch + 1}", leave=False) as pbar:
            for count, (images, labels) in enumerate(pbar):
                # 根据设备选择数据传输位置
                images = images.to(self.config.device, non_blocking=True)
                labels = labels.to(self.config.device, non_blocking=True)

                # 执行前向传播
                with autocast(**self.autocast_kwargs):  # 保持与训练一致的AMP设置
                    if self.step_counter >= len(self.train_loader) * self.config.ema_start_epoch:
                        self.ema_model.eval()
                        preds = self.ema_model(images)
                    else:
                        preds = self.model(images)

                    loss = self.loss_fn(preds, labels)

                preds = torch.argmax(preds.detach(), dim=1)
                # 更新指标
                for name, metric in self.val_metrics.items():
                    metric.update(preds, labels)

                val_loss += loss.item()
                pbar.set_postfix({
                    'each val loss': f"{loss.detach().item():.4f}",
                    'mean val loss': f"{(val_loss / (count + 1)).detach().item():.4f}",
                    **{k: f"{metric.compute().item():.4f}" for k, metric in self.val_metrics.items()},
                })

        # 恢复原始参数并清理显存
        torch.cuda.empty_cache()

        # 汇总验证指标
        return {
            'loss': (val_loss / len(self.val_loader)).detach().item(),
            **{k: metric.compute().item() for k, metric in self.val_metrics.items()}
        }

    @staticmethod
    def _calc_combined_score(metrics):

        return 0.5 * metrics['miou'] + 0.3 * metrics['g dice'] + 0.15 * metrics['kappa'] + 0.05 * metrics['f1 score']

    def _save_checkpoint(self, epoch, best_combined_score, no_improve):
        """保存检查点（新增EMA状态）"""
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'ema_state': self.ema_model.state_dict() if self.ema_model else None,
            'best_ema_state': self.best_ema_state,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.lr_scheduler.state_dict(),
            'generator_state': self.g.get_state(),
            'best_combined_score': best_combined_score,
            'no_improve': no_improve,
            'step_counter': self.step_counter,
            'config': vars(self.config)  # 保存配置信息
        }

        # 常规检查点
        torch.save(checkpoint,
                   os.path.join(self.config.save_dir, f"{self.model.backbone_name}_checkpoint_epoch_{epoch + 1}.pth"))

    def _check_early_stop(self, epoch, no_improve):
        return (epoch >= self.config.min_epochs) and (no_improve >= self.config.early_stop_patience)

    def _finalize_training(self):
        # 保存最终EMA模型（CPU参数）
        if self.ema_model:
            torch.save(
                {'ema_model': self.ema_model.state_dict()},
                os.path.join(self.config.save_dir, f'{self.model.backbone_name}_final_ema.pth')
            )
        # 保存常规模型
        torch.save(
            self.model.state_dict(),
            os.path.join(self.config.save_dir, f'{self.model.backbone_name}_final.pth')
        )

    def run(self):
        os.makedirs(self.config.save_dir, exist_ok=True)

        start_epoch, no_improve, best_combined_score = 0, 0, 0.
        # 如果是从检查点恢复，覆盖初始值
        if self.resumed:
            best_combined_score = self.best_combined_score
            no_improve = self.no_improve
            start_epoch = self.start_epoch

        # epoch, val_metrics = 0, {}
        for epoch in range(start_epoch, self.config.max_epochs):
            # 训练,验证阶段
            train_metrics = self.train_step(epoch)
            self.lr_scheduler.step()
            val_metrics = self.validate_step(epoch)

            val_combined_score = self._calc_combined_score(val_metrics)
            train_combined_score = self._calc_combined_score(train_metrics)

            # 更新日志格式
            log_msg = (
                f"Epoch {epoch + 1:03d} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "

                f"Train Acc: {train_metrics['acc']:.4f} | "
                f"Val Acc: {val_metrics['acc']:.4f} | "

                f"Train F1: {train_metrics['f1 score']:.4f} | "
                f"Val F1: {val_metrics['f1 score']:.4f} | "

                f"Train Kappa: {train_metrics['kappa']:.4f} | "
                f"Val Kappa: {val_metrics['kappa']:.4f} | "

                f"Train mIoU: {train_metrics['miou']:.4f} | "
                f"Val mIoU: {val_metrics['miou']:.4f} | "

                f"Train G Dice: {train_metrics['g dice']:.4f} | "
                f"Val G Dice: {val_metrics['g dice']:.4f} | "

                f"Train Combined Score: {train_combined_score:.4f} | "
                f"Val Combined Score: {val_combined_score:.4f} | "

                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            print('\n' + log_msg)
            with open(os.path.join(
                    self.config.save_dir, '_'.join([self.model.backbone_name, self.config.log_file])
            ), 'a') as f:
                f.write(log_msg + '\n')

            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, best_combined_score, no_improve)

            # 保存最佳模型
            if val_combined_score >= best_combined_score + 1e-4:
                best_combined_score = val_combined_score
                no_improve = 0

                self.best_ema_state = copy.deepcopy(self.ema_model.state_dict()) if self.ema_model else None
                torch.save(self.model.state_dict() if not self.best_ema_state else self.best_ema_state,
                           os.path.join(self.config.save_dir, f"{self.model.backbone_name}_best_model.pth"))

                # 清理显存
                torch.cuda.empty_cache()

            else:
                no_improve += 1

            if self._check_early_stop(epoch, no_improve):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
                break

        # 最终处理（包含ema）
        self._finalize_training()


# --------------------
# 主程序入口
# --------------------
if __name__ == "__main__":
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

    args_dict = {

        "Train Txt Path": r'E:\_Train Data\Custom Planet Scope\256_256\Custom Planet Scope Train_80%.txt',
        "Val Txt Path": r'E:\_Train Data\Custom Planet Scope\256_256\Custom Planet Scope Val_80%.txt',

        'Mean And STD Path': r'E:\_Train Data\Custom Planet Scope\256_256\Custom Planet Scope_train_mean&std_80%.xlsx',
        'Class Weight Xlsx Path': r'E:\_Train Data\Custom Planet Scope\256_256\Custom Planet Scope_train_class_weight_80%.xlsx',

        "Save Dir": r'G:\_Model Weight\直接训练',

        "Batch Size": 16,
        "Use AMP": False,
        'Grad Clip': 1.,

        "Decoder Channels": 384,

        # 优化器相关
        "Learning Rate": 1e-3,  # 基础学习率降低（原3e-4）
        "Weight Decay": 1e-2,  # 头部权重衰减加强（原1e-2）

        # 调度器相关
        "Warmup Epochs": 10,  # 更长的预热（原5）
        # "COS T0": 10,  # 余弦周期延长(原5)
        "COS ETA MIN": 1e-4,  # 最小学习率更低（原1e-5）

        'Max Epoches': 100,
        'Min Epoches': 50,

        'EMA Start Epoch': 0,
        'EMA Decay': 0.999,
        'Early Stop Patience': 100,

        # 'Resume Checkpoint': r'G:\_Model Weight\直接训练\Swin Transformer_checkpoint_epoch_75.pth',

        "Device": 'cuda:1',

    }
    total_config = Config(args_dict)
    set_random_seed(total_config.seed)
    # 训练流程
    trainer = SegmentationTrainer(total_config)
    trainer.run()
