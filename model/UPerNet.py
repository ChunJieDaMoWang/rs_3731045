import os
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from VMambaEcoder import VmambaBackbone
from ptflops import get_model_complexity_info


def get_model_size(custom_model):
    param_size = 0
    for param in custom_model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in custom_model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2  # 转换为 MB
    return size_all_mb


class TimmBackbone(nn.Module):
    features_only = True

    def __init__(self, model_cfg: dict = None, img_size: int = 256):
        super().__init__()

        timm_model = partial(
            timm.create_model,
            features_only=self.features_only,
        )
        self.timm_backbone = timm_model(**model_cfg)
        self.output_stride = self.timm_backbone.feature_info.info[-1]['reduction']
        self.feature_size = tuple([int((2 ** i) * img_size) // self.output_stride
                                   for i in reversed(range(len(self.timm_backbone.feature_info.out_indices)))])
        self.feature_channels = tuple([self.timm_backbone.feature_info.info[i]['num_chs']
                                       for i in self.timm_backbone.feature_info.out_indices])

    def _feature_transform(self, features):

        if self.timm_backbone.output_fmt.value in ('NHWC', 'NCHW'):
            _, dim1_32x, dim2_32x, dim3_32x = features[-1].size()
            if dim3_32x == self.feature_channels[-1] and dim1_32x == dim2_32x == self.feature_size[-1]:
                for i in range(len(features)):
                    features[i] = features[i].permute(0, 3, 1, 2).contiguous()

        elif self.timm_backbone.output_fmt.value in ('NLC', 'NCL'):
            _, dim1_32x, dim2_32x = features[-1].size()
            if dim2_32x == self.feature_channels[-1] and dim1_32x == int(self.feature_size[-1] ** 2):
                for i in range(len(features)):
                    features[i] = features[i].permute(0, 2, 1)

            for i in range(len(features)):
                features[i] = features[i].reshape(-1, self.feature_channels[i],
                                                  self.feature_size[i], self.feature_size[i])
        else:
            raise ValueError('维度参数存在问题')

        return features

    def forward(self, x):

        return self._feature_transform(self.timm_backbone(x))


def resnet_v2(
        pretrained: bool = False,
        model_name: str = 'resnetv2_50d_gn',
        in_chans: int = 3,
        out_indices: tuple = (1, 2, 3, 4),
        output_stride: int = 32,
        img_size: int = 256,
        **kwargs
) -> TimmBackbone:
    model_cfg = dict(
        in_chans=in_chans,
        model_name=model_name,
        out_indices=out_indices,
        output_stride=output_stride,
        pretrained=pretrained,
        **kwargs
    )

    return TimmBackbone(model_cfg, img_size)


def xception_p(
        pretrained: bool = False,
        model_name: str = 'xception65p',
        in_chans: int = 3,
        out_indices: tuple = (1, 2, 3, 4),
        output_stride: int = 32,
        img_size: int = 256,
        **kwargs
) -> TimmBackbone:
    model_cfg = dict(
        pretrained=pretrained,
        model_name=model_name,
        in_chans=in_chans,
        block_cfg=[
            # entry flow
            dict(in_chs=64, out_chs=128, stride=2),
            dict(in_chs=128, out_chs=256, stride=2),
            dict(in_chs=256, out_chs=736, stride=2),
            # middle flow
            *([dict(in_chs=736, out_chs=736, stride=1)] * 16),
            # exit flow
            dict(in_chs=736, out_chs=(736, 1024, 1024), stride=2),
            dict(in_chs=1024, out_chs=(1536, 1536, 2048), stride=1, no_skip=True),
        ],
        norm_layer=partial(nn.GroupNorm, num_groups=32),
        out_indices=out_indices,
        output_stride=output_stride,
        **kwargs
    )

    return TimmBackbone(model_cfg, img_size)


def swin_transformer(
        pretrained: bool = False,
        model_name: str = 'swin_tiny_patch4_window7_224',
        in_chans: int = 3,
        out_indices: tuple = (0, 1, 2, 3),
        output_stride: int = 32,
        img_size: int = 256,
        **kwargs
) -> TimmBackbone:
    model_cfg = dict(
        pretrained=pretrained,
        model_name=model_name,
        in_chans=in_chans,
        img_size=img_size,
        out_indices=out_indices,
        output_stride=output_stride,
        window_size=8,
        **kwargs
    )

    return TimmBackbone(model_cfg, img_size, )


def convnext(
        pretrained: bool = False,
        model_name: str = 'convnext_tiny', # v2可能不收敛
        in_chans: int = 3,
        out_indices: tuple = (0, 1, 2, 3),
        output_stride: int = 32,
        img_size: int = 256,
        **kwargs
) -> TimmBackbone:
    model_cfg = dict(
        pretrained=pretrained,
        model_name=model_name,
        in_chans=in_chans,
        out_indices=out_indices,
        output_stride=output_stride,
        **kwargs
    )

    return TimmBackbone(model_cfg, img_size)


def coatnet(
        pretrained: bool = False,
        model_name: str = 'coatnet_rmlp_0_rw_224',
        in_chans: int = 3,
        out_indices: tuple = (1, 2, 3, 4),
        img_size: int = 256,
        **kwargs
) -> TimmBackbone:

    model_cfg = dict(
        pretrained=pretrained,
        model_name=model_name,
        in_chans=in_chans,
        img_size=img_size,
        out_indices=out_indices,
        **kwargs
    )

    return TimmBackbone(model_cfg, img_size)

def maxxvit(
        pretrained: bool = False,
        model_name: str = 'maxxvit_rmlp_tiny_rw_256',
        in_chans: int = 3,
        out_indices: tuple = (1, 2, 3, 4),
        img_size: int = 256,
        **kwargs
) -> TimmBackbone:

    model_cfg = dict(
        pretrained=pretrained,
        model_name=model_name,
        in_chans=in_chans,
        img_size=img_size,
        out_indices=out_indices,
        **kwargs
    )

    return TimmBackbone(model_cfg, img_size)

def mambaout(
        pretrained: bool = False,
        model_name: str = 'mambaout_tiny',
        in_chans: int = 3,
        out_indices: tuple = (0, 1, 2, 3),
        img_size: int = 256,
        **kwargs
) -> TimmBackbone:

    model_cfg = dict(
        pretrained=pretrained,
        model_name=model_name,
        in_chans=in_chans,
        out_indices=out_indices,
        **kwargs
    )

    return TimmBackbone(model_cfg, img_size)

def vmamba(in_chans: int = 3):

    return VmambaBackbone(in_chans=in_chans)


class Decoder(nn.Module):
    def __init__(
            self,
            channels: tuple = (256, 512, 1024, 2048),
            out_channels: int = 256,
            pool_scales: tuple = (1, 2, 3, 6),
    ):
        super().__init__()
        self.pool_scales = pool_scales
        self.ppm_stages = nn.ModuleList([nn.Sequential(
            nn.AdaptiveAvgPool2d(scale),
            nn.Conv2d(channels[-1], out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        ) for scale in pool_scales])
        self.ppm_fusion = nn.Sequential(
            nn.Conv2d(len(pool_scales) * out_channels + channels[-1], out_channels, 1, bias=False),
            nn.Conv2d(out_channels, out_channels, 3, groups=out_channels, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        )

        self.feature_projection = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_c, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        ) for in_c in channels[:-1]])

        self.fpn = nn.ModuleList([nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        ) for _ in channels[:-1]])

        self.feature_fusion = nn.Sequential(
            nn.Conv2d(len(channels) * out_channels, out_channels, 1, bias=False),
            nn.Conv2d(out_channels, out_channels, 3, groups=out_channels, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        )

    def _pyramid_pooling_module(self, input_32x):
        _, _, height, width = input_32x.size()
        output_32x = [input_32x] + [
            F.interpolate(stage(input_32x), size=(height, width), mode='bilinear', align_corners=False)
            for stage in self.ppm_stages
        ]
        output_32x = torch.cat(output_32x, dim=1)

        return self.ppm_fusion(output_32x)

    def _cascade_feature_fusion(self, multi_features):

        for i, each_projection in enumerate(self.feature_projection):
            multi_features[i] = each_projection(multi_features[i])
        multi_features[-1] = self._pyramid_pooling_module(multi_features[-1])

        for i in range(len(multi_features) - 1, 0, -1):
            multi_features[i - 1] = multi_features[i - 1] + F.interpolate(
                multi_features[i], size=multi_features[i - 1].shape[2:], mode='bilinear', align_corners=False)

        for i, each_fpn_conv in enumerate(self.fpn):
            multi_features[i] = each_fpn_conv(multi_features[i])

        for i in range(len(multi_features) - 1, 0, -1):
            multi_features[i] = F.interpolate(
                multi_features[i], size=multi_features[0].shape[2:], mode='bilinear', align_corners=False
            )

        return self.feature_fusion(torch.cat(multi_features, dim=1))

    def forward(self, x):
        return self._cascade_feature_fusion(x)


class UPerNet(nn.Module):
    def __init__(self, backbone_name, en_in_channels=4, de_out_channels=384, img_size=256, num_classes=2):
        super(UPerNet, self).__init__()

        self.backbone_name, self.in_channels, self.out_channels = backbone_name, en_in_channels, de_out_channels
        self.img_size, self.num_classes = img_size, num_classes

        self.encoder = self._get_encoder()
        self.decoder = Decoder(self.encoder.feature_channels, de_out_channels)

        self.cls_seg = nn.Sequential(
            nn.Conv2d(de_out_channels, de_out_channels, 3, groups=de_out_channels, padding=1),
            nn.Conv2d(de_out_channels, num_classes, 1, bias=False)
        )

    def _get_encoder(self):
        if self.backbone_name == 'ResNet':
            return resnet_v2(in_chans=self.in_channels, img_size=self.img_size)
        elif self.backbone_name == 'Xception':
            return xception_p(in_chans=self.in_channels, img_size=self.img_size)
        elif self.backbone_name == 'Swin Transformer':
            return swin_transformer(in_chans=self.in_channels, img_size=self.img_size)
        elif self.backbone_name == 'ConvNeXt':
            return convnext(in_chans=self.in_channels, img_size=self.img_size)
        elif self.backbone_name == 'CoAtNet':
            return coatnet(in_chans=self.in_channels, img_size=self.img_size)
        elif self.backbone_name == 'MaxXViT':
            return maxxvit(in_chans=self.in_channels, img_size=self.img_size)
        elif self.backbone_name == 'MambaOut':
            return mambaout(in_chans=self.in_channels, img_size=self.img_size)
        elif self.backbone_name == 'VMamba':
            return vmamba(in_chans=self.in_channels)
        else:
            raise ValueError('还不支持你所输入的模型类型')

    def forward(self, x):
        _, _, height, width = x.size()
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.cls_seg(x)
        return F.interpolate(x, size=(height, width), mode='bilinear', align_corners=False)


def analyze_model(model, input_size=(4, 256, 256), temp_path="temp_model.pth"):
    # 1. 参数量（Parameter Count）
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f} M)")

    # 2. 模型大小（Model File Size）
    torch.save(model.state_dict(), temp_path)
    size_bytes = os.path.getsize(temp_path)
    size_mb = size_bytes / (1024 ** 2)
    print(f"Model size on disk: {size_mb:.2f} MB")
    os.remove(temp_path)

    # 3. FLOPs 计算（浮点运算次数）
    if get_model_complexity_info:
        macs, params = get_model_complexity_info(
            model, input_size, as_strings=False, print_per_layer_stat=False
        )
        # 将 MACs (Multiply–Accumulate operations) 转换为 FLOPs: FLOPs ≈ 2 * MACs
        flops = 2 * macs
        print(f"MACs: {macs/1e9:.2f} G, FLOPs: {flops/1e9:.2f} G")
    else:
        print("ptflops 未安装，无法计算 FLOPs。如需计算，请运行: pip install ptflops")


if __name__ == '__main__':
    def has_batchnorm(in_model):
        for module in in_model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                return True
        return False

    print("加载中......")
    # create_dataset()
    A = UPerNet('Xception', img_size=256).cuda()

    # summary(A, (4, 256, 256), batch_size=16, device='cuda')
    B = A(torch.randn(8, 4, 256, 256).cuda())
    print(has_batchnorm(A))
    analyze_model(A)




    a = torch.randn(8, 6, 224, 224)
    b = A(torch.randn(24, 4, 256, 256).cuda())
    models = timm.create_model('xception65p',
                              pretrained=False,
                              # num_classes=0,
                              # return_interm_layers=True,
                              # global_pool='',
                              features_only=True,
                              # img_size=256
                              # out_indices=(1, 2, 3, 4)
                              )
    # b = model(a)
    print(' ')
