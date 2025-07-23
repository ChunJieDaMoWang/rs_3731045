import math
import torch
import torch.nn as nn
from functools import partial
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...


### ======== Base module ======= ###
class DropPath(nn.Module):
    """
        Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        初始化 DropPath。
    Args:
        drop_prob: 丢弃概率（0.0-1.0）。
        scale_by_keep: 是否在丢弃后按保持的比例缩放。

    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        """
        前向传播逻辑，直接实现路径丢弃。
        :param x: 输入张量。
        :return: 随机丢弃路径后的张量。
        """
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 适配不同维度的张量
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)  # 创建随机掩码

        if self.scale_by_keep and keep_prob > 0.0:
            random_tensor.div_(keep_prob)  # 根据保持率缩放

        return x * random_tensor  # 应用丢弃掩码

    def extra_repr(self):
        """
        :return: 返回额外的模块信息（如丢弃概率）。
        """
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class LayerNorm2D(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, channel_first=True):
        super().__init__()

        self.norm_layer = nn.GroupNorm(1, normalized_shape, eps) \
            if channel_first else nn.LayerNorm(normalized_shape, eps)

    def forward(self, x):
        return self.norm_layer(x)


class Linear2D(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, channel_first=True):
        super().__init__()

        self.linear_proj = nn.Conv2d(in_channels, out_channels, 1, bias=bias) \
            if channel_first else nn.Linear(in_channels, out_channels, bias)

    def forward(self, x: torch.Tensor):
        return self.linear_proj(x)


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros',
                 channel_first=True):
        super().__init__()

        self.channel_first = channel_first
        self.conv_proj = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                   dilation, groups, bias, padding_mode)

    def forward(self, x: torch.Tensor):
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()

        x = self.conv_proj(x)

        if not self.channel_first:
            x = x.permute(0, 2, 3, 1).contiguous()

        return x


### ======== Stem Part ======== ###
class Stem(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, patch_size=3, channel_first=True):
        super().__init__()
        self.channel_first = channel_first

        self.conv1 = self._bsconv_u(in_channels, out_channels // 2, patch_size, 2, False)
        self.norm1 = LayerNorm2D(out_channels // 2, channel_first=channel_first)
        self.conv2 = self._bsconv_u(out_channels // 2, out_channels, patch_size, 2, False)
        self.norm2 = LayerNorm2D(out_channels, channel_first=channel_first)
        self.gelu = nn.GELU()

    def _bsconv_u(self, in_channels, out_channels, kernel_size, stride, bias):
        return nn.Sequential(
            Linear2D(in_channels, out_channels, bias=False, channel_first=self.channel_first),
            Conv2D(out_channels, out_channels, kernel_size, stride,
                   kernel_size // 2, 1, out_channels, bias, channel_first=self.channel_first),
        )

    def forward(self, x):
        return self.gelu(self.norm2(self.conv2(self.norm1(self.conv1(x)))))


### ======== GRN Part ======== ###

class GlobalResponseNorm(nn.Module):
    """
    GRN（全局响应归一化）层。
    """

    def __init__(self, channels, channel_first=True):
        super().__init__()
        # 可训练参数 gamma 和 beta，分别用于缩放和偏移
        if channel_first:
            self.gamma, self.beta = [nn.Parameter(torch.zeros(1, channels, 1, 1)) for _ in range(2)]
            self.chw_dims = (1, 2, 3)
        else:
            self.gamma, self.beta = [nn.Parameter(torch.zeros(1, 1, 1, channels)) for _ in range(2)]
            self.chw_dims = (3, 1, 2)

        self.channel_first = channel_first

    def _apply_grn(self, x):
        # 计算输入张量在空间维度 (height, width) 上的 L2 范数
        g_x = torch.norm(x, p=2, dim=self.chw_dims[1:], keepdim=True)

        # 计算 g_x 在通道维度 (channel) 上的均值，用于归一化
        mean_g_x = torch.mean(g_x, dim=self.chw_dims[0], keepdim=True)

        # 对 g_x 进行归一化，并使用 clamp_min 避免数值过小导致的不稳定
        normalized_g_x = g_x / torch.clamp_min(mean_g_x, 1e-6)

        # 应用可训练参数 gamma 和 beta，以及原始输入张量 x
        # 输出结果的形状与输入相同
        return self.gamma * (x * normalized_g_x) + self.beta + x

    def forward(self, x):

        return self._apply_grn(x)


### ======== Decoder Part (Mamba) ======== ###
class SelectiveScanCuda(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None


# U型扫描
# 由左上角开始往下纵向扫描
def vertical_scan(tensor: torch.Tensor, initial_direction: str = 'down' or 'up'):
    batch, channel, height, width = tensor.size()
    match initial_direction:
        case 'down':
            tensor = tensor
        case 'up':
            tensor = tensor.flip(-1, -2)
        case _:
            ValueError('所输入的初始方向参数有误，请检查')

    index = (torch.arange(height, device=tensor.device).unsqueeze(1)
             + torch.zeros(width, dtype=torch.long, device=tensor.device))
    index[:, range(1, width, 2)] = index[:, range(1, width, 2)].flipud()
    index = index.unsqueeze(0).unsqueeze(0).expand(batch, channel, -1, -1)

    return tensor.gather(-2, index).transpose(-1, -2).reshape(batch, channel, height, width)


def vertical_merge(tensor: torch.Tensor,
                   initial_direction: str = 'down' or 'up'):
    batch, channel, height, width = tensor.size()

    index = (torch.arange(height, device=tensor.device).unsqueeze(1)
             + torch.zeros(width, dtype=torch.long, device=tensor.device))
    index[:, range(1, width, 2)] = index[:, range(1, width, 2)].flipud()
    index = index.unsqueeze(0).unsqueeze(0).expand(batch, channel, -1, -1)
    tensor = tensor.transpose(-1, -2).gather(-2, index)
    match initial_direction:
        case 'down':
            tensor = tensor
        case 'up':
            tensor = tensor.flip(-1, -2)
        case _:
            ValueError('所输入的初始方向参数有误，请检查')

    return tensor


# 由左上角开始往右横向扫描
def horizontal_scan(tensor: torch.Tensor, initial_direction: str = 'right' or 'left'):
    batch, channel, height, width = tensor.size()
    match initial_direction:
        case 'right':
            tensor = tensor
        case 'left':
            tensor = tensor.flip(-1, -2)
        case _:
            ValueError('所输入的初始方向参数有误，请检查')

    index = (torch.arange(width, device=tensor.device)
             + torch.zeros(height, dtype=torch.long, device=tensor.device).unsqueeze(1))
    index[range(1, height, 2), :] = index[range(1, height, 2), :].fliplr()
    index = index.unsqueeze(0).unsqueeze(0).expand(batch, channel, -1, -1)

    return tensor.gather(-1, index).reshape(batch, channel, height, width)


def horizontal_merge(tensor: torch.Tensor,
                     initial_direction: str = 'right' or 'left'):
    batch, channel, height, width = tensor.size()

    index = (torch.arange(width, device=tensor.device)
             + torch.zeros(height, dtype=torch.long, device=tensor.device).unsqueeze(1))
    index[range(1, height, 2), :] = index[range(1, height, 2), :].fliplr()
    index = index.unsqueeze(0).unsqueeze(0).expand(batch, channel, -1, -1)
    tensor = tensor.gather(-1, index)

    match initial_direction:
        case 'right':
            tensor = tensor
        case 'left':
            tensor = tensor.flip(-1, -2)
        case _:
            ValueError('所输入的初始方向参数有误，请检查')

    return tensor


def cross_scan_fwd(x: torch.Tensor):
    B, C, H, W = x.shape
    y = x.new_empty((B, 4, C, H, W))
    for i, (each_direction, each_merge) in enumerate(zip(
            ('down', 'up', 'right', 'left',
                    # 'Top left', 'Lower right', 'Top left Anti', 'Lower right Anti'
             ),
            (vertical_scan, vertical_scan, horizontal_scan, horizontal_scan,
                    # diagonal_scan, diagonal_scan, diagonal_scan, diagonal_scan
             )
    )):
        y[:, i] = each_merge(x, each_direction)

    return y


def cross_merge_fwd(y: torch.Tensor):
    B, K, D, H, W = y.shape
    x = y.new_zeros((B, D, H, W))
    for i, (each_direction, each_merge) in enumerate(zip(
            ('down', 'up', 'right', 'left',
                    # 'Top left', 'Lower right', 'Top left Anti', 'Lower right Anti'
             ),
            (vertical_merge, vertical_merge, horizontal_merge, horizontal_merge,
                    # diagonal_merge, diagonal_merge, diagonal_merge, diagonal_merge
             )
    )):
        x += each_merge(y[:, i].reshape(B, D, H, W), each_direction)

    return x


class CrossScanF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        # x: (B, C, H, W) | (B, 8, C, H, W)
        # y: (B, 8, C, H * W)
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)

        return cross_scan_fwd(x)

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape

        return cross_merge_fwd(ys)


class CrossMergeF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        # x: (B, C, H, W) | (B, 8, C, H, W)
        # y: (B, 8, C, H * W)
        B, K, C, H, W = ys.shape
        ctx.shape = (B, C, H, W)
        return cross_merge_fwd(ys)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, h, w)
        B, C, H, W = ctx.shape
        return cross_scan_fwd(x)


class MambaInitialization:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))  # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))  # (K, inner)
        del dt_projs

        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class SS2DBlock(nn.Module):
    # https://github.com/XiaoBuL/CM-UNet/blob/main/mamba_sys.py
    def __init__(
            self,
            # basic dims ===========
            d_model: int = 96,
            d_state: int = 16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v2",
            channel_first=True,
            # ======================
            **kwargs,
    ):
        # factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model, self.d_state = d_model, d_state
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_d_conv = d_conv > 1

        # tags for forward_type ==============================
        def check_postfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_z, forward_type = check_postfix("_noz", forward_type)

        self.out_norm = LayerNorm2D(self.d_inner, channel_first=channel_first)

        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=False),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        # in proj =======================================
        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear2D(self.d_model, d_proj, bias, channel_first)
        self.grn = GlobalResponseNorm(d_proj // 2, channel_first) if self.disable_z else nn.Identity()
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_d_conv:
            self.conv2d = Conv2D(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                groups=self.d_inner,
                bias=conv_bias,
                channel_first=channel_first,
                # **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = Linear2D(self.d_inner, self.d_model, bias, channel_first)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = MambaInitialization.init_dt_A_D(
                self.d_state, self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                k_group=self.k_group,
            )

    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            # ==============================
            **kwargs,
    ):
        delta_softplus = True
        # out_norm = self.out_norm
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(
                u: torch.Tensor,  # (B, K * C, L)
                delta: torch.Tensor,  # (B, K * C, L)
                A: torch.Tensor,  # (K * C, N)
                B: torch.Tensor,  # (B, K, N, L)
                C: torch.Tensor,  # (B, K, N, L)
                D: torch.Tensor = None,  # (K * C)
                delta_bias: torch.Tensor = None,  # (K * C)
                delta_softplus=True,
                oflex=True,
        ):

            return SelectiveScanCuda.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex)

        x_proj_bias = getattr(self, "x_proj_bias", None)
        xs = CrossScanF.apply(x)
        xs = xs.reshape(B, K, D, L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.reshape(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        if hasattr(self, "dt_projs_weight"):
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        # xs = xs.view(B, -1, L)
        # dts = dts.contiguous().view(B, -1, L)
        # As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
        # Ds = self.Ds.to(torch.float)  # (K * c)
        # Bs = Bs.contiguous().view(B, K, N, L)
        # Cs = Cs.contiguous().view(B, K, N, L)
        # delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

        xs = xs.reshape(B, -1, L)
        dts = dts.reshape(B, -1, L)
        As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
        Ds = self.Ds.to(torch.float)  # (K * c)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        delta_bias = self.dt_projs_bias.reshape(-1).to(torch.float)

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        # 这里把矩阵拆分成不同方向的序列，并进行扫描
        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, ssoflex
        ).reshape(B, K, -1, H, W)

        y: torch.Tensor = CrossMergeF.apply(ys)

        y = y.reshape(B, -1, H, W)
        # y = out_norm(y)

        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            z = self.grn(z)

        if self.with_d_conv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)

        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()

        y = self.forward_core(x)

        if not self.channel_first:
            y = y.permute(0, 2, 3, 1).contiguous()

        y = self.out_norm(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


### ======== MLP Part ======== ###
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., channel_first=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear2D(in_features, hidden_features, channel_first=channel_first)
        self.act = act_layer()
        self.fc2 = Linear2D(hidden_features, out_features, channel_first=channel_first)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VMambaBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = LayerNorm2D,
            channel_first: bool = True,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            # =============================
            _SS2D: type = SS2DBlock,
            **kwargs,
    ):
        super().__init__()

        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim, channel_first=channel_first)
            self.op = _SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0

        if self.mlp_branch:
            _MLP = Mlp
            self.norm2 = norm_layer(hidden_dim, channel_first=channel_first)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channel_first=channel_first)

    def _forward(self, x_input: torch.Tensor):
        x = x_input.clone()

        if self.ssm_branch:
            x = x + self.drop_path(self.op(self.norm(x)))

        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def forward(self, x: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)

class VmambaBackbone(nn.Module):
    def __init__(
            self,
            in_chans=3,
            patch_size=3,
            depths=(2, 2, 6, 2),
            dims=96,
            norm_layer: nn.Module = LayerNorm2D,
            channel_first: bool = True,
            # =============================
            ssm_d_state: int = 1,
            ssm_conv_bias=False,
            forward_type="v2_noz",
            # =============================
            drop_path_rate=0.2,
            # =============================
    ):
        super().__init__()
        self.feature_channels = tuple([dims * (2 ** i) for i in range(len(depths))])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.stage1 = nn.Sequential(
            Stem(in_chans, self.feature_channels[0], patch_size, channel_first),
            *[VMambaBlock(self.feature_channels[0], dpr[cur + i], norm_layer,
                          channel_first=channel_first, ssm_d_state=ssm_d_state,
                          ssm_conv_bias=ssm_conv_bias, forward_type=forward_type)
              for i in range(depths[0])]
        )
        cur += depths[0]

        self.stage2 = nn.Sequential(
            Conv2D(self.feature_channels[0], self.feature_channels[1], 3, 2, 1,
                   bias=False, channel_first=channel_first),
            norm_layer(self.feature_channels[1], channel_first=channel_first),
            *[VMambaBlock(self.feature_channels[1], dpr[cur + i],
                          channel_first=channel_first, ssm_d_state=ssm_d_state,
                          ssm_conv_bias=ssm_conv_bias, forward_type=forward_type)
              for i in range(depths[1])]
        )
        cur += depths[1]

        self.stage3 = nn.Sequential(
            Conv2D(self.feature_channels[1], self.feature_channels[2], 3, 2, 1,
                   bias=False, channel_first=channel_first),
            norm_layer(self.feature_channels[2], channel_first=channel_first),
            *[VMambaBlock(self.feature_channels[2], dpr[cur + i],
                          channel_first=channel_first, ssm_d_state=ssm_d_state,
                          ssm_conv_bias=ssm_conv_bias, forward_type=forward_type)
              for i in range(depths[2])]
        )
        cur += depths[2]

        self.stage4 = nn.Sequential(
            Conv2D(self.feature_channels[2], self.feature_channels[3], 3, 2, 1,
                   bias=False, channel_first=channel_first),
            norm_layer(self.feature_channels[3], channel_first=channel_first),
            *[VMambaBlock(self.feature_channels[3], dpr[cur + i],
                          channel_first=channel_first, ssm_d_state=ssm_d_state,
                          ssm_conv_bias=ssm_conv_bias, forward_type=forward_type)
              for i in range(depths[3])]
        )
        cur += depths[3]

    def forward(self, x):
        x_4x = self.stage1(x)
        x_8x = self.stage2(x_4x)
        x_16x = self.stage3(x_8x)
        x_32x = self.stage4(x_16x)

        return [x_4x, x_8x, x_16x, x_32x]


if __name__ == "__main__":
    a = torch.randn((8, 4, 256, 256)).cuda()
    b = VmambaBackbone(4).cuda()


    c = b(a)
    print(' ')
    print(' ')
