import torch
import timm
import torch.distributed
import torch.nn.functional as F
from torch import nn
from einops import rearrange, einsum
from natten.functional import na2d_av
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath, to_2tuple
from timm.models.registry import register_model
from mmdet.models.builder import MODELS
from mmdet.utils import get_root_logger
try:
    from mmcv.runner import load_checkpoint
except:
    from mmengine.runner import load_checkpoint
# from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM

def clear_memory_cache():
    """清理GPU缓存，降低显存占用"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@torch.jit.script
def fused_scale_shift(x, scale, shift):
    return x * scale + shift

@torch.jit.script
def fused_scale(x, scale):
    return x * scale

class JITLayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1) * init_value)
        self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        return fused_scale_shift(x, self.weight, self.bias)
    
@torch.jit.script
def fused_grn_apply(x, gamma, beta, eps: float):
    sq_sum = x.pow(2).sum(dim=[2, 3], keepdim=True)
    gx = torch.sqrt(sq_sum)
    gx_mean = gx.mean(dim=[1], keepdim=True)
    nx = gx / (gx_mean + eps)
    scale = gamma * nx + 1.0
    return x * scale + beta

class JITGRN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        return fused_grn_apply(x, self.gamma, self.beta, self.eps)
    
def get_conv2d(in_channels, 
               out_channels, 
               kernel_size, 
               stride, 
               padding, 
               dilation, 
               groups, 
               bias,
               attempt_use_lk_impl=True):
    
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)

    if attempt_use_lk_impl and need_large_impl:
        print('---------------- trying to import iGEMM implementation for large-kernel conv')
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            print('---------------- found iGEMM implementation ')
        except:
            DepthWiseConv2dImplicitGEMM = None
            print('---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
        if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:
            print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    
    return nn.Conv2d(in_channels, out_channels, 
                     kernel_size=kernel_size, 
                     stride=stride,
                     padding=padding, 
                     dilation=dilation, 
                     groups=groups, 
                     bias=bias)


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)


def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std

def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1)).to(kernel.device)
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


def stem(in_chans=3, embed_dim=96):
    return nn.Sequential(
        nn.Conv2d(in_chans, embed_dim//2, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim//2),
        nn.GELU(),
        nn.Conv2d(embed_dim//2, embed_dim//2, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim//2),
        nn.GELU(),
        nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim),
        nn.GELU(),
        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim)
    )


def downsample(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
    )        


class SEModule(nn.Module):
    def __init__(self, dim, red=8, inner_act=nn.ReLU, out_act=nn.Hardsigmoid):
        super().__init__()
        inner_dim = max(16, dim // red)
        self.fc = nn.Sequential(
            nn.Linear(dim, inner_dim),
            inner_act(inplace=True),
            nn.Linear(inner_dim, dim),
            out_act(inplace=True)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        y = x.mean(dim=(2, 3)).view(B, C)
        y = self.fc(y)
        y = y.view(B, C, 1, 1)
        return x * y



class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5, use_bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1) * init_value)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1, dim, 1, 1))
        else:
            self.bias = None

    def forward(self, x):
        # x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        # return x
        if self.bias is not None:
            return x * self.weight + self.bias
        return x * self.weight


class LayerNorm2d(nn.Module):
    '''  其实是rmsnorm2d，但是懒得改后面的名字了   '''
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = x.pow(2).mean(dim=1, keepdim=True)
        norm_x = x * torch.rsqrt(var + self.eps)
        
        return norm_x * self.weight


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, dim, eps=1e-6, use_bias=True):
        super().__init__()
        self.eps = eps
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        gx = torch.linalg.vector_norm(x, ord=2, dim=(2, 3), keepdim=True)
        
        gx_mean = gx.mean(dim=1, keepdim=True)
        nx = gx * torch.rsqrt(gx_mean.pow(2) + self.eps)
        
        scale_factor = torch.reciprocal(gx_mean + self.eps)
        nx = gx * scale_factor
        
        scale = (self.gamma * nx).add_(1.0)
        
        if self.beta is not None:
            return torch.addcmul(self.beta, x, scale)
        else:
            return x * scale


class SelfAttention(nn.Module):
    """
    自注意力机制模块（显存优化版本）
    支持多头自注意力，适用于2D特征图 (N, C, H, W)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., chunk_size=512):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # self.chunk_size = chunk_size

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.attn_drop_rate = attn_drop 
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        
        qkv = self.qkv(x)  # B, 3*C, H, W
        qkv = rearrange(qkv, 'b (three head c) h w -> three b head (h w) c', three=3, head=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        del qkv
        
        # if N <= self.chunk_size:
        #     attn = (q @ k.transpose(-2, -1)) * self.scale
        #     attn = torch.softmax(attn, dim=-1)
        #     attn = self.attn_drop(attn)
        #     out = attn @ v
        #     del attn, q, k, v
        # else:
        #     out = self._chunked_attention(q, k, v, N)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_rate if self.training else 0.,
            scale=self.scale
        )
        
        x = rearrange(out, 'b head (h w) c -> b (head c) h w', h=H, w=W)
        del out
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    # def _chunked_attention(self, q, k, v, N):
    #     """分块计算注意力，降低显存占用"""
    #     B, num_heads, _, head_dim = q.shape
    #     chunk_size = self.chunk_size
    #     num_chunks = (N + chunk_size - 1) // chunk_size
        
    #     output = torch.zeros_like(q)
    #     k_t = k.transpose(-2, -1)  # 预先转置K
        
    #     for i in range(num_chunks):
    #         start_idx = i * chunk_size
    #         end_idx = min((i + 1) * chunk_size, N)
            
    #         q_chunk = q[:, :, start_idx:end_idx, :]
    #         attn_chunk = (q_chunk @ k_t) * self.scale
    #         attn_chunk = torch.softmax(attn_chunk, dim=-1)
    #         attn_chunk = self.attn_drop(attn_chunk)
    #         output[:, :, start_idx:end_idx, :] = attn_chunk @ v
    #         del q_chunk, attn_chunk
        
    #     del k_t
    #     return output
    


class DilatedReparamBlock(nn.Module):
    """
    Implementation of the RPB (Re-parameterization Block) as described in:
    "Multi-focus image fusion based on re-parameterized large kernel convolution and edge information fusion"
    
    Replaces the original UniRepLKNet block while maintaining the same API.
    """

    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.deploy = deploy
        self.channels = channels
        self.kernel_size = kernel_size
        self.attempt_use_lk_impl = attempt_use_lk_impl
        self.act = nn.GELU()

        if deploy:
            self.rpb_pw = get_conv2d(channels, channels, kernel_size=1, stride=1, padding=0, 
                                     dilation=1, groups=1, bias=True)
            self.rpb_dw = get_conv2d(channels, channels, kernel_size=kernel_size, stride=1, 
                                     padding=kernel_size//2, dilation=1, groups=channels, bias=True,
                                     attempt_use_lk_impl=attempt_use_lk_impl)
        else:
            self.pw_conv = get_conv2d(channels, channels, kernel_size=1, stride=1, padding=0, 
                                      dilation=1, groups=1, bias=False)
            self.pw_bn = get_bn(channels, use_sync_bn)
            
            self.dw_conv_k = get_conv2d(channels, channels, kernel_size=kernel_size, stride=1,
                                        padding=kernel_size//2, dilation=1, groups=channels, bias=False,
                                        attempt_use_lk_impl=attempt_use_lk_impl)
            self.dw_bn_k = get_bn(channels, use_sync_bn)
            
            self.dw_conv_1 = get_conv2d(channels, channels, kernel_size=1, stride=1, 
                                        padding=0, dilation=1, groups=channels, bias=False)
            self.dw_bn_1 = get_bn(channels, use_sync_bn)

    def forward(self, x):
        if self.deploy:
            x = self.rpb_pw(x)
            x = self.rpb_dw(x)
            return self.act(x)

        # 训练模式
        
        # Stage 1: Pointwise + Identity
        # x_pw = BN(Conv1x1(x))
        x_pw = self.pw_bn(self.pw_conv(x))
        # Add Identity: x_pw = x_pw + x (+ in-place add_)
        x_pw.add_(x) 
        
        # Stage 2: Parallel Depthwise
        # out = BN(DW_KxK(x_pw))
        out = self.dw_bn_k(self.dw_conv_k(x_pw))
        # Add Parallel 1x1 DW: out = out + BN(DW_1x1(x_pw))
        out.add_(self.dw_bn_1(self.dw_conv_1(x_pw)))
        
        return self.act(out)

    def merge_dilated_branches(self):
        """
        但实际上是 Pointwise 和 Depthwise 的重参数化融合
        """
        if self.deploy:
            return

        # 1. 融合 Stage 1 (Pointwise + Identity)
        # 融合 Conv1x1 和 BN
        k_pw, b_pw = fuse_bn(self.pw_conv, self.pw_bn)
        
        # 融合 Identity 连接
        # k_pw shape: (C, C, 1, 1)
        identity_matrix = torch.eye(self.channels).reshape(self.channels, self.channels, 1, 1).to(k_pw.device)
        k_pw = k_pw + identity_matrix
        
        # 创建部署用的 Pointwise 卷积
        self.rpb_pw = get_conv2d(self.channels, self.channels, kernel_size=1, stride=1, padding=0, 
                                 dilation=1, groups=1, bias=True)
        self.rpb_pw.weight.data = k_pw
        self.rpb_pw.bias.data = b_pw

        # 2. 融合 Stage 2 (Depthwise KxK + Depthwise 1x1)
        # 分别融合 BN
        k_dw_k, b_dw_k = fuse_bn(self.dw_conv_k, self.dw_bn_k)
        k_dw_1, b_dw_1 = fuse_bn(self.dw_conv_1, self.dw_bn_1)
        
        # 将 1x1 DW 核填充(Pad)为 KxK 大小，以便相加
        # F.pad 参数顺序: (left, right, top, bottom)
        pad_size = (self.kernel_size - 1) // 2
        k_dw_1_padded = F.pad(k_dw_1, (pad_size, pad_size, pad_size, pad_size))
        
        # 叠加权重和偏置
        k_final_dw = k_dw_k + k_dw_1_padded
        b_final_dw = b_dw_k + b_dw_1
        
        # 创建部署用的 Depthwise 卷积
        self.rpb_dw = get_conv2d(self.channels, self.channels, kernel_size=self.kernel_size, stride=1, 
                                 padding=pad_size, dilation=1, groups=self.channels, bias=True,
                                 attempt_use_lk_impl=self.attempt_use_lk_impl)
        self.rpb_dw.weight.data = k_final_dw
        self.rpb_dw.bias.data = b_final_dw

        for attr in ['pw_conv', 'pw_bn', 'dw_conv_k', 'dw_bn_k', 'dw_conv_1', 'dw_bn_1']:
            if hasattr(self, attr):
                delattr(self, attr)
        
        self.deploy = True
       

class CTXDownsample(nn.Module):
    def __init__(self, dim, h_dim):
        super().__init__()
        
        # self.x_proj = nn.Sequential(
        #     nn.Conv2d(dim, h_dim, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(h_dim)
        # )
        # x_proj: 升维，拆分为 DW(下采样) + PW(升维)
        self.x_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, h_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(h_dim)
        )


        # self.h_proj = nn.Sequential(
        #     nn.Conv2d(h_dim//4, h_dim//4, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(h_dim//4)
        # )

        # h_proj: 通道数不变，直接用 DW 卷积下采样省去 Pointwise
        # 或者使用 AvgPool2d(2) + Conv1x1
        self.h_proj = nn.Sequential(
            nn.Conv2d(h_dim//4, h_dim//4, kernel_size=3, stride=2, padding=1, groups=h_dim//4, bias=False),
            nn.BatchNorm2d(h_dim//4)
        )

    def forward(self, x, ctx):
        x = self.x_proj(x)
        ctx = self.h_proj(ctx)
        return (x, ctx)


class ResDWConv(nn.Conv2d):
    '''
    Depthwise convolution with residual connection
    '''
    """
    支持重参数化的残差深度卷积。
    
    y = x + Conv(x) = (Identity + Conv) * x
    
    把 Identity 视为一个中心为1、其余为0的卷积核，将其加到 Conv 的权重上，推理时为单层卷积。
    """
    def __init__(self, dim, kernel_size=3, deploy=False):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim, bias=True)
        self.deploy = deploy
    
    def forward(self, x):
        if self.deploy:
            return super().forward(x)
        else:
            return x + super().forward(x)

    @torch.no_grad()
    def switch_to_deploy(self):
        """将 Identity 映射融合进卷积权重中"""
        if self.deploy:
            return
        
        kernel_size = self.kernel_size[0]
        center = kernel_size // 2
        
        for i in range(self.out_channels):
            self.weight.data[i, 0, center, center] += 1.0
        
        self.deploy = True


class RepConvBlock(nn.Module):

    def __init__(self, 
                 dim=64,
                 kernel_size=7,
                 mlp_ratio=4,
                 ls_init_value=None,
                 res_scale=False,
                 drop_path=0,
                 norm_layer=LayerNorm2d,
                 use_gemm=False,
                 deploy=False,
                 use_checkpoint=False,
                 use_self_attn=False,
                 num_heads=8):
        super().__init__()
        
        self.res_scale = res_scale
        self.use_checkpoint = use_checkpoint
        self.use_self_attn = use_self_attn
        
        mlp_dim = int(dim*mlp_ratio)
        
        self.dwconv = ResDWConv(dim, kernel_size=3)
    
        self.proj = nn.Sequential(
            norm_layer(dim),
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
            SEModule(dim),
            nn.Conv2d(dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, dim, kernel_size=1),
            DropPath(drop_path) if drop_path > 0 else nn.Identity(),
        )

        self.ls = LayerScale(dim, init_value=ls_init_value, use_bias=True) if ls_init_value is not None else nn.Identity()
        
        # 可选的自注意力模块
        if use_self_attn:
            self.attn = SelfAttention(dim, num_heads=num_heads, qkv_bias=True, attn_drop=0., proj_drop=drop_path, chunk_size=512)
            self.attn_norm = norm_layer(dim)
            self.attn_ls = LayerScale(dim, init_value=ls_init_value, use_bias=True) if ls_init_value is not None else nn.Identity()
            self.attn_drop = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        else:
            self.attn = None
        
    def forward_features(self, x):
        
        x = self.dwconv(x)
        
        if self.attn is not None:
            if self.res_scale:
                x = self.attn_ls(x) + self.attn_drop(self.attn(self.attn_norm(x)))
            else:
                x = x + self.attn_drop(self.attn_ls(self.attn(self.attn_norm(x))))
        
        if self.res_scale:
            x = self.ls(x) + self.proj(x)
        else:
            drop_path = self.proj[-1]
            x = x + drop_path(self.ls(self.proj[:-1](x)))

        return x
    
    def forward(self, x):
        
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self.forward_features, x, use_reentrant=False)
        else:
            x = self.forward_features(x)
        
        return x


class DynamicConvBlock(nn.Module):
    def __init__(self,
                 dim=64,
                 ctx_dim=32,
                 kernel_size=7,
                 smk_size=5,
                 num_heads=2,
                 mlp_ratio=4,
                 ls_init_value=None,
                 res_scale=False,
                 drop_path=0,
                 norm_layer=LayerNorm2d,
                 is_first=False,
                 is_last=False,
                 use_gemm=False,
                 deploy=False,
                 use_checkpoint=False,
                 **kwargs):
        
        super().__init__()
        
        ctx_dim = ctx_dim // 4
        out_dim = dim + ctx_dim
        mlp_dim = int(dim*mlp_ratio)
        self.kernel_size = kernel_size
        self.res_scale = res_scale
        self.use_gemm = use_gemm
        self.smk_size = smk_size
        self.num_heads = num_heads * 2
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.is_first = is_first
        self.is_last = is_last
        self.use_checkpoint = use_checkpoint

        self.deploy = deploy

        if not is_first:
            self.x_scale = LayerScale(ctx_dim, init_value=1, use_bias=True)
            self.h_scale = LayerScale(ctx_dim, init_value=1, use_bias=True)
        
        self.dwconv1 = ResDWConv(out_dim, kernel_size=3, deploy=deploy)
        self.norm1 = norm_layer(out_dim)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, dim, kernel_size=1),
            GRN(dim),
        )
        
        self.weight_query = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim//2),
        )
         
        self.weight_key = nn.Sequential(
            nn.AdaptiveAvgPool2d(7),
            nn.Conv2d(ctx_dim, dim//2, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim//2),
        )
        
        self.weight_proj = nn.Conv2d(49, kernel_size**2 + smk_size**2, kernel_size=1)
        
        self.dyconv_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        
        self.lepe = nn.Sequential(
            DilatedReparamBlock(dim, kernel_size=kernel_size, deploy=deploy, use_sync_bn=False, attempt_use_lk_impl=use_gemm),
            nn.BatchNorm2d(dim),
        )
        
        self.se_layer = SEModule(dim)
        
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )

        self.proj = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, out_dim, kernel_size=1),
        )
        
        self.dwconv2 = ResDWConv(out_dim, kernel_size=3)
        self.norm2 = norm_layer(out_dim)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(out_dim, mlp_dim, kernel_size=1),
            nn.GELU(),
            ResDWConv(mlp_dim, kernel_size=3),
            GRN(mlp_dim),
            nn.Conv2d(mlp_dim, out_dim, kernel_size=1),
        )
        
        self.ls1 = LayerScale(out_dim, init_value=ls_init_value, use_bias=True) if ls_init_value is not None else nn.Identity()
        self.ls2 = LayerScale(out_dim, init_value=ls_init_value, use_bias=True) if ls_init_value is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        self.get_rpb()

        self._precalc_rpb_indices(self.smk_size, '1')
        self._precalc_rpb_indices(self.kernel_size, '2')


    def get_rpb(self):
        self.rpb_size1 = 2 * self.smk_size - 1
        self.rpb1 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size1, self.rpb_size1))
        self.rpb_size2 = 2 * self.kernel_size - 1
        self.rpb2 = nn.Parameter(torch.empty(self.num_heads, self.rpb_size2, self.rpb_size2))
        nn.init.zeros_(self.rpb1)
        nn.init.zeros_(self.rpb2)
    
    def _precalc_rpb_indices(self, k_size, suffix):
        """预计算与 H,W 无关的 RPB 索引，注册为 Buffer 避免重复计算"""
        rpb_size = 2 * k_size - 1
        idx_h = torch.arange(0, k_size)
        idx_w = torch.arange(0, k_size)
        self.register_buffer(f'idx_h_{suffix}', idx_h)
        self.register_buffer(f'idx_w_{suffix}', idx_w)
        idx_k = ((idx_h.unsqueeze(-1) * rpb_size) + idx_w).view(-1)
        self.register_buffer(f'idx_k_{suffix}', idx_k)

    def apply_rpb_optimized(self, attn, rpb, height, width, kernel_size, idx_h, idx_w, idx_k):
        
        num_repeat_h = torch.ones(kernel_size, dtype=torch.long, device=attn.device)
        num_repeat_w = torch.ones(kernel_size, dtype=torch.long, device=attn.device)
        num_repeat_h[kernel_size//2] = height - (kernel_size-1)
        num_repeat_w[kernel_size//2] = width - (kernel_size-1)
        
        bias_hw = (idx_h.repeat_interleave(num_repeat_h).unsqueeze(-1) * (2*kernel_size-1)) + idx_w.repeat_interleave(num_repeat_w)
        bias_idx = bias_hw.unsqueeze(-1) + idx_k
        bias_idx = bias_idx.reshape(-1, int(kernel_size**2))
        bias_idx = torch.flip(bias_idx, [0])
        
        # Gather
        # rpb: (Head, R, R) -> (Head, R*R)
        rpb_flat = rpb.flatten(1, 2)
        rpb_out = rpb_flat[:, bias_idx] # (Head, H*W, K^2)
        
        # Reshape to (1, Head, H, W, K^2) for broadcasting
        rpb_out = rpb_out.reshape(1, self.num_heads, height, width, kernel_size**2)
        
        return attn + rpb_out
    

    def _forward_inner(self, x, h_x, h_r):
        input_resoltion = x.shape[2:]
        B, C, H, W = x.shape
        B, C_h, H_h, W_h = h_x.shape
        
        if not self.is_first:
            h_x = self.x_scale(h_x) + self.h_scale(h_r)

        x_f = torch.cat([x, h_x], dim=1)
        x_f = self.dwconv1(x_f)
        identity = x_f
        x_f_norm = self.norm1(x_f)
        x_fused = self.fusion(x_f_norm)
        del x_f_norm
        
        gate = self.gate(x_fused)
        lepe = self.lepe(x_fused)
        
        is_pad = False
        if min(H, W) < self.kernel_size:
            is_pad = True
            if H < W:
                size = (self.kernel_size, int(self.kernel_size / H * W))
            else:
                size = (int(self.kernel_size / W * H), self.kernel_size)
                
            x_fused_rs = F.interpolate(x_fused, size=size, mode='bilinear', align_corners=False)
            x_f_rs = F.interpolate(x_f, size=size, mode='bilinear', align_corners=False)
            H_real, W_real = H, W
            H, W = size
        else:
            x_fused_rs = x_fused
            x_f_rs = x_f

        q_feat, k_feat = torch.split(x_f_rs, split_size_or_sections=[C, C_h], dim=1)
        del x_f_rs
        
        query = self.weight_query(q_feat) * self.scale
        key = self.weight_key(k_feat)
        del q_feat, k_feat

        query = rearrange(query, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        key = rearrange(key, 'b (g c) h w -> b g c (h w)', g=self.num_heads)
        weight = einsum(query, key, 'b g c n, b g c l -> b g n l')
        del query, key
        
        weight = rearrange(weight, 'b g n l -> b l g n').contiguous()
        weight = self.weight_proj(weight)
        weight = rearrange(weight, 'b l g (h w) -> b g h w l', h=H, w=W)

        attn1, attn2 = torch.split(weight, split_size_or_sections=[self.smk_size**2, self.kernel_size**2], dim=-1)
        del weight
        
        attn1 = self.apply_rpb_optimized(attn1, self.rpb1, H, W, self.smk_size, self.idx_h_1, self.idx_w_1, self.idx_k_1)
        attn2 = self.apply_rpb_optimized(attn2, self.rpb2, H, W, self.kernel_size, self.idx_h_2, self.idx_w_2, self.idx_k_2)
        attn1 = torch.softmax(attn1, dim=-1)
        attn2 = torch.softmax(attn2, dim=-1)
        value = rearrange(x_fused_rs, 'b (m g c) h w -> m b g h w c', m=2, g=self.num_heads)
        
        del x_fused_rs

        x1 = na2d_av(attn1, value[0], kernel_size=self.smk_size)
        del attn1
        x2 = na2d_av(attn2, value[1], kernel_size=self.kernel_size)
        del attn2, value

        x_out = torch.cat([x1, x2], dim=1)
        del x1, x2
        x_out = rearrange(x_out, 'b g h w c -> b (g c) h w', h=H, w=W)
        
        if is_pad:
            x_out = F.adaptive_avg_pool2d(x_out, input_resoltion)
        
        x_out = self.dyconv_proj(x_out)
        x_out.add_(lepe)
        del lepe
        
        x_out = self.se_layer(x_out)
        x_out = gate * x_out
        del gate
        
        x_out = self.proj(x_out)

        if self.res_scale:
            x_out = self.ls1(identity) + self.drop_path(x_out)
        else:
            # x_out = identity + self.drop_path(self.ls1(x_out))
            scaled_out = self.ls1(x_out)
            x_out = identity + self.drop_path(scaled_out)
        
        identity2 = x_out
        x_out = self.dwconv2(x_out)

        x_mlp = self.norm2(x_out)
        x_mlp = self.mlp(x_mlp)
         
        if self.res_scale:
            x_out = self.ls2(x_out) + self.drop_path(x_mlp)
        else:
            x_out = x_out + self.drop_path(self.ls2(x_mlp))

        if self.is_last:
            return (x_out, None)
        else:
            l_x, h_x = torch.split(x_out, split_size_or_sections=[C, C_h], dim=1)
            return (l_x, h_x)
    
    def forward(self, x, h_x, h_r):
        if self.use_checkpoint and x.requires_grad:
            x = checkpoint(self._forward_inner, x, h_x, h_r, use_reentrant=False)
        else:
            x = self._forward_inner(x, h_x, h_r)
        return x

    def switch_to_deploy(self):
        """支持递归切换到部署模式"""
        if self.deploy:
            return
        
        # 融合所有内部的 ResDWConv
        for m in [self.dwconv1, self.dwconv2, self.mlp_dw]:
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
                
        # 融合 LEPE (DilatedReparamBlock)
        for m in self.lepe:
            if hasattr(m, 'merge_dilated_branches'):
                m.merge_dilated_branches()
        
        self.deploy = True


class OverLoCK(nn.Module):
    '''
    An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels
    https://arxiv.org/abs/2502.20087
    '''
    def __init__(self, 
                 depth=[2, 2, 2, 2],
                 sub_depth=[4, 2],
                 in_chans=3, 
                 embed_dim=[96, 192, 384, 768],
                 kernel_size=[7, 7, 7, 7],
                 mlp_ratio=[4, 4, 4, 4],
                 sub_mlp_ratio=[4, 4],
                 sub_num_heads=[4, 8],
                 ls_init_value=[None, None, 1, 1],
                 res_scale=True,
                 smk_size=5,
                 deploy=False,
                 use_gemm=True,
                 use_ds=True,
                 drop_rate=0,
                 drop_path_rate=0,
                 norm_layer=LayerNorm2d,
                 projection=1024,
                 num_classes=1000,
                 use_checkpoint=[1, 1, 1, 1],
                 use_self_attn=[False, False, False, False],
                 attn_num_heads=[4, 4, 8, 8],  # 每个stage的注意力头数
                 attn_chunk_size=256,  # 自注意力分块大小

            ):
 
        super().__init__()
        self.attn_chunk_size = attn_chunk_size
        
        fusion_dim = embed_dim[-1] + embed_dim[-1]//4
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed1 = stem(in_chans, embed_dim[0])
        self.patch_embed2 = downsample(embed_dim[0], embed_dim[1])
        self.patch_embed3 = downsample(embed_dim[1], embed_dim[2])
        self.patch_embed4 = downsample(embed_dim[2], embed_dim[3])
        self.high_level_proj = nn.Conv2d(embed_dim[-1], embed_dim[-1]//4, kernel_size=1)
        self.patch_embedx = CTXDownsample(embed_dim[2], embed_dim[3])
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth) + sum(sub_depth))]

        self.blocks1 = nn.ModuleList()
        self.blocks2 = nn.ModuleList()
        self.blocks3 = nn.ModuleList()
        self.blocks4 = nn.ModuleList()
        self.sub_blocks3 = nn.ModuleList()
        self.sub_blocks4 = nn.ModuleList()
        
        for i in range(depth[0]):
            self.blocks1.append(
                RepConvBlock(
                    dim=embed_dim[0],
                    kernel_size=kernel_size[0],
                    mlp_ratio=mlp_ratio[0],
                    ls_init_value=ls_init_value[0],
                    res_scale=res_scale,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[0]),
                    use_self_attn=use_self_attn[0],
                    num_heads=attn_num_heads[0],
                )
            )
        
        for i in range(depth[1]):
            self.blocks2.append(
                RepConvBlock(
                    dim=embed_dim[1],
                    kernel_size=kernel_size[1],
                    mlp_ratio=mlp_ratio[1],
                    ls_init_value=ls_init_value[1],
                    res_scale=res_scale,
                    drop_path=dpr[i+depth[0]],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[1]),
                    use_self_attn=use_self_attn[1],
                    num_heads=attn_num_heads[1],
                )
            )
            
        for i in range(depth[2]):
            self.blocks3.append(
                RepConvBlock(
                    dim=embed_dim[2],
                    kernel_size=kernel_size[2],
                    mlp_ratio=mlp_ratio[2],
                    ls_init_value=ls_init_value[2],
                    res_scale=res_scale,
                    drop_path=dpr[i+sum(depth[:2])],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[2]),
                    use_self_attn=use_self_attn[2],
                    num_heads=attn_num_heads[2],
                )
            )

        for i in range(depth[3]):
            self.blocks4.append(
                RepConvBlock(
                    dim=embed_dim[3],
                    kernel_size=kernel_size[3],
                    mlp_ratio=mlp_ratio[3],
                    ls_init_value=ls_init_value[3],
                    res_scale=res_scale,
                    drop_path=dpr[i+sum(depth[:3])],
                    norm_layer=norm_layer,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[3]),
                    use_self_attn=use_self_attn[3],
                    num_heads=attn_num_heads[3],
                )
            )
            
        for i in range(sub_depth[0]):
            self.sub_blocks3.append(
                DynamicConvBlock(
                    dim=embed_dim[2],
                    ctx_dim=embed_dim[-1],
                    kernel_size=kernel_size[2],
                    num_heads=sub_num_heads[0],
                    pool_size=7,
                    mlp_ratio=sub_mlp_ratio[0],
                    ls_init_value=ls_init_value[2],
                    res_scale=res_scale,
                    drop_path=dpr[i+sum(depth)],
                    norm_layer=norm_layer,
                    smk_size=smk_size,
                    use_gemm=use_gemm,
                    deploy=deploy,
                    is_first=(i==0),
                    use_checkpoint=(i<use_checkpoint[2]),
                )
            )
        
        for i in range(sub_depth[1]):
            self.sub_blocks4.append(
                DynamicConvBlock(
                    dim=embed_dim[3],
                    ctx_dim=embed_dim[-1],
                    kernel_size=kernel_size[-1],
                    num_heads=sub_num_heads[1],
                    pool_size=7,
                    mlp_ratio=sub_mlp_ratio[1],
                    ls_init_value=ls_init_value[3],
                    res_scale=res_scale,
                    drop_path=dpr[i+sum(depth)+sub_depth[0]],
                    norm_layer=norm_layer,
                    smk_size=smk_size,
                    is_first=False,
                    is_last=(i==sub_depth[1]-1),
                    use_gemm=use_gemm,
                    deploy=deploy,
                    use_checkpoint=(i<use_checkpoint[3]),
                )
            )

        self.h_proj = nn.Sequential(
            nn.Conv2d(embed_dim[-1], fusion_dim, kernel_size=1),
            LayerScale(fusion_dim, init_value=1e-5, use_bias=True),
        )
        
        # Aux Cls Head
        if use_ds:
            self.aux_head = nn.Sequential(
                nn.BatchNorm2d(embed_dim[-1]),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(embed_dim[-1], num_classes, kernel_size=1) if num_classes > 0 else nn.Identity()
            )
        
        # Main Cls Head
        self.head = nn.Sequential(
            nn.Conv2d(fusion_dim, projection, kernel_size=1, bias=False),
            nn.BatchNorm2d(projection),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(projection, num_classes, kernel_size=1) if num_classes > 0 else nn.Identity()
        )
        
        self.extra_norm = nn.ModuleList()
        for idx in range(4):
            dim = embed_dim[idx]
            if idx >= 2:
                dim = dim + embed_dim[-1]//4
            self.extra_norm.append(norm_layer(dim))
        self.extra_norm.append(norm_layer(embed_dim[-1]))
        
        del self.aux_head
        del self.head
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    def _convert_sync_batchnorm(self):
        if torch.distributed.is_initialized():
            self = nn.SyncBatchNorm.convert_sync_batchnorm(self)
            
    def forward_pre_features(self, x):
        
        outs = []
        
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x)
        
        outs.append(self.extra_norm[0](x))
            
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)

        outs.append(self.extra_norm[1](x))
        
        return outs
    
    
    def forward_base_features(self, x):
        
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
            
        ctx = self.patch_embed4(x)
        for blk in self.blocks4:
            ctx = blk(ctx)

        return (x, ctx)
    

    def forward_sub_features(self, x, ctx):
        
        outs = []
        
        ctx_cls = ctx
        ctx_ori = self.high_level_proj(ctx)
        ctx_up = F.interpolate(ctx_ori, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        for idx, blk in enumerate(self.sub_blocks3):
            if idx == 0:
                ctx = ctx_up
            x, ctx = blk(x, ctx, ctx_up)
        
        x_cat = torch.cat([x, ctx], dim=1)
        outs.append(self.extra_norm[2](x_cat))
        del x_cat
        
        x, ctx = self.patch_embedx(x, ctx)
        for idx, blk in enumerate(self.sub_blocks4):
            x, ctx = blk(x, ctx, ctx_ori)
        
        ctx_norm = self.extra_norm[-1](ctx_cls)
        del ctx_cls
        h_proj = self.h_proj(ctx_norm)
        del ctx_norm
        x = self.extra_norm[3](x) + h_proj
        del h_proj
        
        outs.append(x)
        
        return outs

    def forward_features(self, x):
        
        x0, x1 = self.forward_pre_features(x)
        x, ctx = self.forward_base_features(x1)
        x2, x3 = self.forward_sub_features(x, ctx)

        return (x0, x1, x2, x3)

    def forward(self, x):
        
        x = self.forward_features(x)
        
        return x

    def switch_to_deploy(self):
        """
        将模型切换到部署模式，融合所有的重参数化模块 (DilatedReparamBlock, ResDWConv 等)
        """
        for m in self.modules():
            # 1. 处理 ResDWConv 的融合
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
            
            # 2. 处理 DilatedReparamBlock 的融合
            if hasattr(m, 'merge_dilated_branches'):
                m.merge_dilated_branches()
                
        print("OverLoCK model switched to deploy mode (fused weights).")


@MODELS.register_module()
def overlock_xt(pretrained=False, pretrained_cfg=None, **kwargs):
    
    model = OverLoCK(
        depth=[2, 2, 3, 2],
        sub_depth=[6, 2],
        embed_dim=[56, 112, 256, 336],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4, 4, 4, 4],
        sub_num_heads=[4, 6],
        sub_mlp_ratio=[3, 3],
        **kwargs
    )

    if pretrained:
        pretrained = 'https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_xt_in1k_224.pth'
        logger = get_root_logger()
        load_checkpoint(model, pretrained, logger=logger)
    model._convert_sync_batchnorm()
    return model


@MODELS.register_module()
def overlock_t(pretrained=True, pretrained_cfg=None, **kwargs):
    
    model = OverLoCK(
        depth=[4, 4, 6, 2],
        sub_depth=[12, 2],
        embed_dim=[64, 128, 256, 512],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4, 4, 4, 4],
        sub_num_heads=[4, 8],
        sub_mlp_ratio=[3, 3],
        use_checkpoint=[1, 1, 0, 0],
        **kwargs
    )

    if pretrained:
        pretrained = 'https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_t_in1k_224.pth'
        logger = get_root_logger()
        load_checkpoint(model, pretrained, logger=logger)
    model._convert_sync_batchnorm()
    return model


@MODELS.register_module()
def overlock_s(pretrained=False, pretrained_cfg=None, **kwargs):
    
    model = OverLoCK(
        depth=[6, 6, 8, 3],
        sub_depth=[16, 3],
        embed_dim=[64, 128, 320, 512],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4, 4, 4, 4],
        sub_num_heads=[8, 16],
        sub_mlp_ratio=[3, 3],
        **kwargs
    )

    if pretrained:
        pretrained = 'https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_s_in1k_224.pth'
        logger = get_root_logger()
        load_checkpoint(model, pretrained, logger=logger)
    model._convert_sync_batchnorm()
    return model


@MODELS.register_module()
def overlock_b(pretrained=None, pretrained_cfg=None, **kwargs):
    
    model = OverLoCK(
        depth=[8, 8, 10, 4],
        sub_depth=[20, 4],
        embed_dim=[80, 160, 384, 576],
        kernel_size=[17, 15, 13, 7],
        mlp_ratio=[4, 4, 4, 4],
        sub_num_heads=[6, 9],
        sub_mlp_ratio=[3, 3],
        **kwargs
    )

    if pretrained:
        pretrained = 'https://github.com/LMMMEng/OverLoCK/releases/download/v1/overlock_b_in1k_224.pth'
        logger = get_root_logger()
        load_checkpoint(model, pretrained, logger=logger)
    model._convert_sync_batchnorm()
    return model
