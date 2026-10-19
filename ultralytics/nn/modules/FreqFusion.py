import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import warnings
import numpy as np

# from mmcv.ops.carafe import normal_init, xavier_init, carafe
# try:
#     from mmcv.ops.carafe import normal_init, xavier_init, carafe
# except ImportError:

def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def carafe(x, normed_mask, kernel_size,up,down):
    b, c, h, w = x.shape
    _, m_c, m_h, m_w = normed_mask.shape
    pad = kernel_size // 2
    pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
    unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
    unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
    unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)
    normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
    res = unfold_x * normed_mask
    res = res.sum(dim=2).reshape(b, c, m_h, m_w)
    return res

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)

def hamming2D(M, N):
    """
    生成二维Hamming窗

    参数：
    - M：窗口的行数
    - N：窗口的列数

    返回：
    - 二维Hamming窗
    """
    # 生成水平和垂直方向上的Hamming窗
    # hamming_x = np.blackman(M)
    # hamming_x = np.kaiser(M)
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    # 通过外积生成二维Hamming窗
    hamming_2d = np.outer(hamming_x, hamming_y)
    return hamming_2d



class FreqFusion6(nn.Module):
    def __init__(self,
                 hr_channels,
                 lr_channels,
                 scale_factor=1,
                 lowpass_kernel=5,
                 highpass_kernel=3,
                 up_group=1,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 compressed_channels=64,
                 align_corners=False,
                 upsample_mode='nearest',
                 feature_resample=True,  # use offset generator or not
                 feature_resample_group=4,
                 comp_feat_upsample=True,  # use ALPF & AHPF for init upsampling
                 use_high_pass=True,
                 use_low_pass=True,
                 hr_residual=True,
                 semi_conv=True,
                 hamming_window=True,  # for regularization, do not matter really
                 feature_resample_norm=True,
                 **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)
        self.content_encoder_l_rgb = nn.Conv2d(  # ALPF generator1
            self.compressed_channels,
            lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        self.content_encoder_l_ir = nn.Conv2d(  # ALPF generator2
            self.compressed_channels,
            lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass

        self.comp_feat_upsample = comp_feat_upsample
        
        if self.use_high_pass:
            self.content_encoder_h_rgb = nn.Conv2d(  # AHPF generator
                self.compressed_channels,
                highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
            self.content_encoder_h_ir = nn.Conv2d(  # AHPF generator
                self.compressed_channels,
                highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
        self.hamming_window = hamming_window
        lowpass_pad = 0
        highpass_pad = 0
        if self.hamming_window:
            self.register_buffer('hamming_lowpass', torch.FloatTensor(
                hamming2D(lowpass_kernel + 2 * lowpass_pad, lowpass_kernel + 2 * lowpass_pad))[None, None,])
            self.register_buffer('hamming_highpass', torch.FloatTensor(
                hamming2D(highpass_kernel + 2 * highpass_pad, highpass_kernel + 2 * highpass_pad))[None, None,])
        else:
            self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
            self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder_l_ir, std=0.001)
        normal_init(self.content_encoder_l_rgb, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoder_h_ir, std=0.001)
            normal_init(self.content_encoder_h_rgb, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel ** 2))
        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask


    def forward(self, input):

        rgb_feat, ir_feat = input[0], input[1]

        # =  step1  压缩特征
        compressed_rgb_feat = self.hr_channel_compressor(rgb_feat).to(rgb_feat.dtype)
        compressed_ir_feat = self.lr_channel_compressor(ir_feat).to(rgb_feat.dtype)

        #= 对 rgb 特征高频自增强
        rgb_hf_feat = self.content_encoder_h_rgb(compressed_rgb_feat)
        rgb_hf_feat_init = self.kernel_normalizer(rgb_hf_feat, self.highpass_kernel,hamming=self.hamming_highpass).to(rgb_feat.dtype)
        compressed_rgb_feat = compressed_rgb_feat + compressed_rgb_feat - carafe(compressed_rgb_feat, rgb_hf_feat_init,self.highpass_kernel,1,1)

        #= 利用红外低频去平滑可见光的低频，减少类内不一致
        ir_lf_feat = self.content_encoder_l_ir(compressed_ir_feat)
        rgb_lf_feat = self.content_encoder_l_rgb(compressed_rgb_feat)
        ir_lf_feat_init = self.kernel_normalizer(ir_lf_feat, self.lowpass_kernel, hamming=self.hamming_lowpass).to(rgb_feat.dtype)
        enhance_rgb_lf_feat = carafe(rgb_lf_feat, ir_lf_feat_init, self.lowpass_kernel,1,1)
        fusion_lf_feat = enhance_rgb_lf_feat  + ir_lf_feat #= 红外低频引导可见光，得到混合低频特征

        # = rgb-ir混合低通滤波器特征  初始化  可见光高频引导红外高频
        ir_hf_feat = self.content_encoder_h_ir(compressed_ir_feat)
        fusion_lf_feat_init = self.kernel_normalizer(fusion_lf_feat, self.lowpass_kernel, hamming=self.hamming_lowpass).to(rgb_feat.dtype)
        enhance_lr_hf_feat = carafe(ir_hf_feat, fusion_lf_feat_init , self.lowpass_kernel,1,1)
        fusion_hf_feat = enhance_lr_hf_feat + rgb_hf_feat

        # =  最终处理
        ir_feat_final = carafe(ir_feat, fusion_lf_feat_init, self.lowpass_kernel,1,1)
        fusion_hf_feat_init = self.kernel_normalizer(fusion_hf_feat, self.highpass_kernel,hamming=self.hamming_highpass).to(rgb_feat.dtype)
        rgb_feat_final = rgb_feat + rgb_feat - carafe(rgb_feat, fusion_hf_feat_init, self.highpass_kernel,1,1)

        return rgb_feat_final, ir_feat_final

#======   CFM   ======#
class Fusion1(nn.Module):   
    def __init__(self,
                 hr_channels,
                 lr_channels,
                 scale_factor=1,
                 lowpass_kernel=5,
                 highpass_kernel=3,
                 up_group=1,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 compressed_channels=64,
                 align_corners=False,
                 upsample_mode='nearest',
                 comp_feat_upsample=True,  # use ALPF & AHPF for init upsampling
                 use_high_pass=True,
                 use_low_pass=True,
                 hr_residual=True,
                 hamming_window=True,  # for regularization, do not matter really
                 **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)
        self.content_encoder_l_rgb = nn.Conv2d(  # ALPF generator1
            self.compressed_channels,
            lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        self.content_encoder_l_ir = nn.Conv2d(  # ALPF generator2
            self.compressed_channels,
            lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass

        self.comp_feat_upsample = comp_feat_upsample
        
        if self.use_high_pass:
            self.content_encoder_h_rgb = nn.Conv2d(  # AHPF generator
                self.compressed_channels,
                highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
            self.content_encoder_h_ir = nn.Conv2d(  # AHPF generator
                self.compressed_channels,
                highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
        self.hamming_window = hamming_window
        lowpass_pad = 0
        highpass_pad = 0
        if self.hamming_window:
            self.register_buffer('hamming_lowpass', torch.FloatTensor(
                hamming2D(lowpass_kernel + 2 * lowpass_pad, lowpass_kernel + 2 * lowpass_pad))[None, None,])
            self.register_buffer('hamming_highpass', torch.FloatTensor(
                hamming2D(highpass_kernel + 2 * highpass_pad, highpass_kernel + 2 * highpass_pad))[None, None,])
        else:
            self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
            self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder_l_ir, std=0.001)
        normal_init(self.content_encoder_l_rgb, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoder_h_ir, std=0.001)
            normal_init(self.content_encoder_h_rgb, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel ** 2))
        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

    def forward(self, input):

        rgb_feat, ir_feat = input[0], input[1]

        # =  step1  压缩特征
        compressed_rgb_feat = self.hr_channel_compressor(rgb_feat) 
        compressed_ir_feat = self.lr_channel_compressor(ir_feat)

        #= 对 rgb 特征高频自增强
        rgb_hf_feat = self.content_encoder_h_rgb(compressed_rgb_feat)
        rgb_hf_feat_init = self.kernel_normalizer(rgb_hf_feat, self.highpass_kernel,hamming=self.hamming_highpass)
        compressed_rgb_feat = compressed_rgb_feat + compressed_rgb_feat - carafe(compressed_rgb_feat, rgb_hf_feat_init,self.highpass_kernel,1,1)

        #= 利用红外低频去平滑可见光的低频，减少类内不一致
        ir_lf_feat = self.content_encoder_l_ir(compressed_ir_feat)
        rgb_lf_feat = self.content_encoder_l_rgb(compressed_rgb_feat)
        ir_lf_feat_init = self.kernel_normalizer(ir_lf_feat, self.lowpass_kernel, hamming=self.hamming_lowpass)
        enhance_rgb_lf_feat = carafe(rgb_lf_feat, ir_lf_feat_init, self.lowpass_kernel,1,1)
        fusion_lf_feat = enhance_rgb_lf_feat  + ir_lf_feat   #= 红外低频引导可见光，得到混合低频特征

        # = rgb-ir混合低通滤波器特征  初始化  可见光高频引导红外高频
        ir_hf_feat = self.content_encoder_h_ir(compressed_ir_feat)
        fusion_lf_feat_init = self.kernel_normalizer(fusion_lf_feat, self.lowpass_kernel, hamming=self.hamming_lowpass)
        enhance_lr_hf_feat = carafe(ir_hf_feat, fusion_lf_feat_init , self.lowpass_kernel,1,1)
        fusion_hf_feat = enhance_lr_hf_feat + rgb_hf_feat

        # =  最终处理
        ir_feat_final = carafe(ir_feat, fusion_lf_feat_init, self.lowpass_kernel,1,1)
        fusion_hf_feat_init = self.kernel_normalizer(fusion_hf_feat, self.highpass_kernel,hamming=self.hamming_highpass)
        rgb_feat_final = rgb_feat + rgb_feat - carafe(rgb_feat, fusion_hf_feat_init, self.highpass_kernel,1,1)

        return rgb_feat_final, ir_feat_final

