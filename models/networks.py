import torch
import logging
import models.modules.discriminator_vgg_arch as SRGAN_arch
from models.modules.Inv_arch import *
from models.modules.Subnet_constructor import subnet
import math
logger = logging.getLogger('base')


####################
# define network
####################
def define_G(opt, subnet_type, block_num):
    """
    Hint:
    subnet_type DBNet
    block_num [4, 4, 4, 2]
    """
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    # subnet_type = which_model['subnet_type'] # default DBNet
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'

    down_num = int(math.log(opt_net['scale'], 2))

    netG = InvRescaleNet(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), block_num, down_num)

    return netG


#### Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD

def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation,with_attn=False):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        if self.with_attn:
            return out ,attention
        else:
            return out

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='kaiming', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class NormalGenerator(nn.Module):
    def __init__(self, dims_in=[[3, 64, 64]], down_num=3, block_num=[4,4,4],out_channel=3):
        super(NormalGenerator, self).__init__()
        self.out_channel = out_channel
        operations = []

        current_dims = dims_in
        for i in range(down_num):
            b = HaarDownsampling(current_dims)
            # b = Squeeze()
            operations.append(b)
            current_dims[0][0] = current_dims[0][0] * 4
            current_dims[0][1] = current_dims[0][1] // 2
            current_dims[0][2] = current_dims[0][2] // 2
            for j in range(block_num[i]):
                b = ResBlock(current_dims[0][0],current_dims[0][0])
                # b = RNVPCouplingBlock(current_dims, subnet_constructor=DenseBlock, clamp=1.0)
                operations.append(b)
        block_num = block_num[:-1][::-1]
        block_num.append(0)
        for i in range(down_num):
            b = HaarUpsampling(current_dims)
            # b = Unsqueeze()
            operations.append(b)
            current_dims[0][0] = current_dims[0][0] // 4
            current_dims[0][1] = current_dims[0][1] * 2
            current_dims[0][2] = current_dims[0][2] * 2
            for j in range(block_num[i]):
                b = ResBlock(current_dims[0][0],current_dims[0][0])
                operations.append(b)

        # self.out_layer = nn.Conv2d(current_dims[0][0], out_channel, 1,1,0)
        # operations.append(self.out_layer)
        self.operations = nn.ModuleList(operations)
        # self.guassianize = Gaussianize(1)

    def forward(self, x):
        out = x

        for op in self.operations:
            out = op.forward(out)
        out = out[:, :self.out_channel, :, :]
        return out

# class Extractor(BaseNetwork):
#     def __init__(self, in_channels=3, out_channels=3, residual_blocks=8, init_weights=True):
#         super(Extractor, self).__init__()
#         dim=32
#         self.encoder_0 = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=7, padding=0,bias=False),
#             nn.BatchNorm2d(dim, affine=True),
#             # nn.InstanceNorm2d(dim),
#             nn.GELU(),
#             nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1,bias=False),
#             # nn.InstanceNorm2d(64),
#             nn.BatchNorm2d(dim, affine=True),
#             nn.GELU(),
#         )
#         self.encoder_1 = nn.Sequential(
#
#             nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=4, stride=2, padding=1,bias=False),
#             nn.BatchNorm2d(dim*2, affine=True),
#             # nn.InstanceNorm2d(dim*2),
#             nn.GELU(),
#             nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, padding=1, bias=False),
#             # nn.InstanceNorm2d(dim*2),
#             nn.BatchNorm2d(dim*2, affine=True),
#             nn.GELU(),
#         )
#         self.encoder_2 = nn.Sequential(
#             nn.Conv2d(in_channels=dim*2, out_channels=dim*4, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(dim*4, affine=True),
#             # nn.InstanceNorm2d(dim*4),
#             nn.GELU(),
#             nn.Conv2d(in_channels=dim*4, out_channels=dim*4, kernel_size=3, padding=1, bias=False),
#             # nn.InstanceNorm2d(dim*4),
#             nn.BatchNorm2d(dim*4, affine=True),
#             nn.GELU()
#         )
#
#         blocks = []
#         for _ in range(residual_blocks): #residual_blocks
#             block = ResnetBlock(dim*4, dilation=2, use_spectral_norm=False)
#             blocks.append(block)
#
#         self.middle = nn.Sequential(*blocks)
#
#         self.decoder_2 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=dim*4*2, out_channels=dim*2, kernel_size=4, stride=2, padding=1,bias=False),
#             nn.BatchNorm2d(dim*2, affine=True),
#             # nn.InstanceNorm2d(dim*2),
#             nn.GELU(),
#             nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, padding=1, bias=False),
#             # nn.InstanceNorm2d(dim*2),
#             nn.BatchNorm2d(dim*2, affine=True),
#             nn.GELU(),
#         )
#
#         self.decoder_1 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=dim*2*2, out_channels=dim, kernel_size=4, stride=2, padding=1,bias=False),
#             nn.BatchNorm2d(dim, affine=True),
#             # nn.InstanceNorm2d(dim),
#             nn.GELU(),
#             nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False),
#             # nn.InstanceNorm2d(dim),
#             nn.BatchNorm2d(dim, affine=True),
#             nn.GELU(),
#         )
#
#         self.decoder_0 = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_channels=dim*2, out_channels=out_channels, kernel_size=7, padding=0),
#         )
#
#         if init_weights:
#             self.init_weights()
#
#     def forward(self, x):
#         e0 = self.encoder_0(x)
#         e1 = self.encoder_1(e0)
#         e2 = self.encoder_2(e1)
#         m = self.middle(e2)
#         d2 = self.decoder_2(torch.cat((e2,m),dim=1))
#         d1 = self.decoder_1(torch.cat((e1,d2),dim=1))
#         x = self.decoder_0(torch.cat((e0, d1), dim=1))
#         x = (torch.tanh(x) + 1) / 2
#
#         return x

class InpaintGenerator(BaseNetwork):
    def __init__(self, in_channels=3, out_channels=3, residual_blocks=8, init_weights=True):
        super(InpaintGenerator, self).__init__()
        dim=16
        self.encoder_0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=7, padding=0,bias=False),
            # nn.BatchNorm2d(dim, affine=True),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1,bias=False),
            nn.InstanceNorm2d(64),
            # nn.BatchNorm2d(dim, affine=True),
            nn.GELU(),
        )
        self.encoder_1 = nn.Sequential(

            nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=4, stride=2, padding=1,bias=False),
            # nn.BatchNorm2d(dim*2, affine=True),
            nn.InstanceNorm2d(dim*2),
            nn.GELU(),
            nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim*2),
            # nn.BatchNorm2d(dim*2, affine=True),
            nn.GELU(),
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(in_channels=dim*2, out_channels=dim*4, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(dim*4, affine=True),
            nn.InstanceNorm2d(dim*4),
            nn.GELU(),
            nn.Conv2d(in_channels=dim*4, out_channels=dim*4, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim*4),
            # nn.BatchNorm2d(dim*4, affine=True),
            nn.GELU()
        )

        blocks = []
        for _ in range(residual_blocks): #residual_blocks
            block = ResnetBlock(dim*4, dilation=2, use_spectral_norm=False)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim*4*2, out_channels=dim*2, kernel_size=4, stride=2, padding=1,bias=False),
            # nn.BatchNorm2d(dim*2, affine=True),
            nn.InstanceNorm2d(dim*2),
            nn.GELU(),
            nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim*2),
            # nn.BatchNorm2d(dim*2, affine=True),
            nn.GELU(),
        )

        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim*2*2, out_channels=dim, kernel_size=4, stride=2, padding=1,bias=False),
            # nn.BatchNorm2d(dim, affine=True),
            nn.InstanceNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(dim),
            # nn.BatchNorm2d(dim, affine=True),
            nn.GELU(),
        )

        self.decoder_0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=dim*2, out_channels=out_channels, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        e0 = self.encoder_0(x)
        e1 = self.encoder_1(e0)
        e2 = self.encoder_2(e1)
        m = self.middle(e2)
        d2 = self.decoder_2(torch.cat((e2,m),dim=1))
        d1 = self.decoder_1(torch.cat((e1,d2),dim=1))
        x = self.decoder_0(torch.cat((e0, d1), dim=1))
        # x = (torch.tanh(x) + 1) / 2

        return x


class HaarDownsampling(nn.Module):
    '''Uses Haar wavelets to split each channel into 4 channels, with half the
    width and height.'''

    def __init__(self, dims_in, order_by_wavelet=False, rebalance=1.):
        super().__init__()

        self.in_channels = dims_in[0][0]
        self.fac_fwd = 0.5 * rebalance
        self.fac_rev = 0.5 / rebalance
        self.haar_weights = torch.ones(4,1,2,2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights]*self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

        self.permute = order_by_wavelet
        self.last_jac = None

        if self.permute:
            permutation = []
            for i in range(4):
                permutation += [i+4*j for j in range(self.in_channels)]

            self.perm = torch.LongTensor(permutation)
            self.perm_inv = torch.LongTensor(permutation)

            for i, p in enumerate(self.perm):
                self.perm_inv[p] = i

    def forward(self, x, rev=False):
        if not rev:
            # self.last_jac = self.elements / 4 * (np.log(16.) + 4 * np.log(self.fac_fwd))
            out = F.conv2d(x, self.haar_weights,
                           bias=None, stride=2, groups=self.in_channels)
            if self.permute:
                return out[:, self.perm] * self.fac_fwd
            else:
                return out * self.fac_fwd

        else:
            # self.last_jac = self.elements / 4 * (np.log(16.) + 4 * np.log(self.fac_rev))
            if self.permute:
                x_perm = x[:, self.perm_inv]
            else:
                x_perm = x

            return F.conv_transpose2d(x_perm * self.fac_rev, self.haar_weights,
                                     bias=None, stride=2, groups=self.in_channels)

    def jacobian(self, x, rev=False):
        # TODO respect batch dimension and .cuda()
        return self.last_jac

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        c2, w2, h2 = c*4, w//2, h//2
        self.elements = c*w*h
        assert c*h*w == c2*h2*w2, "Uneven input dimensions"
        return (c2, w2, h2)


class HaarUpsampling(nn.Module):
    '''Uses Haar wavelets to merge 4 channels into one, with double the
    width and height.'''

    def __init__(self, dims_in):
        super().__init__()

        self.in_channels = dims_in[0][0] // 4
        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights *= 0.5
        self.haar_weights = torch.cat([self.haar_weights]*self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if rev:
            return F.conv2d(x, self.haar_weights,
                             bias=None, stride=2, groups=self.in_channels)
        else:
            return F.conv_transpose2d(x, self.haar_weights,
                                       bias=None, stride=2,
                                       groups=self.in_channels)

    def jacobian(self, x, rev=False):
        # TODO respect batch dimension and .cuda()
        return 0

    def output_dims(self, input_dims):
        assert len(input_dims) == 1, "Can only use 1 input"
        c, w, h = input_dims[0]
        c2, w2, h2 = c//4, w*2, h*2
        assert c*h*w == c2*h2*w2, "Uneven input dimensions"
        return (c2, w2, h2)

class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out, use_spectral_norm=False):
        super(ResBlock, self).__init__()
        feature = channel_in
        if not use_spectral_norm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(channel_in, feature, kernel_size=3, padding=1,bias=False),
                nn.InstanceNorm2d(feature),
                # nn.BatchNorm2d(feature, affine=True),
                nn.GELU(),
            )
        else:
            self.conv_1 = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels=channel_in, out_channels=feature, kernel_size=3, padding=1,
                                        bias=not use_spectral_norm), use_spectral_norm),
                nn.GELU(),
            )

        if not use_spectral_norm:
            self.conv2 = nn.Sequential(
                nn.Conv2d(feature, feature, kernel_size=3, padding=1,bias=False),
                nn.InstanceNorm2d(feature),
                # nn.BatchNorm2d(feature, affine=True),
                nn.GELU(),
            )
        else:
            self.conv_2 = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels=feature, out_channels=feature, kernel_size=3, padding=1,
                                        bias=not use_spectral_norm), use_spectral_norm),
                nn.GELU(),
            )

        if not use_spectral_norm:
            self.conv3 = nn.Sequential(
                nn.Conv2d(feature, feature, kernel_size=3, padding=1,bias=False),
                nn.InstanceNorm2d(feature),
                # nn.BatchNorm2d(feature, affine=True),
                nn.GELU(),
            )
        else:
            self.conv_3 = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels=feature, out_channels=feature, kernel_size=3, padding=1,
                                        bias=not use_spectral_norm), use_spectral_norm),
                nn.GELU(),
            )

        if not use_spectral_norm:
            self.conv4 = nn.Sequential(
                nn.Conv2d(feature, feature, kernel_size=3, padding=1,bias=False),
                nn.InstanceNorm2d(feature),
                # nn.BatchNorm2d(feature, affine=True),
                nn.GELU(),
            )
        else:
            self.conv_4 = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels=feature, out_channels=feature, kernel_size=3, padding=1,
                                        bias=not use_spectral_norm), use_spectral_norm),
                nn.GELU(),
            )

        self.conv5 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        residual = self.conv4(residual)
        input = torch.cat((x, residual), dim=1)
        out = self.conv5(input)
        return out

def symm_pad(im, padding):
    h, w = im.shape[-2:]
    left, right, top, bottom = padding

    x_idx = np.arange(-left, w + right)
    y_idx = np.arange(-top, h + bottom)

    x_pad = reflect(x_idx, -0.5, w - 0.5)
    y_pad = reflect(y_idx, -0.5, h - 0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]

def reflect(x, minx, maxx):
    """ Reflects an array around two points making a triangular waveform that ramps up
    and down,  allowing for pad lengths greater than the input length """
    rng = maxx - minx
    double_rng = 2 * rng
    mod = np.fmod(x - minx, double_rng)
    normed_mod = np.where(mod < 0, mod + double_rng, mod)
    out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)


class DG_discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True, use_SRM=False):
        super(DG_discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        dim = 256
        self.use_SRM = use_SRM
        self.in_channels = in_channels

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU(),
            # spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=3, stride=1, padding=1,
            #                         bias=not use_spectral_norm), use_spectral_norm),
            # nn.GELU()

        )

        self.conv2 = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels=dim , out_channels=dim , kernel_size=4,
                          stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU(),

            # spectral_norm(nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, stride=1, padding=1,
            #                         bias=not use_spectral_norm), use_spectral_norm),
            # nn.GELU()
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim , out_channels=dim , kernel_size=4, stride=2, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU(),

            # spectral_norm(nn.Conv2d(in_channels=dim*4, out_channels=dim*4, kernel_size=3, stride=1, padding=1,
            #                         bias=not use_spectral_norm), use_spectral_norm),
            # nn.GELU()
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim , out_channels=dim , kernel_size=4, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU(),

            # spectral_norm(nn.Conv2d(in_channels=dim*8, out_channels=dim*8, kernel_size=3, stride=1, padding=1,
            #                         bias=not use_spectral_norm), use_spectral_norm),
            # nn.GELU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=dim , out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):

        conv1 = self.conv1(x)
        # conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True,use_SRM=False):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        dim=32
        self.use_SRM = use_SRM
        self.in_channels = in_channels
        # if self.use_SRM:
        #
        #     self.init_conv = nn.Conv2d(3, dim - 12, 5, 1, padding=0, bias=False)
        #
        #     self.SRMConv2D = nn.Conv2d(3, 9, 5, 1, padding=0, bias=False)
        #     self.SRMConv2D.weight.data = torch.load('MantraNetv4.pt')['SRMConv2D.weight']
        #     ##SRM filters (fixed)
        #
        #     for param in self.SRMConv2D.parameters():
        #         param.requires_grad = False
        #
        #     self.BayarConv2D = nn.Conv2d(3, 3, 5, 1, padding=0, bias=False)
        #     self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
        #     self.bayar_mask[2, 2] = 0
        #
        #     self.bayar_final = (torch.tensor(np.zeros((5, 5)))).cuda()
        #     self.bayar_final[2, 2] = -1
        #
        # else:
        #     self.init_conv = nn.Conv2d(in_channels, dim, 5, 1, padding=0, bias=False)
        #
        # self.activation = nn.GELU()
        # if self.in_channels>3:
        #     self.canny_init_conv = nn.Conv2d(1, 4, 5, 1, padding=0, bias=False)

        self.init_conv = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=4, stride=2,
                          padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU(),

            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU()


        )

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim*2, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU(),

            spectral_norm(nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=3, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(dim*2, out_channels=dim*4, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU(),

            spectral_norm(nn.Conv2d(in_channels=dim*4, out_channels=dim*4, kernel_size=3, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU()
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim*4, out_channels=dim*8, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim*8, out_channels=dim*8, kernel_size=3, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU()
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim*8, out_channels=dim*16, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim*16, out_channels=dim*16, kernel_size=3, stride=1, padding=1,
                                    bias=not use_spectral_norm), use_spectral_norm),
            nn.GELU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=dim*16, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        # if self.use_SRM:
        #     self.BayarConv2D.weight.data *= self.bayar_mask
        #     self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
        #     self.BayarConv2D.weight.data += self.bayar_final
        #
        #     # Symmetric padding
        #     x = symm_pad(x, (2, 2, 2, 2))
        #
        #     conv_init = self.init_conv(x[:,:3,:,:])
        #     conv_bayar = self.BayarConv2D(x[:,:3,:,:])
        #     conv_srm = self.SRMConv2D(x[:,:3,:,:])
        #
        #     conv1 = torch.cat((conv_init, conv_srm, conv_bayar), dim=1)
        #
        #     if self.in_channels>3:
        #         conv = self.canny_init_conv(x[:,3:,:,:])
        #         conv1 = torch.cat((conv1, conv), dim=1)
        # else:
        conv0 = self.init_conv(x)

        # conv1 = self.activation(conv1)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs #, [conv1, conv2, conv3, conv4, conv5]

from network.PureUpSample import PureUpsampling
class tianchi_Unet(BaseNetwork):
    def __init__(self, in_channels=3, out_channels=1, residual_blocks=8, init_weights=True, use_spectral_norm=True,
                   dim=32):
        super(tianchi_Unet, self).__init__()
  
        self.init_conv = nn.Conv2d(in_channels, dim-12, 5, 1, padding=0, bias=False)

        self.SRMConv2D = nn.Conv2d(in_channels, 9, 5, 1, padding=0, bias=False)
        self.SRMConv2D.weight.data = torch.load('MantraNetv4.pt')['SRMConv2D.weight']
        ##SRM filters (fixed)
        for param in self.SRMConv2D.parameters():
            param.requires_grad = False

        self.BayarConv2D = nn.Conv2d(in_channels, 3, 5, 1, padding=0, bias=False)
        self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
        self.bayar_mask[2, 2] = 0

        self.bayar_final = (torch.tensor(np.zeros((5, 5)))).cuda()
        self.bayar_final[2, 2] = -1
        self.activation = nn.GELU()
      
        self.encoder_1_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1), use_spectral_norm),
            nn.GELU(),
        )
        self.encoder_1_7 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, stride=2, padding=3),use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, stride=1, padding=3),use_spectral_norm),
            nn.GELU(),
        )
        self.encoder_2_3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1),use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),use_spectral_norm),
            nn.GELU(),
        )
        self.encoder_2_7 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, stride=2, padding=3),use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, stride=1, padding=3),use_spectral_norm),
            nn.GELU(),
        )

        self.middle_layer = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1, dilation=1, padding=2),use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1, dilation=2, padding=4),use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1, dilation=4, padding=8),use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, stride=1, dilation=8, padding=16),use_spectral_norm),
            nn.GELU(),
        )

        # PureUpsampling(scale=2)
        self.decoder_2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 4 * 2, out_channels=dim * 2, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU(),
        )

        self.decoder_1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 2 * 2, out_channels=dim, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU(),
        )
        if not self.additional_conv:
            self.decoder_0 = nn.Sequential(
                # nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=1, padding=0),
            )
        else:
            self.decoder_0 = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=3, padding=1),
                              use_spectral_norm),
                nn.GELU(),
                # nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=1, padding=0),
            )


        if init_weights:
            self.init_weights()


    def forward(self, x, qf=None):

        ## **Bayar constraints**

        self.BayarConv2D.weight.data *= self.bayar_mask
        self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
        self.BayarConv2D.weight.data += self.bayar_final

        # Symmetric padding
        x = symm_pad(x, (2, 2, 2, 2))

        conv_init = self.init_conv(x)
        conv_bayar = self.BayarConv2D(x)
        conv_srm = self.SRMConv2D(x)

        first_block = torch.cat((conv_init, conv_srm, conv_bayar), dim=1)
        e0 = self.activation(first_block)

        # encoder 1
        x = symm_pad(x, (1, 1, 1, 1))

      
        e1 = self.encoder_1(e0)
        e2 = self.encoder_2(e1)
        m = self.middle(e2)
        if self.with_attn:
            qf_embedding = self.qf_embed(qf)
            gamma_3 = self.to_gamma_3(qf_embedding).unsqueeze(-1).unsqueeze(-1)
            beta_3 = self.to_beta_3(qf_embedding).unsqueeze(-1).unsqueeze(-1)

            gamma_2 = self.to_gamma_2(qf_embedding).unsqueeze(-1).unsqueeze(-1)
            beta_2 = self.to_beta_2(qf_embedding).unsqueeze(-1).unsqueeze(-1)

            gamma_1 = self.to_gamma_1(qf_embedding).unsqueeze(-1).unsqueeze(-1)
            beta_1 = self.to_beta_1(qf_embedding).unsqueeze(-1).unsqueeze(-1)

            m_a = ((gamma_3) * self.attn_3(m) + beta_3)

        d2 = self.decoder_2(torch.cat((e2, m), dim=1)) if not self.with_attn else self.decoder_2(torch.cat((e2, m_a), dim=1))
        if self.with_attn:
            d2_a = ((gamma_2) * self.attn_2(d2) + beta_2)
        d1 = self.decoder_1(torch.cat((e1, d2), dim=1)) if not self.with_attn else self.decoder_1(torch.cat((e1, d2_a), dim=1))
        if self.with_attn:
            d1_a = ((gamma_1) * self.attn_1(d1) + beta_1)
        x = self.decoder_0(torch.cat((e0, d1), dim=1)) if not self.with_attn else self.decoder_0(torch.cat((e0, d1_a), dim=1))
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        if self.with_attn:
            return x, (d2, d1) # (d2, d1, m)
        else:
            return x, (d2, d1)



class UNetDiscriminator(BaseNetwork):
    def __init__(self, in_channels=3, out_channels=3, residual_blocks=8, init_weights=True, use_spectral_norm=True,
                 use_SRM=True, with_attn=False, additional_conv=False, dim=32,use_sigmoid=False):
        super(UNetDiscriminator, self).__init__()
        # dim = 32
        self.use_SRM = use_SRM
        self.use_sigmoid = use_sigmoid
        self.with_attn = with_attn
        self.additional_conv=additional_conv
        if self.use_SRM:
            self.init_conv = nn.Conv2d(in_channels, dim-12, 5, 1, padding=0, bias=False)

            self.SRMConv2D = nn.Conv2d(in_channels, 9, 5, 1, padding=0, bias=False)
            self.SRMConv2D.weight.data = torch.load('MantraNetv4.pt')['SRMConv2D.weight']
            ##SRM filters (fixed)

            for param in self.SRMConv2D.parameters():
                param.requires_grad = False

            self.BayarConv2D = nn.Conv2d(in_channels, 3, 5, 1, padding=0, bias=False)
            self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
            self.bayar_mask[2, 2] = 0

            self.bayar_final = (torch.tensor(np.zeros((5, 5)))).cuda()
            self.bayar_final[2, 2] = -1
            self.activation = nn.GELU()
            # self.activation = nn.GELU()
        else:
            self.init_conv = nn.Sequential(
                # nn.ReflectionPad2d(3),
                spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
                nn.GELU(),
                spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
                nn.GELU(),
            )

        # self.middle_and_last_block = nn.ModuleList([
        #     nn.Conv2d(16, 32, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, 1, padding=0)]
        # )

        # self.encoder_0 = nn.Sequential(
        #     nn.ReflectionPad2d(3),
        #     spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=7, padding=0), use_spectral_norm),
        #     nn.GELU(),
        #     spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
        #     nn.GELU(),
        # )
        self.encoder_1 = nn.Sequential(

            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.GELU(),
            # nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU(),
            # nn.GELU(),
        )
        self.encoder_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 4, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.GELU(),
            # nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU()
            # nn.GELU()
        )

        blocks = []
        for _ in range(residual_blocks):  # residual_blocks
            block = ResnetBlock(dim * 4, dilation=2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder_2 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 4 * 2, out_channels=dim * 2, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.GELU(),
            # nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU(),
            # nn.GELU(),
        )

        self.decoder_1 = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=dim * 2 * 2, out_channels=dim, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.GELU(),
            # nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU(),
            # nn.GELU(),
        )
        if not self.additional_conv:
            self.decoder_0 = nn.Sequential(
                # nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=1, padding=0),
            )
        else:
            self.decoder_0 = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=3, padding=1),
                              use_spectral_norm),
                nn.GELU(),
                # nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=1, padding=0),
            )


        if init_weights:
            self.init_weights()

        if self.with_attn:
            self.attn_1 = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, padding=0),
            )

            self.attn_2 = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=7, padding=0),
            )

            self.attn_3 = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=7, padding=0),
            )

            self.qf_embed = nn.Sequential(torch.nn.Linear(1, 512),
                                       nn.ReLU(),
                                       torch.nn.Linear(512, 512),
                                       nn.ReLU(),
                                       torch.nn.Linear(512, 512),
                                       nn.ReLU()
                                       )
            self.to_gamma_3 = nn.Sequential(torch.nn.Linear(512, dim * 4), nn.Sigmoid())
            self.to_beta_3 = nn.Sequential(torch.nn.Linear(512, dim * 4), nn.Tanh())
            self.to_gamma_2 = nn.Sequential(torch.nn.Linear(512, dim * 2), nn.Sigmoid())
            self.to_beta_2 = nn.Sequential(torch.nn.Linear(512, dim * 2), nn.Tanh())
            self.to_gamma_1 = nn.Sequential(torch.nn.Linear(512, dim), nn.Sigmoid())
            self.to_beta_1 = nn.Sequential(torch.nn.Linear(512, dim), nn.Tanh())

    def forward(self, x, qf=None):

        ## **Bayar constraints**
        if self.use_SRM:
            self.BayarConv2D.weight.data *= self.bayar_mask
            self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
            self.BayarConv2D.weight.data += self.bayar_final

            # Symmetric padding
            x = symm_pad(x, (2, 2, 2, 2))

            conv_init = self.init_conv(x)
            conv_bayar = self.BayarConv2D(x)
            conv_srm = self.SRMConv2D(x)

            first_block = torch.cat([conv_init, conv_srm, conv_bayar], axis=1)
            e0 = self.activation(first_block)
        else:
            e0 = self.init_conv(x)

        # for layer in self.middle_and_last_block:
        #
        #     if isinstance(layer, nn.Conv2d):
        #         last_block = symm_pad(last_block, (1, 1, 1, 1))
        #
        #     last_block = layer(last_block)
        #
        # return (torch.tanh(last_block) + 1) / 2

        # e0 = self.encoder_0(x)
        e1 = self.encoder_1(e0)
        e2 = self.encoder_2(e1)
        m = self.middle(e2)
        if self.with_attn:
            qf_embedding = self.qf_embed(qf)
            gamma_3 = self.to_gamma_3(qf_embedding).unsqueeze(-1).unsqueeze(-1)
            beta_3 = self.to_beta_3(qf_embedding).unsqueeze(-1).unsqueeze(-1)

            gamma_2 = self.to_gamma_2(qf_embedding).unsqueeze(-1).unsqueeze(-1)
            beta_2 = self.to_beta_2(qf_embedding).unsqueeze(-1).unsqueeze(-1)

            gamma_1 = self.to_gamma_1(qf_embedding).unsqueeze(-1).unsqueeze(-1)
            beta_1 = self.to_beta_1(qf_embedding).unsqueeze(-1).unsqueeze(-1)

            m_a = ((gamma_3) * self.attn_3(m) + beta_3)

        d2 = self.decoder_2(torch.cat((e2, m), dim=1)) if not self.with_attn else self.decoder_2(torch.cat((e2, m_a), dim=1))
        if self.with_attn:
            d2_a = ((gamma_2) * self.attn_2(d2) + beta_2)
        d1 = self.decoder_1(torch.cat((e1, d2), dim=1)) if not self.with_attn else self.decoder_1(torch.cat((e1, d2_a), dim=1))
        if self.with_attn:
            d1_a = ((gamma_1) * self.attn_1(d1) + beta_1)
        x = self.decoder_0(torch.cat((e0, d1), dim=1)) if not self.with_attn else self.decoder_0(torch.cat((e0, d1_a), dim=1))
        if self.use_sigmoid:
            x = torch.sigmoid(x)
        if self.with_attn:
            return x, (d2, d1) # (d2, d1, m)
        else:
            return x, (d2, d1)


class JPEGGenerator(BaseNetwork):
    def __init__(self, in_channels=3, out_channels=3, residual_blocks=8, init_weights=True, use_spectral_norm=True, use_SRM=True, with_attn=False, additional_conv=False, dim=32):
        super(JPEGGenerator, self).__init__()
        # dim = 32
        self.use_SRM = False #use_SRM
        self.with_attn = with_attn
        self.additional_conv=additional_conv
        if self.use_SRM:
            self.init_conv = nn.Conv2d(in_channels, dim-12, 5, 1, padding=0, bias=False)

            self.SRMConv2D = nn.Conv2d(in_channels, 9, 5, 1, padding=0, bias=False)
            self.SRMConv2D.weight.data = torch.load('MantraNetv4.pt')['SRMConv2D.weight']
            ##SRM filters (fixed)

            for param in self.SRMConv2D.parameters():
                param.requires_grad = False

            self.BayarConv2D = nn.Conv2d(in_channels, 3, 5, 1, padding=0, bias=False)
            self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5)))).cuda()
            self.bayar_mask[2, 2] = 0

            self.bayar_final = (torch.tensor(np.zeros((5, 5)))).cuda()
            self.bayar_final[2, 2] = -1
            self.activation = nn.GELU()
        else:
            self.init_conv = nn.Sequential(
                nn.ReflectionPad2d(3),
                spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=7, padding=0), use_spectral_norm),
                nn.GELU(),
                spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
                nn.GELU(),
            )

        # self.middle_and_last_block = nn.ModuleList([
        #     nn.Conv2d(16, 32, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 128, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 256, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, 1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 256, 3, 1, padding=0)]
        # )

        self.encoder_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=3, stride=1, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
        )

        self.encoder_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 4, kernel_size=3, stride=1, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
        )

        # blocks = []
        # for _ in range(residual_blocks):  # residual_blocks
        #     block = ResnetBlock(dim * 4, dilation=2, use_spectral_norm=use_spectral_norm)
        #     blocks.append(block)
        #
        # self.middle = nn.Sequential(*blocks)

        self.decoder_2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 4, out_channels=dim * 2, kernel_size=3, stride=1, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=3, padding=1),use_spectral_norm),
            nn.GELU(),
        )

        self.decoder_1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=3, stride=1, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU(),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1), use_spectral_norm),
            nn.GELU(),
        )

        if not self.additional_conv:
            self.decoder_0 = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=dim * 2, out_channels=out_channels, kernel_size=7, padding=0),
            )
        else:
            self.decoder_0 = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=3, padding=1),
                              use_spectral_norm),
                nn.GELU(),
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=dim, out_channels=out_channels, kernel_size=7, padding=0),
            )


        if init_weights:
            self.init_weights()

        if self.with_attn:
            self.attn_1 = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, padding=0),
            )

            self.attn_2 = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=7, padding=0),
            )

            self.attn_3 = nn.Sequential(
                nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=7, padding=0),
            )

            self.qf_embed = nn.Sequential(torch.nn.Linear(1, 512),
                                       nn.ReLU(),
                                       torch.nn.Linear(512, 512),
                                       nn.ReLU(),
                                       torch.nn.Linear(512, 512),
                                       nn.ReLU()
                                       )
            self.to_gamma_3 = nn.Sequential(torch.nn.Linear(512, dim * 4), nn.Sigmoid())
            self.to_beta_3 = nn.Sequential(torch.nn.Linear(512, dim * 4), nn.Tanh())
            self.to_gamma_2 = nn.Sequential(torch.nn.Linear(512, dim * 2), nn.Sigmoid())
            self.to_beta_2 = nn.Sequential(torch.nn.Linear(512, dim * 2), nn.Tanh())
            self.to_gamma_1 = nn.Sequential(torch.nn.Linear(512, dim), nn.Sigmoid())
            self.to_beta_1 = nn.Sequential(torch.nn.Linear(512, dim), nn.Tanh())

    def forward(self, x, qf=None):

        ## **Bayar constraints**
        if self.use_SRM:
            self.BayarConv2D.weight.data *= self.bayar_mask
            self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
            self.BayarConv2D.weight.data += self.bayar_final

            # Symmetric padding
            x = symm_pad(x, (2, 2, 2, 2))

            conv_init = self.init_conv(x)
            conv_bayar = self.BayarConv2D(x)
            conv_srm = self.SRMConv2D(x)

            first_block = torch.cat([conv_init, conv_srm, conv_bayar], axis=1)
            e0 = self.activation(first_block)
        else:
            e0 = self.init_conv(x)

        if self.with_attn:
            qf_embedding = self.qf_embed(qf)
            gamma_3 = self.to_gamma_3(qf_embedding).unsqueeze(-1).unsqueeze(-1)
            beta_3 = self.to_beta_3(qf_embedding).unsqueeze(-1).unsqueeze(-1)

            gamma_2 = self.to_gamma_2(qf_embedding).unsqueeze(-1).unsqueeze(-1)
            beta_2 = self.to_beta_2(qf_embedding).unsqueeze(-1).unsqueeze(-1)

            gamma_1 = self.to_gamma_1(qf_embedding).unsqueeze(-1).unsqueeze(-1)
            beta_1 = self.to_beta_1(qf_embedding).unsqueeze(-1).unsqueeze(-1)

        # e0 = self.encoder_0(x)
        e1 = self.encoder_1(e0)
        m = self.encoder_2(e1)
        # m = self.middle(e2)

        if self.with_attn:
            m_a = m + ((gamma_3) * self.attn_3(m) + beta_3)

        d2 = self.decoder_2(m) if not self.with_attn else self.decoder_2(m_a)
        if self.with_attn:
            d2_a = d2+((gamma_2) * self.attn_2(d2) + beta_2)
        d1 = self.decoder_1(d2) if not self.with_attn else self.decoder_1(d2_a)
        if self.with_attn:
            d1_a = d1+((gamma_1) * self.attn_1(d1) + beta_1)
        x = self.decoder_0(d1) if not self.with_attn else self.decoder_0(d1_a)
        # x = (torch.tanh(x) + 1) / 2
        if self.with_attn:
            return x, (m_a, d2_a, d1_a) # (d2, d1, m)
        else:
            return x, (m, d2, d1)


class EdgeGenerator(BaseNetwork):
    def __init__(self, in_channels=3, out_channels=1, residual_blocks=8, use_spectral_norm=True, init_weights=True, dims_in=[[3, 64, 64]], down_num=3, block_num=[2, 2, 2]):
        super(EdgeGenerator, self).__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            # nn.InstanceNorm2d(64),
            # nn.BatchNorm2d(64, affine=True),
            nn.GELU(),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            # nn.InstanceNorm2d(128),
            # nn.BatchNorm2d(128, affine=True),
            nn.GELU(),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            # nn.InstanceNorm2d(256),
            # nn.BatchNorm2d(256, affine=True),
            nn.GELU()
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            # nn.InstanceNorm2d(128),
            # nn.BatchNorm2d(128, affine=True),
            nn.GELU(),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            # nn.InstanceNorm2d(64),
            # nn.BatchNorm2d(64, affine=True),
            nn.GELU(),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=7, padding=0),
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        # x = (torch.tanh(x) + 1) / 2
        return x

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        if use_spectral_norm:
            self.conv_block = nn.Sequential(
                nn.ReflectionPad2d(dilation),
                spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
                # nn.InstanceNorm2d(dim),
                nn.GELU(),
                # nn.GELU(),

                nn.ReflectionPad2d(1),
                spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
                # nn.InstanceNorm2d(dim),
            )
        else:
            self.conv_block = nn.Sequential(
                nn.ReflectionPad2d(dilation),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation,bias=False),
                # nn.BatchNorm2d(dim, affine=True),
                nn.InstanceNorm2d(dim),
                nn.GELU(),

                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1,bias=True),
            )


    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out




