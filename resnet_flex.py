'''ResNet_RS improved
https://arxiv.org/pdf/2103.07579.pdf
https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py
https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_rs/configs/resnetrs50_i160.yaml
'''


import torch
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
# from lsa_layer import LSAConv2D


class LayerNormIm(nn.LayerNorm):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)

        self.dim=dim
        self.reduce_dim = tuple(range(2, 2+self.dim))
        if self.elementwise_affine:
            sh = (1,) + self.normalized_shape + (1,) * dim
            self.weight = torch.nn.Parameter(self.weight.reshape(sh))
            self.bias = torch.nn.Parameter(self.bias.reshape(sh))

    @torch.jit.script
    def norm(x):
        reduce_dim = [i for i in range(2, x.ndim)]
        mean = x.mean(dim=reduce_dim, keepdim=True)
        x = x - mean
        var = (x**2).mean(dim=reduce_dim, keepdim=True)
        x = x / (var + 1e-5)
        return x

    def forward(self, x):

        # mean = torch.mean(x, dim=self.reduce_dim, keepdim=True)
        # x = x - mean
        # var = torch.mean(x**2, dim=self.reduce_dim, keepdims=True)
        # x =  x/(var + self.eps)

        x = self.norm(x)
        if self.elementwise_affine:
            x = self.weight * x + self.bias

        return x

class LayerNorm2(nn.LayerNorm):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)

        if self.elementwise_affine:
            sh = (1,) + self.normalized_shape + (1,) * dim
            self.weight = torch.nn.Parameter(self.weight.reshape(sh))
            self.bias = torch.nn.Parameter(self.bias.reshape(sh))


    def forward(self, x):

        mean = torch.mean(x, dim=1, keepdims=True)
        var = torch.var(x, dim=1, keepdims=True, unbiased=False)
        x = ((x-mean) / (var + self.eps))

        if self.elementwise_affine:
            x = self.weight * x + self.bias

        return x


def norm_func(norm, num_features, affine=True, num_groups=8, dim=2):

    if dim == 2:
        if norm == 'batchnorm':
            f = nn.BatchNorm2d(num_features, affine=affine)
        elif norm == 'groupnorm':
            f = nn.GroupNorm(num_groups=num_groups, num_channels=num_features, affine=affine)
        elif norm == 'instancenorm':
            f = nn.InstanceNorm2d(num_features=num_features, affine=affine)
        elif norm == 'layernorm':
            f = LayerNormIm(dim=2, normalized_shape=num_features,  elementwise_affine=affine)
        elif norm == 'layernorm2':
            f = LayerNorm2(dim=2, normalized_shape=num_features,  elementwise_affine=affine)
        else:
            raise ValueError('Wrong norm name'+str(norm))

    elif dim==3:
        if norm == 'batchnorm':
            f = nn.BatchNorm3d(num_features, affine=affine)
        elif norm == 'groupnorm':
            # print('groupnorm with group', num_groups)
            f = nn.GroupNorm(num_groups=num_groups, num_channels=num_features, affine=affine)
        elif norm == 'instancenorm':
            f = nn.InstanceNorm3d(num_features=num_features, affine=affine)
        elif norm == 'layernorm':
            f = LayerNormIm(dim=3, normalized_shape=num_features,  elementwise_affine=affine)
        elif norm == 'layernorm2':
            f = LayerNorm2(dim=2, normalized_shape=num_features,  elementwise_affine=affine)
        else:
            raise ValueError('Wrong norm name' + str(norm))

    else:
         raise ValueError('Wrong norm dim'+str(dim))

    return f

def relu_func(relu='relu', alpha=0.01, inplace=True):

    if relu == 'relu':
        f = nn.ReLU(inplace=inplace)
    elif relu in ['leaky_relu', 'leaky']:
        f = nn.LeakyReLU(negative_slope=alpha, inplace=inplace)
    else:
        raise ValueError('Wrong activation name')

    return f


def conv_func(name='conv', dim=2, **kwargs):

    if name=='conv':
        if dim == 2:
            f = nn.Conv2d(**kwargs)
        elif dim == 3:
            f = nn.Conv3d(**kwargs)
        else:
            raise ValueError('wrong dim')

    elif name=='lsa_conv':

        # from lsa_layer import LSAConv2D
        f = LSAConv2D(**kwargs)

    else:
        raise ValueError('Unknown conv name'+str(name))

    return f

def conv_transpose_func(dim=2, **kwargs):
    if dim == 2:
        f = nn.ConvTranspose2d(**kwargs)
    elif dim == 3:
        f = nn.ConvTranspose3d(**kwargs)
    else:
        raise ValueError('wrong dim')

    return f

def avgpool_func(dim=2, **kwargs):
    if dim == 2:
        f = nn.AvgPool2d(**kwargs)
    elif dim == 3:
        f = nn.AvgPool3d(**kwargs)
    else:
        raise ValueError('wrong dim')

    return f

def global_avg_pool(x, keepdims=False):
    # print('dim', list(range(2,len(x.shape))))
    return x.mean(dim=list(range(2,x.ndim)), keepdims=keepdims)


def interp_func(dim=2, **kwargs):

    if 'mode' in kwargs:
        if kwargs['mode'] in ['linear', 'bilinear', 'trilinear']:
            kwargs['mode'] = 'bilinear' if dim == 2 else 'trilinear'

        elif kwargs['mode']=='nearest' and 'align_corners' in kwargs and kwargs['align_corners'] is not None:
            kwargs['align_corners']=None

    return nn.Upsample(**kwargs)

class SELayer(nn.Module):

    dim=2
    def __init__(self, inplanes, relu=None):
        super().__init__()

        self.conv1 = conv_func(dim=self.dim, in_channels=inplanes, out_channels=inplanes // 4, kernel_size=1, stride=1)
        self.conv2 = conv_func(dim=self.dim, in_channels=inplanes // 4, out_channels=inplanes, kernel_size=1, stride=1)

        if relu is None:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = relu()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = global_avg_pool(x, keepdims=True)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.sigmoid(out)

        return x * out



class BasicBlock(nn.Module):

    expansion = 1
    dim=2
    norm=nn.BatchNorm2d
    relu=nn.ReLU
    use_se = True
    lastnorm_init_zero = False

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()


        out_channels = self.expansion*planes
        downsize=False
        if stride >=1:
            self.conv1 = conv_func(dim=self.dim,  in_channels=in_planes, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            out_channels = planes
            downsize = True
            # self.conv1 = nn.Sequential(
            #     conv_func(dim=self.dim, in_channels=in_planes, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            #     Interpolate(scale_factor=int(1 / stride), dim=self.dim)
            # )
            self.conv1 = conv_transpose_func(dim=self.dim,  in_channels=in_planes, out_channels=out_channels, kernel_size=3, stride=int(1/stride), padding=1, output_padding=1, bias=False)

        self.bn1 = self.norm(num_features=out_channels)
        self.relu1 = self.relu()

        self.conv2 = conv_func(dim=self.dim, in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = self.norm(num_features=out_channels)

        if self.lastnorm_init_zero:
            try:
                self.bn3.weight.data.zero_()  # init to zero
            except AttributeError:
                print('Warning, could not init norm to zero', self.bn3)

        if stride != 1 or in_planes != out_channels:
            shortcut_layers = []
            if stride > 1:
                shortcut_layers.append(avgpool_func(dim=self.dim, kernel_size=stride, stride=stride))

            shortcut_layers.append(conv_func(dim=self.dim, in_channels=in_planes, out_channels=out_channels, kernel_size=1, stride=1, bias=False))

            if stride < 1:
                shortcut_layers.append(interp_func(dim=self.dim, scale_factor=int(1/stride), mode='linear', align_corners=False))

                # shortcut_layers.append(Interpolate(scale_factor=int(1/stride), dim=self.dim, mode='nearest'))


            shortcut_layers.append(self.norm(num_features=out_channels))
            self.shortcut = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut=nn.Identity()

        self.selayer = SELayer(out_channels, relu=self.relu) if self.use_se else nn.Identity()

        self.relu2 = self.relu(inplace=not downsize) #inplace True for conv_transpose
        # self.relu2 = self.relu() #inplace True for conv_transpose


    def forward(self, x):

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.selayer(out)
        out += self.shortcut(x)
        out = self.relu2(out)
        return out



class BasicBlockLSA(nn.Module):

    expansion = 1
    dim=2
    norm=nn.BatchNorm2d
    relu=nn.ReLU
    use_se = True
    lastnorm_init_zero = False

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()


        out_channels = self.expansion*planes
        downsize=False
        if stride >=1:
            self.conv1 = conv_func(name='lsa_conv', dim=self.dim,  in_channels=in_planes, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            out_channels = planes
            downsize = True
            # self.conv1 = nn.Sequential(
            #     conv_func(dim=self.dim, in_channels=in_planes, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            #     Interpolate(scale_factor=int(1 / stride), dim=self.dim)
            # )
            self.conv1 = conv_transpose_func(dim=self.dim,  in_channels=in_planes, out_channels=out_channels, kernel_size=3, stride=int(1/stride), padding=1, output_padding=1, bias=False)

        self.bn1 = self.norm(num_features=out_channels)
        self.relu1 = self.relu()

        self.conv2 = conv_func(name='lsa_conv', dim=self.dim, in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = self.norm(num_features=out_channels)

        if self.lastnorm_init_zero:
            try:
                self.bn3.weight.data.zero_()  # init to zero
            except AttributeError:
                print('Warning, could not init norm to zero', self.bn3)

        if stride != 1 or in_planes != out_channels:
            shortcut_layers = []
            if stride > 1:
                shortcut_layers.append(avgpool_func(dim=self.dim, kernel_size=stride, stride=stride))

            shortcut_layers.append(conv_func(dim=self.dim, in_channels=in_planes, out_channels=out_channels, kernel_size=1, stride=1, bias=False))

            if stride < 1:
                shortcut_layers.append(interp_func(dim=self.dim, scale_factor=int(1/stride), mode='linear', align_corners=False))

                # shortcut_layers.append(Interpolate(scale_factor=int(1/stride), dim=self.dim, mode='nearest'))


            shortcut_layers.append(self.norm(num_features=out_channels))
            self.shortcut = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut=nn.Identity()

        self.selayer = SELayer(out_channels, relu=self.relu) if self.use_se else nn.Identity()

        self.relu2 = self.relu(inplace=not downsize) #inplace True for conv_transpose
        # self.relu2 = self.relu() #inplace True for conv_transpose


    def forward(self, x):

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.selayer(out)
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class BasicBlockPre(nn.Module):

    expansion = 1
    dim=2
    norm=nn.BatchNorm2d
    relu=nn.ReLU
    use_se = True
    lastnorm_init_zero = False

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()


        out_channels = self.expansion*planes
        # downsize=False

        self.bn1 = self.norm(num_features=in_planes)
        self.relu1 = self.relu()

        if stride >=1:
            self.conv1 = conv_func(dim=self.dim,  in_channels=in_planes, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            out_channels = planes
            # downsize = True
            # self.conv1 = nn.Sequential(
            #     conv_func(dim=self.dim, in_channels=in_planes, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            #     Interpolate(scale_factor=int(1 / stride), dim=self.dim)
            # )
            self.conv1 = conv_transpose_func(dim=self.dim,  in_channels=in_planes, out_channels=out_channels, kernel_size=3, stride=int(1/stride), padding=1, output_padding=1, bias=False)


        self.bn2 = self.norm(num_features=out_channels)
        # self.relu2 = self.relu(inplace=not downsize) #inplace True for conv_transpose
        self.relu2 = self.relu()
        self.conv2 = conv_func(dim=self.dim, in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if self.lastnorm_init_zero:
            try:
                self.bn3.weight.data.zero_()  # init to zero
            except AttributeError:
                print('Warning, could not init norm to zero', self.bn3)


        if stride != 1 or in_planes != out_channels:
            shortcut_layers = []
            if stride > 1:
                shortcut_layers.append(avgpool_func(dim=self.dim, kernel_size=stride, stride=stride))
            shortcut_layers.append(conv_func(dim=self.dim, in_channels=in_planes, out_channels=out_channels, kernel_size=1, stride=1, bias=False))
            if stride < 1:
                shortcut_layers.append(interp_func(dim=self.dim, scale_factor=int(1/stride), mode='linear', align_corners=False))

            shortcut_layers.append(self.norm(num_features=out_channels))
            self.shortcut = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut=nn.Identity()

        self.selayer = SELayer(out_channels, relu=self.relu) if self.use_se else nn.Identity()


    def forward(self, x):

        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.selayer(out)
        out += self.shortcut(x)

        return out


class BasicBlockPreNoSkip(nn.Module):

    expansion = 1
    dim=2
    norm=nn.BatchNorm2d
    relu=nn.ReLU
    use_se = True
    lastnorm_init_zero = False

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()


        out_channels = self.expansion*planes
        # downsize=False

        self.bn1 = self.norm(num_features=in_planes)
        self.relu1 = self.relu()

        if stride >=1:
            self.conv1 = conv_func(dim=self.dim,  in_channels=in_planes, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            out_channels = planes
            # downsize = True
            # self.conv1 = nn.Sequential(
            #     conv_func(dim=self.dim, in_channels=in_planes, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            #     Interpolate(scale_factor=int(1 / stride), dim=self.dim)
            # )
            self.conv1 = conv_transpose_func(dim=self.dim,  in_channels=in_planes, out_channels=out_channels, kernel_size=3, stride=int(1/stride), padding=1, output_padding=1, bias=False)


        self.bn2 = self.norm(num_features=out_channels)
        # self.relu2 = self.relu(inplace=not downsize) #inplace True for conv_transpose
        self.relu2 = self.relu()
        self.conv2 = conv_func(dim=self.dim, in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if self.lastnorm_init_zero:
            try:
                self.bn3.weight.data.zero_()  # init to zero
            except AttributeError:
                print('Warning, could not init norm to zero', self.bn3)


        # if stride != 1 or in_planes != out_channels:
        #     shortcut_layers = []
        #     if stride > 1:
        #         shortcut_layers.append(avgpool_func(dim=self.dim, kernel_size=stride, stride=stride))
        #     shortcut_layers.append(conv_func(dim=self.dim, in_channels=in_planes, out_channels=out_channels, kernel_size=1, stride=1, bias=False))
        #     if stride < 1:
        #         shortcut_layers.append(interp_func(dim=self.dim, scale_factor=int(1/stride), mode='linear', align_corners=False))
        #
        #     shortcut_layers.append(self.norm(num_features=out_channels))
        #     self.shortcut = nn.Sequential(*shortcut_layers)
        # else:
        #     self.shortcut=nn.Identity()

        self.selayer = SELayer(out_channels, relu=self.relu) if self.use_se else nn.Identity()


    def forward(self, x):

        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.selayer(out)
        # out += self.shortcut(x)

        return out



class BasicBlockPreLSA(nn.Module):

    expansion = 1
    dim=2
    norm=nn.BatchNorm2d
    relu=nn.ReLU
    use_se = True
    lastnorm_init_zero = False

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()


        out_channels = self.expansion*planes
        # downsize=False

        self.bn1 = self.norm(num_features=in_planes)
        self.relu1 = self.relu()

        if stride >=1:
            self.conv1 = conv_func(name='lsa_conv', dim=self.dim,  in_channels=in_planes, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        else:
            out_channels = planes
            # downsize = True
            # self.conv1 = nn.Sequential(
            #     conv_func(dim=self.dim, in_channels=in_planes, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            #     Interpolate(scale_factor=int(1 / stride), dim=self.dim)
            # )
            self.conv1 = conv_transpose_func(dim=self.dim,  in_channels=in_planes, out_channels=out_channels, kernel_size=3, stride=int(1/stride), padding=1, output_padding=1, bias=False)


        self.bn2 = self.norm(num_features=out_channels)
        # self.relu2 = self.relu(inplace=not downsize) #inplace True for conv_transpose
        self.relu2 = self.relu()
        self.conv2 = conv_func(name='lsa_conv', dim=self.dim, in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        if self.lastnorm_init_zero:
            try:
                self.bn3.weight.data.zero_()  # init to zero
            except AttributeError:
                print('Warning, could not init norm to zero', self.bn3)


        if stride != 1 or in_planes != out_channels:
            shortcut_layers = []
            if stride > 1:
                shortcut_layers.append(avgpool_func(dim=self.dim, kernel_size=stride, stride=stride))
            shortcut_layers.append(conv_func(dim=self.dim, in_channels=in_planes, out_channels=out_channels, kernel_size=1, stride=1, bias=False))
            if stride < 1:
                shortcut_layers.append(interp_func(dim=self.dim, scale_factor=int(1/stride), mode='linear', align_corners=False))

            shortcut_layers.append(self.norm(num_features=out_channels))
            self.shortcut = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut=nn.Identity()

        self.selayer = SELayer(out_channels, relu=self.relu) if self.use_se else nn.Identity()


    def forward(self, x):

        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.selayer(out)
        out += self.shortcut(x)

        return out


class Bottleneck(nn.Module):
    # expansion = 4
    expansion = 1

    dim=2
    norm=nn.BatchNorm2d
    relu=nn.ReLU
    use_se = True
    lastnorm_init_zero = False

    def __init__(self, in_planes, planes,  stride=1):
        super().__init__()

        # print('Bottleneck', in_planes, planes,stride)

        self.conv1 = conv_func(dim=self.dim, in_channels=in_planes, out_channels=planes, kernel_size=1, bias=False)
        self.bn1 = self.norm(num_features=planes)
        self.relu1 = self.relu()


        if stride >=1:
            self.conv2 = conv_func(dim=self.dim, in_channels=planes, out_channels=planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
        else:
            out_channels = planes
            self.conv2 = conv_transpose_func(dim=self.dim, in_channels=planes, out_channels=out_channels, kernel_size=3, stride=int(1/stride),
                                   padding=1,  output_padding=1, bias=False)

        # self.conv2 = conv_func(dim=self.dim, in_channels=planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = self.norm(num_features=planes)
        self.relu2 = self.relu()

        self.conv3 = conv_func(dim=self.dim, in_channels=planes, out_channels=self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = self.norm(num_features=self.expansion * planes)

        if self.lastnorm_init_zero:
            try:
                self.bn3.weight.data.zero_()  # init to zero
            except AttributeError:
                print('Warning, could not init norm to zero', self.bn3)


        self.selayer = SELayer(self.expansion * planes, relu=self.relu) if self.use_se else nn.Identity()

        if stride != 1 or in_planes != self.expansion*planes:
            shortcut_layers = []
            if stride > 1:
                shortcut_layers.append(avgpool_func(dim=self.dim, kernel_size=stride, stride=stride))
            shortcut_layers.append(conv_func(dim=self.dim, in_channels=in_planes, out_channels=self.expansion * planes, kernel_size=1, stride=1, bias=False))
            if stride < 1:
                shortcut_layers.append(interp_func(dim=self.dim, scale_factor=int(1/stride), mode='linear', align_corners=False))

            shortcut_layers.append(self.norm(num_features=self.expansion * planes))
            self.shortcut = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut = nn.Identity()

        self.relu3 = self.relu()


    def forward(self, x):

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.selayer(out)

        x = self.shortcut(x)
        out = self.relu3(out+x)
        return out


class ResNetEncoder(nn.Module):

    def __init__(self, block='basic_pre',
                 num_blocks=(1, 2, 2, 4),
                 in_channels=3,
                 norm='groupnorm',
                 relu='relu',
                 norm_param = {},
                 relu_param = {},
                 conv_init_stack=False,
                 lastnorm_init_zero=False,
                 use_se=False,
                 no_skip = False,
                 base_filters = 16,
                 mode=1,
                 dim = 2, #spatial dim 2 or 3
                 **kwargs):

        super().__init__()


        #save input params
        self.block = block
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.norm = norm
        self.relu = relu
        self.conv_init_stack = conv_init_stack
        self.lastnorm_init_zero = lastnorm_init_zero
        self.use_se = use_se
        self.no_skip = no_skip
        # self.width_multiplier = width_multiplier
        # self.args=args
        self.norm_param = norm_param
        self.relu_param = relu_param
        self.dim=dim
        self.base_filters = base_filters

        # block, norm, relu, conv = self._parse_funcs(dim=dim, block=block, norm=norm, relu=relu, norm_param=norm_param, relu_param=relu_param)
        block, norm, relu = self._parse_funcs(dim=self.dim, block=self.block, norm=self.norm, relu=self.relu, norm_param=self.norm_param, relu_param=self.relu_param)

        # nf = width_multiplier * base_filters #base number of features
        filters = base_filters #base number of features

        # update block class params
        block.dim=self.dim
        block.norm=norm
        block.relu=relu
        block.use_se = self.use_se
        block.lastnorm_init_zero = self.lastnorm_init_zero
        self.mode=mode

        #update dimensions (2D vs 3D)
        SELayer.dim=self.dim

        if conv_init_stack:
            self.conv_init = nn.Sequential(conv_func(dim=dim, in_channels=in_channels, out_channels=filters//2, kernel_size=3, stride=2, padding=1, bias=False),
                                           norm(num_features=filters//2),
                                           relu(),
                                           conv_func(dim=dim, in_channels=filters//2, out_channels=filters//2, kernel_size=3, stride=1, padding=1, bias=False),
                                           norm(num_features=filters//2),
                                           relu(),
                                           conv_func(dim=dim, in_channels=filters//2, out_channels=filters, kernel_size=3, stride=1, padding=1, bias=False),
                                           norm(num_features=filters),
                                           relu(),
                                           conv_func(dim=dim, in_channels=filters, out_channels=filters, kernel_size=3, stride=2, padding=1, bias=False),
                                           norm(num_features=filters),
                                           relu()
                                          )
        else:

            if block == 'basic_pre' or block==BasicBlockPre  or  block == 'basic_pre_lsa' or block==BasicBlockPreLSA or block == 'basic_pre_noskip' or block==BasicBlockPreNoSkip:
                self.conv_init = conv_func(dim=self.dim, in_channels=self.in_channels, out_channels=filters, kernel_size=3, stride=1,  padding=1, bias=False)
            else:
                self.conv_init = nn.Sequential( conv_func(dim=dim, in_channels=in_channels, out_channels=filters, kernel_size=3, stride=1, padding=1, bias=False), norm(num_features=filters), relu())

        self.layers = nn.ModuleDict()

        if self.mode == 0:
            self.layers.update({'layer1' :  self._make_layer(block, filters, filters, num_blocks[0], stride=1)})
            for i in range(1, len(num_blocks)):
                layer = self._make_layer(block, filters*block.expansion, 2*filters, num_blocks[i], stride=2)
                self.layers.update({'layer'+str(i+1) : layer})
                filters *=2

        elif self.mode==1:
            self.layers.update({'layer1': self._make_layer(block, filters, filters, self.num_blocks[0], stride=1)})
            for i in range(1, len(self.num_blocks)):
                self.layers.update({'downconv' + str(i + 1): conv_func(dim=self.dim, in_channels=filters, out_channels=filters * 2, kernel_size=3, stride=2, padding=1, bias=False)})
                self.layers.update({'layer' + str(i + 1): self._make_layer(block, filters * 2, filters * 2, self.num_blocks[i], stride=1)})
                filters *= 2

        else:
            raise ValueError('Unsupported mode '+str(self.mode))



        self.feature_dimension = filters*block.expansion


    def _parse_funcs(self, dim, block, norm, relu, norm_param={}, relu_param={}):

        if isinstance(norm, str):
            norm = partial(norm_func, norm=norm,  dim=dim, **norm_param)
        if isinstance(relu, str):
            relu = partial(relu_func, relu=relu, **relu_param)
        if isinstance(block, str):
            if block=='bottleneck':
                block = Bottleneck
            elif  block=='basic':
                block = BasicBlock
            elif  block=='basic_pre':
                block = BasicBlockPre
            elif  block=='basic_pre_noskip':
                block = BasicBlockPreNoSkip
            elif  block=='basic_pre_lsa':
                block = BasicBlockPreLSA
            elif  block=='basic_lsa':
                block = BasicBlockLSA
            else:
                raise ValueError('Unknown block: '+str(block))

        return block, norm, relu


    def _make_layer(self, block, in_planes, planes, num_blocks, stride):

        layers = []
        layers.append(block(in_planes, planes, stride))
        in_planes = planes * block.expansion

        for stride in range(1, num_blocks):
            layers.append(block(in_planes, planes))

        return nn.Sequential(*layers)

    def _forward(self, x):

        x = self.conv_init(x)
        for layer_name, layer in self.layers.items():
            x = layer(x)
            # print(layer_name, x.shape)

        return x

    def forward(self, x):
        return self._forward(x)



class ResNetLSA(ResNetEncoder):

    def __init__(self, lsa_mode, lsa_locality, lsa_kfun,  lsa_kernel_size, num_classes=10,  **kwargs):

        LSAConv2D.locality=lsa_locality
        LSAConv2D.mode=lsa_mode
        LSAConv2D.kfun=lsa_kfun
        LSAConv2D.kernel_size=lsa_kernel_size



        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.linear = nn.Linear(self.feature_dimension, num_classes)


    def forward(self, x):
        x = self._forward(x)
        x = global_avg_pool(x)
        x = self.linear(x)
        return x

class ResNet(ResNetEncoder):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.linear = nn.Linear(self.feature_dimension, num_classes)

    def forward(self, x):
        x = self._forward(x)
        x = global_avg_pool(x)
        x = self.linear(x)
        return x




class SegResNet(ResNetEncoder):
    def __init__(self,
                 num_blocks_up=None,
                 num_classes=1,
                 **kwargs):
        super().__init__(**kwargs)


        if num_blocks_up is None:
            num_blocks_up = (1,) * len(self.num_blocks)

        self.num_blocks_up = num_blocks_up
        self.num_classes = num_classes
        # self.concat = concat
        block, norm, relu = self._parse_funcs(dim=self.dim, block=self.block, norm=self.norm, relu=self.relu,
                                              norm_param=self.norm_param, relu_param=self.relu_param)

        #UP
        block.expansion=1 #dont expand
        filters = self.feature_dimension

        if self.mode==0:
            #standard post  act resnet down
            self.up_layers = nn.ModuleDict()
            for i in range(0, len(num_blocks_up)-1):
                layer = self._make_layer_up(block, filters, filters//2, num_blocks_up[i], stride=0.5)
                self.up_layers.update({'up_level'+str(i+1) : layer})
                filters = filters // 2
            self.up_layers.update({'up_layer'+str(len(num_blocks_up)) :  self._make_layer(block, filters, filters, num_blocks_up[-1], stride=1)})

        elif self.mode==1:

            # self.num_blocks_up #last element may not be used in mode1 (pre act)
            self.up_layers = nn.ModuleDict()
            for i in range(0, len(self.num_blocks)-1):
                self.up_layers.update({'up_conv' + str(i + 1): conv_func(dim=self.dim, in_channels=filters, out_channels=filters // 2,  kernel_size=1, bias=False)})
                self.up_layers.update({'up_level' + str(i + 1): interp_func(dim=self.dim, scale_factor=2, mode='linear', align_corners=False)})
                self.up_layers.update({'up_layer' + str(i + 1): self._make_layer(block, filters//2, filters//2,   num_blocks_up[0], stride=1)})
                filters = filters // 2

        else:
            raise ValueError('Unsupported mode'+str(self.mode))


        filters = self.base_filters
        if block == 'basic_pre' or block==BasicBlockPre or  block == 'basic_pre_lsa' or block==BasicBlockPreLSA or block == 'basic_pre_noskip' or block==BasicBlockPreNoSkip:
            self.conv_final = nn.Sequential(norm(num_features=filters), relu(), conv_func(dim=self.dim, in_channels=filters, out_channels=num_classes, kernel_size=1, bias=True))
        else:
            self.conv_final = conv_func(dim=self.dim, in_channels=filters, out_channels=num_classes, kernel_size=1, bias=True)

    def _make_layer_up(self, block, in_planes, planes, num_blocks, stride):

        layers = []
        if num_blocks >  1:
            for _ in range(0, num_blocks-1):
                layers.append(block(in_planes, in_planes))

        layers.append(block(in_planes, planes, stride))

        return nn.Sequential(*layers)

    # def decoder_params(self):
    #     params = []

    def _forward(self, x):

        # print('xinit', x.shape)
        x = self.conv_init(x)
        # print('x', x.shape)

        x_down=[]
        for layer_name, layer in self.layers.items():
            x = layer(x)
            if 'layer' in layer_name:
                x_down.append(x)
                # print('added', layer_name)
            # print(layer_name, x.shape)


        x_down.reverse()
        x_down=x_down[1:]
        # x_down.append(0)

        i=0
        for layer_name, layer in self.up_layers.items():
            x = layer(x)
            if i < len(x_down) and 'up_level' in layer_name:
                # print(layer_name, x.shape, x_down[i].shape)
                x = x + x_down[i] # if not self.concat else torch.cat((x, x_down[i]), dim=1)
                i +=1
            # print(layer_name, x.shape)

        x = self.conv_final(x)
        # print('final', x.shape)

        return x

    def forward(self, x):
        return self._forward(x)





def ResNet18(**kwargs):
    return ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], **kwargs)

def ResNet34(**kwargs):
    return ResNet(block=BasicBlock, num_blocks=[3, 4, 6, 3], **kwargs)

def ResNet50(**kwargs):
    return ResNet(block=Bottleneck, num_blocks=[3, 4, 6, 3], **kwargs)

def ResNet101(**kwargs):
    return ResNet(block=Bottleneck, num_blocks=[3, 4, 23, 3], **kwargs)

def ResNet152(**kwargs):
    return ResNet(block=Bottleneck, num_blocks=[3, 8, 36, 3], **kwargs)


def ResNet_classification(depth, **kwargs):

    model_params = {
        18: {'block': BasicBlock, 'layers': [2, 2, 2, 2]},
        34: {'block': BasicBlock, 'layers': [3, 4, 6, 3]},
        50: {'block': Bottleneck, 'layers': [3, 4, 6, 3]},
        101: {'block': Bottleneck, 'layers': [3, 4, 23, 3]},
        152: {'block': Bottleneck, 'layers': [3, 8, 36, 3]},
        200: {'block': Bottleneck, 'layers': [3, 24, 36, 3]},
        270: {'block': Bottleneck, 'layers': [4, 29, 53, 4]},
        350: {'block': Bottleneck, 'layers': [4, 36, 72, 4]},
        420: {'block': Bottleneck, 'layers': [4, 44, 87, 4]}
    }

    return ResNet(block=model_params[depth]['block'], num_blocks=model_params[depth]['layers'], **kwargs)

def ResNet_segmentation(depth, **kwargs):

    model_params = {
        18: {'block': BasicBlock, 'layers': [2, 2, 2, 2]},
        34: {'block': BasicBlock, 'layers': [3, 4, 6, 3]},
        50: {'block': Bottleneck, 'layers': [3, 4, 6, 3]},
        101: {'block': Bottleneck, 'layers': [3, 4, 23, 3]},
        152: {'block': Bottleneck, 'layers': [3, 8, 36, 3]},
        200: {'block': Bottleneck, 'layers': [3, 24, 36, 3]},
        270: {'block': Bottleneck, 'layers': [4, 29, 53, 4]},
        350: {'block': Bottleneck, 'layers': [4, 36, 72, 4]},
        420: {'block': Bottleneck, 'layers': [4, 44, 87, 4]}
    }

    return SegResNet(block=model_params[depth]['block'], num_blocks=model_params[depth]['layers'], **kwargs)


# def test():
#     net = ResNet18()
#     y = net(torch.randn(1, 3, 32, 32))
#     print(y.size())
#
# # test()