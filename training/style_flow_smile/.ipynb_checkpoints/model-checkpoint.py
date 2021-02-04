import torch
from torch import nn
from frn import FRN, TLU
from kornia.filters import filter2D


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2D(x, f, normalized=True)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.layers = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        )

    def forward(self, x):
        return self.layers(x)


class ConvNormLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=True, frn=False):
        super().__init__()

        if frn:
            layers = [
                ConvLayer(in_channels, out_channels, kernel_size, stride),
                FRN(out_channels),
            ]
            if activation:
                layers.append(TLU(out_channels))
        else:
            layers = [
                ConvLayer(in_channels, out_channels, kernel_size, stride),
                nn.InstanceNorm2d(out_channels, affine=True),
            ]
            if activation:
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, frn=False):
        super().__init__()
        self.branch = nn.Sequential(
            ConvNormLayer(in_channels, out_channels, kernel_size, 1, frn=frn),
            ConvNormLayer(out_channels, out_channels, kernel_size, 1, activation=False, frn=frn)
        )

        if frn:
            self.activation = TLU(out_channels)
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x + self.branch(x)
        x = self.activation(x)
        return x


class ConvTanhLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.layers = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size, stride),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)
        
class ConvNoTanhLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.layers = nn.Sequential(
            ConvLayer(in_channels, out_channels, kernel_size, stride),
        )

    def forward(self, x):
        return self.layers(x)



class Encoder2(nn.Module):
    def __init__(self, a, b, frn=False, use_skip=False):
        super().__init__()
        self.use_skip = use_skip
        filter_counts = list(map(lambda x: int(a * x), [
            32, 48, 64
        ]))
        if not use_skip:
            self.layers = nn.Sequential(
                ConvNormLayer(3, filter_counts[0], 3, 1, frn=frn),
                ConvNormLayer(filter_counts[0], filter_counts[1], 3, 2, frn=frn),
                ConvNormLayer(filter_counts[1], filter_counts[2], 3, 2, frn=frn),
            )
            res_layer_count = int(b * 4)
            [self.layers.add_module(f'{i + 3}', ResLayer(filter_counts[2], filter_counts[2], 3, frn=frn)) 
            for i in range(res_layer_count)]
        else:
            self.layers_first = nn.Sequential(
                ConvNormLayer(3, filter_counts[0], 3, 1, frn=frn),
            )
            self.layers_second = nn.Sequential(
                ConvNormLayer(filter_counts[0], filter_counts[1], 3, 2, frn=frn),
                ConvNormLayer(filter_counts[1], filter_counts[2], 3, 2, frn=frn),
            )
            res_layer_count = int(b * 4)
            [self.layers_second.add_module(f'{i + 3}', ResLayer(filter_counts[2], filter_counts[2], 3, frn=frn)) 
            for i in range(res_layer_count)]


    def forward(self, x):
        if not self.use_skip:
            return self.layers(x)
        else:
            x = self.layers_first(x)
            f_map = x
            x = self.layers_second(x)
            return x, f_map

class Decoder2(nn.Module):
    def __init__(self, a, b, frn=False, use_skip=False):
        super().__init__()
        self.use_skip = use_skip
        filter_counts = list(map(lambda x: int(a * x), [
            64, 48, 32
        ]))
        if not use_skip:
            self.layers = nn.Sequential(
                nn.Upsample(scale_factor=2),
                Blur(),
                ConvNormLayer(filter_counts[0], filter_counts[1], 3, 1, frn=frn),
                nn.Upsample(scale_factor=2),
                Blur(),
                ConvNormLayer(filter_counts[1], filter_counts[2], 3, 1, frn=frn),
                ConvNoTanhLayer(filter_counts[2], 3, 3, 1)
            )
        else:
            skip_concat = 2
            self.layers_first = nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvNormLayer(filter_counts[0], filter_counts[1], 3, 1, frn=frn),
                nn.Upsample(scale_factor=2),
                ConvNormLayer(filter_counts[1], filter_counts[2], 3, 1, frn=frn),
            )
            self.layers_second = nn.Sequential(
                ConvNoTanhLayer(filter_counts[2] * skip_concat, 3, 3, 1)
            )
            self.conv = ConvLayer(filter_counts[2], filter_counts[2], 3, 1)

    def forward(self, x):
        if not self.use_skip:
            return self.layers(x)
        else:
            x, f_map = x
            x = self.layers_first(x)
            f_map = self.conv(f_map)
            x = torch.cat([x, f_map], dim=1)
            #x += f_map
            x = self.layers_second(x)
            return x
        
        
class Encoder(nn.Module):
    def __init__(self, a, b, frn=False, use_skip=False):
        super().__init__()
        self.use_skip = use_skip
        filter_counts = list(map(lambda x: int(a * x), [
            32, 48, 64
        ]))
        if not use_skip:
            self.layers = nn.Sequential(
                ConvNormLayer(3, filter_counts[0], 3, 1, frn=frn),
                ConvNormLayer(filter_counts[0], filter_counts[1], 3, 2, frn=frn),
                ConvNormLayer(filter_counts[1], filter_counts[2], 3, 2, frn=frn),
            )
            res_layer_count = int(b * 4)
            [self.layers.add_module(f'{i + 3}', ResLayer(filter_counts[2], filter_counts[2], 3, frn=frn)) 
            for i in range(res_layer_count)]
        else:
            self.layers1 = nn.Sequential(
                ConvNormLayer(3, filter_counts[0], 3, 1, frn=frn),
            )
            self.layers2 = nn.Sequential(
                ConvNormLayer(filter_counts[0], filter_counts[1], 3, 2, frn=frn),
            )
            self.layers3 = nn.Sequential(
                ConvNormLayer(filter_counts[1], filter_counts[2], 3, 2, frn=frn),
            )
            res_layer_count = int(b * 4)
            [self.layers3.add_module(f'{i + 3}', ResLayer(filter_counts[2], filter_counts[2], 3, frn=frn)) 
            for i in range(res_layer_count)]


    def forward(self, x):
        if not self.use_skip:
            return self.layers(x)
        else:
            f_maps = []
            x = self.layers1(x)
            f_maps.append(x.clone())
            x = self.layers2(x)
            f_maps.append(x.clone())
            x = self.layers3(x)
            return x, f_maps

class Decoder(nn.Module):
    def __init__(self, a, b, frn=False, use_skip=False):
        super().__init__()
        self.use_skip = use_skip
        filter_counts = list(map(lambda x: int(a * x), [
            64, 48, 32
        ]))
        if not use_skip:
            self.layers = nn.Sequential(
                nn.Upsample(scale_factor=2),
                Blur(),
                ConvNormLayer(filter_counts[0], filter_counts[1], 3, 1, frn=frn),
                nn.Upsample(scale_factor=2),
                Blur(),
                ConvNormLayer(filter_counts[1], filter_counts[2], 3, 1, frn=frn),
                ConvNoTanhLayer(filter_counts[2], 3, 3, 1)
            )
        else:
            skip_concat = 2
            self.layers1 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvNormLayer(filter_counts[0], filter_counts[1], 3, 1, frn=frn),
            )
            self.layers2 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvNormLayer(filter_counts[1] * skip_concat, filter_counts[2], 3, 1, frn=frn),
            )
            self.layers3 = nn.Sequential(
                ConvNoTanhLayer(filter_counts[2] * skip_concat, 3, 3, 1)
            )
            self.conv1 = ConvLayer(filter_counts[1], filter_counts[1], 3, 1)
            self.conv2 = ConvLayer(filter_counts[2], filter_counts[2], 3, 1)

    def forward(self, x):
        if not self.use_skip:
            return self.layers(x)
        else:
            x, f_maps = x
            x = self.layers1(x)
            f_map = self.conv1(f_maps[1])
            x = torch.cat([x, f_map], dim=1)
            #x += f_map
            x = self.layers2(x)
            f_map = self.conv2(f_maps[0])
            x = torch.cat([x, f_map], dim=1)
            x = self.layers3(x)
            return x

class ReCoNetMobile(nn.Module):
    def __init__(self, frn=True, a=0.5, b=0.75, use_skip=False):
        super().__init__()
        self.use_skip = use_skip
        self.encoder = Encoder(a=a, b=b, frn=frn, use_skip=use_skip)
        self.decoder = Decoder(a=a, b=b, frn=frn, use_skip=use_skip)

    def forward(self, x):
        if not self.use_skip:
            x = self.encoder(x)
            features = x
            x = self.decoder(x)
        else:
            x, f_map = self.encoder(x)
            features = x
            x = self.decoder((x, f_map))
        return (features, x)
