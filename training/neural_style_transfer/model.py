from torch import nn
from frn import FRN, TLU


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



class Encoder(nn.Module):
    def __init__(self, a, b, frn=False):
        super().__init__()
        filter_counts = list(map(lambda x: int(a * x), [
            32, 48, 64
        ]))
        self.layers = nn.Sequential(
            ConvNormLayer(3, filter_counts[0], 3, 1, frn=frn),
            ConvNormLayer(filter_counts[0], filter_counts[1], 3, 2, frn=frn),
            ConvNormLayer(filter_counts[1], filter_counts[2], 3, 2, frn=frn),
        )
        res_layer_count = int(b * 4)
        [self.layers.add_module(f'{i + 3}', ResLayer(filter_counts[2], filter_counts[2], 3, frn=frn)) 
        for i in range(res_layer_count)]


    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, a, b, frn=False):
        super().__init__()
        filter_counts = list(map(lambda x: int(a * x), [
            64, 48, 32
        ]))
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvNormLayer(filter_counts[0], filter_counts[1], 3, 1, frn=frn),
            nn.Upsample(scale_factor=2),
            ConvNormLayer(filter_counts[1], filter_counts[2], 3, 1, frn=frn),
            ConvNoTanhLayer(filter_counts[2], 3, 3, 1)
        )

    def forward(self, x):
        return self.layers(x)


class ReCoNetMobile(nn.Module):
    def __init__(self, frn=True, a=0.5, b=0.75):
        super().__init__()
        self.encoder = Encoder(a=a, b=b, frn=frn)
        self.decoder = Decoder(a=a, b=b, frn=frn)


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
