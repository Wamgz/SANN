import torch
from torch import nn
from torch.nn import functional as F

from .resnet import make_layer, BasicBlock

class SpoofDecoder(nn.Module):
    def __init__(
        self,
        conv2x2_in=(512, 512, 256, 128, 64),
        conv2x2_out=(512, 256, 128, 64, 64),
        
        conv1x1_in=(1024, 512, 256, 128, 64),
        conv1x1_out=(512, 256, 128, 64, 3),
    ):
        super().__init__()
        self.num_ins = 5

        self.deres_layers = []
        self.conv2x2 = []
        self.conv1x1 = []
        for i in range(self.num_ins):  # 43210
            deres_layer = make_layer(
                BasicBlock,
                inplanes=conv1x1_in[i],
                planes=conv1x1_out[i],
                blocks=2,
                stride=1,
                dilation=1,
                norm_layer=nn.InstanceNorm2d,
            )
            conv2x2 = nn.Sequential(
                nn.Conv2d(conv2x2_in[i], out_channels=conv2x2_out[i], kernel_size=2),
                nn.InstanceNorm2d(conv2x2_out[i]),
                nn.ReLU(),
            )
            conv1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=conv1x1_in[i],
                    out_channels=conv1x1_out[i],
                    kernel_size=1,
                ),
                nn.InstanceNorm2d(conv1x1_out[i]),
            )
            self.deres_layers.append(deres_layer)
            self.conv2x2.append(conv2x2)
            self.conv1x1.append(conv1x1)

        self.deres_layers: nn.ModuleList = nn.ModuleList(self.deres_layers)
        self.conv2x2: nn.ModuleList = nn.ModuleList(self.conv2x2)
        self.conv1x1: nn.ModuleList = nn.ModuleList(self.conv1x1)


    def forward(self, inputs):
        outs = []
        out = inputs[-1]
        outs.append(out)
        for i in range(self.num_ins):
            if i < 4:
                out = F.interpolate(out, scale_factor=2, mode="nearest")
            out = F.pad(out, [0, 1, 0, 1])
            out = self.conv2x2[i](out)

            if i < 4:
                out = torch.cat([out, inputs[-i - 2]], dim=1)
            identity = self.conv1x1[i](out)
            out = self.deres_layers[i](out) + identity
            outs.append(out)
        outs[-1] = torch.tanh(outs[-1])

        return outs

if __name__ == '__main__':
    # (torch.Size([2, 64, 128, 128]), torch.Size([2, 64, 64, 64]), torch.Size([2, 128, 32, 32]), torch.Size([2, 256, 16, 16]), torch.Size([2, 512, 8, 8]))
    # inputs = (torch.randn([2, 64, 128, 128]), torch.randn([2, 64, 64, 64]), torch.randn([2, 128, 32, 32]), torch.randn([2, 256, 16, 16]), torch.randn([2, 512, 8, 8]))
    batch = 1
    x1 = torch.randn([batch, 64, 256, 256])
    x2 = torch.randn([batch, 128, 128, 128])
    x3 = torch.randn([batch, 256, 64, 64])
    x4 = torch.randn([batch, 512, 32, 32])
    x_spoof = torch.randn([batch, 512, 16, 16])
    inputs = [x1, x2, x3, x4, x_spoof]
    decoder = SpoofDecoder()
    outs = decoder(inputs)
    for out in outs:
        print(out.shape)