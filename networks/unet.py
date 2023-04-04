
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=3, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # torch.Size([5, 1024, 16, 16]) torch.Size([5, 512, 32, 32]) torch.Size([5, 256, 64, 64]) torch.Size([5, 128, 128, 128]) torch.Size([5, 64, 256, 256])
        # print(x5.shape, x4.shape, x3.shape, x2.shape, x1.shape)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    x = torch.randn(5, 3, 256, 256).cuda()
    encoder = UNetEncoder(3, True).cuda()
    out = encoder(x)
    print(out[0].shape, out[1].shape, out[2].shape, out[3].shape, out[4].shape)
    decoder = UNetDecoder(3, True).cuda()
    out = decoder(*out)
    print(out.shape)