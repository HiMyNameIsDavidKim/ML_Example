import torch
from torchsummary import summary


class rgb2pentile(torch.nn.Module):
    def __init__(self):
        super().__init__()

        kernel_size = (3, 3)
        padding = 1
        channels = 256
        relu_inplace = True

        self.layers_f = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=2, out_channels=channels, kernel_size=kernel_size, padding=padding,
                            padding_mode='replicate'),
            torch.nn.ReLU(inplace=relu_inplace),

            torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding,
                            padding_mode='replicate'),
            torch.nn.ReLU(inplace=relu_inplace),

            torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding,
                            padding_mode='replicate'),
            torch.nn.ReLU(inplace=relu_inplace)
        )
        self.layers_b = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding,
                            padding_mode='replicate'),
            torch.nn.ReLU(inplace=relu_inplace),

            torch.nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=kernel_size, padding=padding,
                            padding_mode='replicate')
        )

    def forward(self, x, pooling_size=(2, 4)):
        x = self.layers_f(x)
        x = torch.nn.MaxPool2d(kernel_size=pooling_size)(x)
        x = self.layers_b(x)

        return x


pentileR = rgb2pentile()
summary(pentileR, (2, 192, 192))
