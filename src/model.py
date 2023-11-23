import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        conv1 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=3,
            padding=1,
        )
        bn1 = nn.BatchNorm2d(output_dim)

        conv2 = nn.Conv2d(
            in_channels=output_dim,
            out_channels=output_dim,
            kernel_size=3,
            padding=1,
        )
        bn2 = nn.BatchNorm2d(output_dim)

        relu = nn.ReLU(inplace=True)

        layer_list = [
            conv1,
            bn1,
            relu,
            conv2,
            bn2,
            relu,
        ]
        self.layers = nn.Sequential(*layer_list)

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()

        self.conv = ConvBlock(input_dim, output_dim)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=2, stride=2, padding=0
        )
        self.conv = ConvBlock(2 * output_dim, output_dim)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, input_dim, output_dim, patch_height, patch_width) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.patch_height = patch_height
        self.patch_width = patch_width

        # Encoder
        self.e1 = EncoderBlock(input_dim, 32)
        self.e2 = EncoderBlock(32, 64)

        # Bottlenect
        self.b = ConvBlock(64, 128)

        # Decoder
        self.d1 = DecoderBlock(128, 64)
        self.d2 = DecoderBlock(64, 32)

        # Classifier
        self.conv = nn.Conv2d(
            in_channels=32, out_channels=output_dim, kernel_size=1, padding=0
        )

    def forward(self, inputs):
        # Encoder
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)

        # Bottlenect
        b = self.b(p2)

        # Decoder
        d1 = self.d1(b, s2)
        d2 = self.d2(d1, s1)

        # Classifier
        outputs = self.conv(d2)

        return outputs
