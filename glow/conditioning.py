import torch
import torch.nn as nn

import glow.modules as modules


class DeepSpeechEncoder(nn.Module):
    """
    First layers for DeepSpeech2:
        https://arxiv.org/pdf/1512.02595.pdf

    The code is rewritten based on:
        https://github.com/SeanNaren/deepspeech.pytorch

    Output:     Last hidden state of RNN
    """

    def __init__(
        self,
        input_shape=(1, 1, 213, 80),
        kernel_size=[(41, 11), (21, 11)],
        stride=[(2, 2), (2, 1)],
        padding=[(20, 5), (10, 5)],
        channels=32,
        rnn_hidden=256,
        rnn_layers=2,
        bidirectional=True,
        output_only_last_state=True,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.stride = stride
        self.padding = stride
        self.channels = channels
        self.bidirectional = bidirectional
        self.output_only_last_state = output_only_last_state

        self.conv = nn.Sequential(
            nn.Conv2d(
                1,
                channels,
                kernel_size=kernel_size[0],
                stride=stride[0],
                padding=padding[0],
            ),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size[1],
                stride=stride[1],
                padding=padding[1],
            ),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
        )
        enc_out = self.encoder_dim_out()
        channels = enc_out[1]
        steps = enc_out[2]
        feats = enc_out[3]

        rnn_in = feats * channels

        self.rnn = nn.GRU(
            input_size=rnn_in,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=bidirectional,
        )

        if bidirectional:
            self.out_size = rnn_hidden * 2
        else:
            self.out_size = rnn_hidden

    def encoder_dim_out(self):
        x = torch.rand(self.input_shape).unsqueeze(0)  # batch
        return self.conv(x).shape

    def forward(self, x):
        z = self.conv(x.unsqueeze(1))
        z = z.permute(
            0, 2, 1, 3
        )  # -> (B, N, channels, Feats): get feature dims to the right (spec values and channels)
        z = z.flatten(
            2
        )  # -> (B, N, Feats*): flatten out spec+channels -> rnn_in_features

        out, h = self.rnn(z)
        if self.output_only_last_state:
            out = out[:, -1]
        return torch.tanh(out)


class EncoderHead(nn.Module):
    """
    Features in NN are going to
    """

    def __init__(
        self, in_channels, out_channels, hidden_channels, condition_input, timesteps
    ):
        super().__init__()
        self.condition_input = condition_input
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.timesteps = timesteps

        self.mlp = nn.Linear(condition_input, in_channels)
        self.audio_to_glow = nn.Conv2d(
            in_channels=1, out_channels=timesteps, kernel_size=1, stride=1
        )

        self.conv = nn.Sequential(
            modules.Conv2d(in_channels + in_channels, hidden_channels),
            nn.ReLU(inplace=False),
            modules.Conv2d(hidden_channels, hidden_channels, kernel_size=[1, 1]),
            nn.ReLU(inplace=False),
            modules.Conv2dZeros(hidden_channels, out_channels),
        )

    def forward(self, z1, audio_features):
        audio_features = self.mlp(audio_features)
        audio_features = audio_features.unsqueeze(1).unsqueeze(-1)
        audio_features = self.audio_to_glow(audio_features)
        audio_features = audio_features.permute(0, 2, 1, 3)
        z = torch.cat((z1, audio_features), dim=1)
        return self.conv(z)


if __name__ == "__main__":

    audio_features = torch.rand((32, 1, 160, 80))
    z1 = torch.rand((32, 140, 64, 1))
    z2 = torch.rand((32, 140, 32, 1))
    z3 = torch.rand((32, 140, 16, 1))

    encoder = DeepSpeechEncoder(input_shape=audio_features.shape[1:])

    head1 = EncoderHead(
        condition_input=encoder.out_size,
        in_channels=z1.shape[1],
        out_channels=z1.shape[1],
        hidden_channels=512,
        timesteps=z1.shape[2],
    )

    head2 = EncoderHead(
        condition_input=encoder.out_size,
        in_channels=z2.shape[1],
        out_channels=z2.shape[1],
        hidden_channels=512,
        timesteps=z2.shape[2],
    )

    head3 = EncoderHead(
        condition_input=encoder.out_size,
        in_channels=z3.shape[1],
        out_channels=z3.shape[1],
        hidden_channels=512,
        timesteps=z3.shape[2],
    )

    audio_out = encoder(audio_features)

    o1 = head1(z1, audio_out)
    o2 = head2(z2, audio_out)
    o3 = head3(z3, audio_out)
