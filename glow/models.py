import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from glow.conditioning import DeepSpeechEncoder, EncoderHead
from tqdm import tqdm

from . import modules, thops, utils

# class f(nn.Module):
#     def __init__(self, in_channels, out_channels, hidden_channels, condition_channels):
#         self.conv = nn.Sequential(
#             modules.Conv2d(in_channels + condition_channels, hidden_channels),
#             nn.ReLU(inplace=False),
#             modules.Conv2d(hidden_channels, hidden_channels, kernel_size=[1, 1]),
#             nn.ReLU(inplace=False),
#             modules.Conv2dZeros(hidden_channels, out_channels),
#         )

#     def forward(self, z, cond):
#         z_cond = torch.cat((z, cond), dim=1)
#         return self.conv(z_cond)
def f(in_channels, out_channels, hidden_channels, cond_channels):
    return nn.Sequential(
        modules.Conv2d(in_channels + cond_channels, hidden_channels),
        nn.ReLU(inplace=False),
        modules.Conv2d(hidden_channels, hidden_channels, kernel_size=[1, 1]),
        nn.ReLU(inplace=False),
        modules.Conv2dZeros(hidden_channels, out_channels),
    )


class f_new(nn.Module):
    """
    input_size:  (glow) channels
    """

    def __init__(self, input_size, output_size, hidden_size, condition_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.condition_size = condition_size
        self.output_size = output_size
        self.rnn = nn.GRUCell(
            input_size=input_size + condition_size, hidden_size=hidden_size
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.inited = False

    def initialize_state(self, z):
        self.hidden = z.data.new(z.size(0), self.hidden_size).zero_()

    def forward(self, z, condition):
        if not self.inited:
            self.initialize_state(z)
            self.inited = True

        rnn_input = torch.cat((z, condition), dim=1).squeeze(-1).squeeze(-1)
        self.hidden = self.rnn(rnn_input, self.hidden)
        return self.linear(self.hidden).unsqueeze(-1).unsqueeze(-1)


class f_old2(nn.Module):
    """
    input_size:  (glow) channels
    """

    def __init__(self, input_size, output_size, hidden_size, condition_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.condition_size = condition_size
        self.output_size = output_size
        self.rnn = nn.GRUCell(
            input_size=input_size + condition_size, hidden_size=hidden_size
        )
        self.linear = nn.Linear(hidden_size, output_size)
        self.inited = False

    def initialize_state(self, z):
        self.hidden = z.data.new(z.size(0), self.hidden_size).zero_()

    def forward(self, z, condition):
        if not self.inited:
            self.initialize_state(z)
            self.inited = True

        rnn_input = torch.cat((z, condition), dim=1).squeeze(-1).squeeze(-1)
        self.hidden = self.rnn(rnn_input, self.hidden)
        return self.linear(self.hidden).unsqueeze(-1).unsqueeze(-1)


class f_old(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, cond_channels):
        super().__init__()
        # self.spectrogram_conv = nn.Sequential(
        #     nn.Conv2d(1, 72, (1, 3), stride=(1, 2), padding=(0, 1)),
        #     nn.ReLU(),
        #     nn.Conv2d(72, 108, (1, 3), stride=(1, 2), padding=(0, 1)),
        #     nn.ReLU(),
        #     nn.Conv2d(108, 162, (1, 3), stride=(1, 2), padding=(0, 1)),
        #     nn.ReLU(),
        #     nn.Conv2d(162, 243, (1, 3), stride=(1, 2), padding=(0, 1)),
        #     nn.ReLU(),
        #     nn.Conv2d(243, cond_channels, (1, 2), stride=(1, 2), padding=(0, 0)),
        #     nn.ReLU(),
        # )
        self.in_channels = in_channels
        self.gru = nn.GRU(80 * 11, 256, 2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(256, cond_channels)

        self.conv = nn.Sequential(
            modules.Conv2d(in_channels + cond_channels, hidden_channels),
            nn.ReLU(inplace=False),
            modules.Conv2d(hidden_channels, hidden_channels, kernel_size=[1, 1]),
            nn.ReLU(inplace=False),
            modules.Conv2dZeros(hidden_channels, out_channels),
        )

    def forward(self, input_, audio_features):
        # import pdb

        # pdb.set_trace()

        o, h = self.gru(audio_features)
        condition = self.fc(o).unsqueeze(3)

        fix_cond = (
            condition[:, :1, :, :]
            .permute(0, 2, 1, 3)
            .expand(-1, -1, input_.shape[2], -1)
        )

        return self.conv(torch.cat((fix_cond, input_), dim=1))


class FlowStep(nn.Module):
    FlowCoupling = ["additive", "affine"]
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev),
    }

    def __init__(
        self,
        in_channels,
        hidden_channels,
        cond_channels,
        actnorm_scale=1.0,
        flow_permutation="invconv",
        flow_coupling="additive",
        LU_decomposed=False,
        L=1,
        K=1,
        timesteps=1,
    ):
        # check configures
        assert (
            flow_coupling in FlowStep.FlowCoupling
        ), "flow_coupling should be in `{}`".format(FlowStep.FlowCoupling)

        assert (
            flow_permutation in FlowStep.FlowPermutation
        ), "float_permutation should be in `{}`".format(FlowStep.FlowPermutation.keys())

        super().__init__()

        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling

        # Custom
        self.L = L  # Which multiscale layer this module is in
        self.K = K  # Which step of flow in self.L
        self.cond_channels = cond_channels
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # 1. actnorm
        self.actnorm = modules.ActNorm2d(in_channels, actnorm_scale)

        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = modules.InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed
            )
        elif flow_permutation == "shuffle":
            self.shuffle = modules.Permute2d(in_channels, shuffle=True)
        else:
            self.reverse = modules.Permute2d(in_channels, shuffle=False)

        # 3. coupling
        if flow_coupling == "additive":
            self.f = f(
                in_channels // 2, in_channels // 2, hidden_channels, cond_channels
            )
        elif flow_coupling == "affine":
            self.f = f(in_channels // 2, in_channels, hidden_channels, cond_channels)
            # self.f = EncoderHead(
            #     in_channels=in_channels // 2,
            #     out_channels=in_channels,
            #     hidden_channels=hidden_channels,
            #     condition_input=self.cond_channels,
            #     timesteps=timesteps,
            # )

    def forward(self, input_, audio_features, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input_, audio_features, logdet)
        else:
            return self.reverse_flow(input_, audio_features, logdet)

    def normal_flow(self, input_, audio_features, logdet):
        assert input_.size(1) % 2 == 0, input_.shape
        # 1. actnorm

        z, logdet = self.actnorm(input_, logdet=logdet, reverse=False)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False
        )
        # 3. coupling
        z1, z2 = thops.split_feature(z, "split")

        z1_cond = torch.cat((z1, audio_features), dim=1)

        if self.flow_coupling == "additive":
            z2 = z2 + self.f(z1_cond)
        elif self.flow_coupling == "affine":
            h = self.f(z1_cond)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = thops.cat_feature(z1, z2)
        return z, audio_features, logdet

    def reverse_flow(self, input_, audio_features, logdet):
        assert input_.size(1) % 2 == 0, input_.shape
        # 1.coupling
        z1, z2 = thops.split_feature(input_, "split")
        z1_cond = torch.cat((z1, audio_features), dim=1)

        if self.flow_coupling == "additive":
            z2 = z2 - self.f(z1_cond)
        elif self.flow_coupling == "affine":
            h = self.f(z1_cond)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = thops.cat_feature(z1, z2)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True
        )
        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, audio_features, logdet


class FlowNet(nn.Module):
    def __init__(
        self,
        image_shape,
        hidden_channels,
        cond_channels,
        K,
        L,
        actnorm_scale=1.0,
        flow_permutation="invconv",
        flow_coupling="additive",
        LU_decomposed=False,
        spec_frames=40,
        n_mels=80,
    ):
        """
                             K                                      K
        --> [Squeeze] -> [FlowStep] -> [Split] -> [Squeeze] -> [FlowStep]
               ^                           v
               |          (L - 1)          |
               + --------------------------+
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        self.L = L
        H, W, C = image_shape  # Timeframes, 1, Features
        N = cond_channels
        # self.conditionNet = DeepSpeechEncoder(input_shape=(1, spec_frames, n_mels))

        for l in range(L):
            # 1. Squeeze
            # C, H, W, N = C * 2, H, W, N * 2  # C: features, H: timesteps
            # self.layers.append(modules.SqueezeLayer(factor=2))
            # self.output_shapes.append([-1, C, H, W])
            # 2. K FlowStep
            for k in range(K):
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        cond_channels=N,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                        L=l,
                        K=k,
                        timesteps=H,
                    )
                )
                self.output_shapes.append([-1, C, H, W])
            # 3. Split2d
            if l < L - 1:
                self.layers.append(modules.Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def forward(self, input_, audio_features, logdet=0.0, reverse=False, eps_std=None):
        # audio_features = self.conditionNet(audio_features)  # Spectrogram

        if not reverse:
            return self.encode(input_, audio_features, logdet)
        else:
            return self.decode(input_, audio_features, eps_std)

    def encode(self, z, audio_features, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            z, audio_features, logdet = layer(z, audio_features, logdet, reverse=False)
        return z, logdet

    def decode(self, z, audio_features, eps_std=None):
        for layer in self.layers:
            if isinstance(layer, modules.SqueezeLayer):
                audio_features = layer.squeeze_cond(audio_features)

        for layer in reversed(self.layers):
            if isinstance(layer, modules.Split2d):
                z, audio_features, logdet = layer(
                    z, audio_features, logdet=0, reverse=True, eps_std=eps_std
                )
            else:
                z, audio_features, logdet = layer(
                    z, audio_features, logdet=0, reverse=True
                )
        return z


class Glow(nn.Module):
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()

    def __init__(self, hparams):
        super().__init__()
        self.flow = FlowNet(
            image_shape=hparams.Glow.image_shape,
            hidden_channels=hparams.Glow.hidden_channels,
            cond_channels=hparams.Glow.cond_channels,
            K=hparams.Glow.K,
            L=hparams.Glow.L,
            actnorm_scale=hparams.Glow.actnorm_scale,
            flow_permutation=hparams.Glow.flow_permutation,
            flow_coupling=hparams.Glow.flow_coupling,
            LU_decomposed=hparams.Glow.LU_decomposed,
            spec_frames=hparams.Glow.spec_frames,
            n_mels=hparams.Glow.n_mels,
        )
        self.hparams = hparams
        self.y_classes = hparams.Glow.y_classes
        self.x_shape = [
            self.flow.output_shapes[-1][1] * 2,
            self.flow.output_shapes[-1][2],
            self.flow.output_shapes[-1][3],
        ]
        

    def forward(
        self,
        x=None,
        audio_features=None,
        y_onehot=None,
        z=None,
        eps_std=None,
        reverse=False,
    ):
        if not reverse:
            return self.normal_flow(x, audio_features.unsqueeze(-1), y_onehot)
        else:
            return self.reverse_flow(z, audio_features.unsqueeze(-1), y_onehot, eps_std)

    def get_sample(self, batch_size, eps_std=None):
        eps_std = eps_std or 1
        x_shape = [batch_size] + self.x_shape
        return torch.normal(
            mean=torch.zeros(x_shape), std=torch.ones(x_shape) * eps_std
        )

    def normal_flow(self, x, audio_features, y_onehot):
        pixels = thops.pixels(x)
        z = x
        logdet = torch.zeros_like(x[:, 0, 0, 0])
        z, objective = self.flow(z, audio_features, logdet=logdet, reverse=False)

        objective += modules.GaussianDiag.logp_simplified(z)

        # return
        nll = (-objective) / float(np.log(2.0) * pixels)
        return z, nll, None

    def reverse_flow(self, z, audio_features, eps_std, y_onehot=None):
        with torch.no_grad():
            if z is None:
                z = self.get_sample(audio_features.shape[0], eps_std)
            x = self.flow(z, audio_features, eps_std=eps_std, reverse=True)
        return x

    def set_actnorm_init(self, inited=True):
        for name, m in self.named_modules():
            if m.__class__.__name__.find("ActNorm") >= 0:
                m.inited = inited

    def generate_z(self, img):
        self.eval()
        B = self.hparams.Train.batch_size
        x = img.unsqueeze(0).repeat(B, 1, 1, 1).cuda()
        z, _, _ = self(x)
        self.train()
        return z[0].detach().cpu().numpy()

    def generate_attr_deltaz(self, dataset):
        assert "y_onehot" in dataset[0]
        self.eval()
        with torch.no_grad():
            B = self.hparams.Train.batch_size
            N = len(dataset)
            attrs_pos_z = [[0, 0] for _ in range(self.y_classes)]
            attrs_neg_z = [[0, 0] for _ in range(self.y_classes)]
            for i in tqdm(range(0, N, B)):
                j = min([i + B, N])
                # generate z for data from i to j
                xs = [dataset[k]["x"] for k in range(i, j)]
                while len(xs) < B:
                    xs.append(dataset[0]["x"])
                xs = torch.stack(xs).cuda()
                zs, _, _ = self(xs)
                for k in range(i, j):
                    z = zs[k - i].detach().cpu().numpy()
                    # append to different attrs
                    y = dataset[k]["y_onehot"]
                    for ai in range(self.y_classes):
                        if y[ai] > 0:
                            attrs_pos_z[ai][0] += z
                            attrs_pos_z[ai][1] += 1
                        else:
                            attrs_neg_z[ai][0] += z
                            attrs_neg_z[ai][1] += 1
                # break
            deltaz = []
            for ai in range(self.y_classes):
                if attrs_pos_z[ai][1] == 0:
                    attrs_pos_z[ai][1] = 1
                if attrs_neg_z[ai][1] == 0:
                    attrs_neg_z[ai][1] = 1
                z_pos = attrs_pos_z[ai][0] / float(attrs_pos_z[ai][1])
                z_neg = attrs_neg_z[ai][0] / float(attrs_neg_z[ai][1])
                deltaz.append(z_pos - z_neg)
        self.train()
        return deltaz

    @staticmethod
    def loss_generative(nll):
        # Generative loss
        return torch.mean(nll)

    @staticmethod
    def loss_multi_classes(y_logits, y_onehot):
        if y_logits is None:
            return 0
        else:
            return Glow.BCE(y_logits, y_onehot.float())

    @staticmethod
    def loss_class(y_logits, y):
        if y_logits is None:
            return 0
        else:
            return Glow.CE(y_logits, y.long())


class AutoregressiveGlow(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.glow = Glow(hparams)
        self.hidden_size = hparams.Glow.cond_hidden_size
        self.output_size = hparams.Glow.image_shape[2]
        self.rnn = nn.LSTMCell(hparams.Glow.n_mels + self.output_size, self.hidden_size)
        self.rnn_initialized = False

    def forward(
        self,
        x=None,
        audio_features=None,
        y_onehot=None,
        z=None,
        eps_std=None,
        reverse=False,
    ):
        self.hidden_input = (
            audio_features.data.new(audio_features.size(0), self.hidden_size).zero_(),
            audio_features.data.new(audio_features.size(0), self.hidden_size).zero_(),
        )
        face_outputs = []
        audio_len = audio_features.size(2)
        audio_features = audio_features.unsqueeze(-1)

        first_x = audio_features.data.new(
            audio_features.size(0), self.output_size, 1, 1
        ).zero_()

        if not reverse:
            x = torch.cat((first_x, x), dim=2)
            nlls = torch.zeros(audio_features.shape[0]).to(audio_features.device)
            assert x.size(2) == audio_len + 1, (x.shape, audio_features.shape)
            while len(face_outputs) < audio_len:
                time = len(face_outputs)
                input_ = audio_features[:, :, time, 0]
                prev_face = x[:, :, time, 0]

                self.hidden_input = self.rnn(
                    torch.cat((input_, prev_face), dim=1), self.hidden_input
                )
                face_output = x[:, :, time + 1 : time + 2]

                z, nll, _ = self.glow(
                    x=face_output,
                    audio_features=self.hidden_input[0].unsqueeze(-1),
                    y_onehot=y_onehot,
                )
                nlls += nll
                face_outputs.append(z)

            output = torch.cat(face_outputs, dim=2)
            return output, nlls / audio_len, None

        else:
            if z is not None:
                assert z.size(2) == audio_len, (z.shape, audio_features.shape)

            z_input = None
            while len(face_outputs) < audio_len:
                time = len(face_outputs)
                input_ = audio_features[:, :, time, 0]

                if z is not None:
                    z_input = z[:, :, time : time + 1]

                if not face_outputs:
                    prev_face = first_x.squeeze(-1).squeeze(-1)
                else:
                    prev_face = face_outputs[-1].squeeze(-1).squeeze(-1)

                self.hidden_input = self.rnn(
                    torch.cat((input_, prev_face), dim=1), self.hidden_input
                )

                x = self.glow(
                    z=z_input, audio_features=self.hidden_input[0].unsqueeze(-1), eps_std=eps_std, y_onehot=y_onehot, reverse=True
                )

                face_outputs.append(x)
            output = torch.cat(face_outputs, dim=2)
            return output
