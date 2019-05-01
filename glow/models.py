import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from glow.conditioning import DeepSpeechEncoder, EncoderHead
from tqdm import tqdm

from . import modules, thops, utils


class f(nn.Module):
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
        actnorm_scale=1.0,
        flow_permutation="invconv",
        flow_coupling="additive",
        LU_decomposed=False,
        cond_channels=None,
        L=1,
        K=1,
        condition_input=256,
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
        self.condition_input = condition_input

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
            # self.f = f(in_channels // 2, in_channels, hidden_channels, cond_channels)
            self.f = EncoderHead(
                in_channels=in_channels // 2,
                out_channels=in_channels,
                hidden_channels=hidden_channels,
                condition_input=condition_input,
                timesteps=timesteps,
            )

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
        if self.flow_coupling == "additive":
            z2 = z2 + self.f(z1, audio_features)
        elif self.flow_coupling == "affine":
            h = self.f(z1, audio_features)
            shift, scale = thops.split_feature(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = thops.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = thops.cat_feature(z1, z2)
        return z, logdet

    def reverse_flow(self, input_, audio_features, logdet):
        assert input_.size(1) % 2 == 0, input_.shape
        # 1.coupling
        z1, z2 = thops.split_feature(input_, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.f(z1, audio_features)
        elif self.flow_coupling == "affine":
            h = self.f(z1, audio_features)
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
        return z, logdet


class FlowNet(nn.Module):
    def __init__(
        self,
        image_shape,
        hidden_channels,
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

        self.conditionNet = DeepSpeechEncoder(input_shape=(1, spec_frames, n_mels))

        for l in range(L):
            # 1. Squeeze
            # C, H, W = C * 2, H // 2, W  # C: features, H: timesteps
            # self.layers.append(modules.SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])
            # 2. K FlowStep
            for k in range(K):
                self.layers.append(
                    FlowStep(
                        in_channels=C,
                        hidden_channels=hidden_channels,
                        actnorm_scale=actnorm_scale,
                        flow_permutation=flow_permutation,
                        flow_coupling=flow_coupling,
                        LU_decomposed=LU_decomposed,
                        L=l,
                        K=k,
                        condition_input=self.conditionNet.out_size,
                        timesteps=H,
                    )
                )
                self.output_shapes.append([-1, C, H, W])
            # 3. Split2d
            # if l < L - 1:
            #     self.layers.append(modules.Split2d(num_channels=C))
            #     self.output_shapes.append([-1, C // 2, H, W])
            #     C = C // 2

    def forward(self, input_, audio_features, logdet=0.0, reverse=False, eps_std=None):
        audio_features = self.conditionNet(audio_features)  # Spectrogram

        if not reverse:
            return self.encode(input_, audio_features, logdet)
        else:
            return self.decode(input_, audio_features, eps_std)

    def encode(self, z, audio_features, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            if isinstance(layer, FlowStep):
                z, logdet = layer(z, audio_features, logdet, reverse=False)
            else:
                z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, audio_features, eps_std=None):
        for layer in reversed(self.layers):
            if isinstance(layer, modules.Split2d):
                z, logdet = layer(z, logdet=0, reverse=True, eps_std=eps_std)
            elif isinstance(layer, FlowStep):
                z, logdet = layer(z, audio_features, logdet=0, reverse=True)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z


class Glow(nn.Module):
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()

    def __init__(self, hparams):
        super().__init__()
        self.flow = FlowNet(
            image_shape=hparams.Glow.image_shape,
            hidden_channels=hparams.Glow.hidden_channels,
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

        # for prior
        if hparams.Glow.learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top = modules.Conv2dZeros(C * 2, C * 2)

        if hparams.Glow.y_condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = modules.LinearZeros(hparams.Glow.y_classes, 2 * C)
            self.project_class = modules.LinearZeros(C, hparams.Glow.y_classes)
        # register prior hidden
        num_device = len(utils.get_proper_device(hparams.Device.glow, False))
        assert hparams.Train.batch_size % num_device == 0
        self.register_parameter(
            "prior_h",
            nn.Parameter(
                torch.zeros(
                    [
                        hparams.Train.batch_size // num_device,
                        self.flow.output_shapes[-1][1] * 2,
                        self.flow.output_shapes[-1][2],
                        self.flow.output_shapes[-1][3],
                    ]
                )
            ),
        )

    def prior(self, y_onehot=None):
        B, C = self.prior_h.size(0), self.prior_h.size(1)
        h = self.prior_h.detach().clone()
        assert torch.sum(h) == 0.0
        if self.hparams.Glow.learn_top:
            h = self.learn_top(h)
        if self.hparams.Glow.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot).view(B, C, 1, 1)
            h += yp
        return thops.split_feature(h, "split")

    def forward(
        self,
        x=None,
        audio_features=None,
        y_onehot=None,
        z=None,
        eps_std=None,
        reverse=False,
    ):
        face_outputs = []
        nlls = torch.zeros(audio_features.shape[0]).to(audio_features.device)
        audio_len = audio_features.size(1)

        while len(face_outputs) < audio_len:
            time = len(face_outputs)
            input_ = audio_features[:, time : time + 1]

            if not reverse:
                face_output = x[:, :, time : time + 1]
                z, nll, _ = self.normal_flow(face_output, input_, y_onehot)
            else:
                z, nll, _ = self.reverse_flow(z, input_, eps_std, y_onehot)
            nlls += nll
            face_outputs.append(z)

        output = torch.cat(face_outputs, dim=2)
        return output, nlls / audio_len, None

    def normal_flow(self, x, audio_features, y_onehot=None):
        pixels = thops.pixels(x)
        # z = x + torch.normal(
        #     mean=torch.zeros_like(x), std=torch.ones_like(x) * (1.0 / 256.0)
        # )
        z = x
        logdet = torch.zeros_like(x[:, 0, 0, 0])
        # logdet += float(-np.log(256.0) * pixels)
        # encode

        z, objective = self.flow(z, audio_features, logdet=logdet, reverse=False)
        # prior
        mean, logs = self.prior(y_onehot)

        objective += modules.GaussianDiag.logp(mean, logs, z)

        if self.hparams.Glow.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        # return
        nll = (-objective) / float(np.log(2.0) * pixels)
        return z, nll, y_logits

    def reverse_flow(self, z, audio_features, eps_std, y_onehot=None):
        with torch.no_grad():
            mean, logs = self.prior(y_onehot)
            if z is None:
                z = modules.GaussianDiag.sample(mean, logs, eps_std)
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
