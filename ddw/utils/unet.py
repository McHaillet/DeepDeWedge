import random

import pytorch_lightning as pl
import torch
import yaml
from torch import nn

from .fourier import apply_fourier_mask_to_tomo
from .losses import data_consistency_loss, equivariance_loss
from .normalization import get_avg_model_input_mean_and_std_from_dataloader
from .rotation import rotate_vol, sample_grid_rotation


class LitUnet3D(pl.LightningModule):
    """
    PyTrochLightning 'wrapper' of a 3D U-Net. This class implements steps for model fitting, validation and logging. This class is the heart of the 'ddw fit-model' command.
    """

    def __init__(
        self,
        unet_params,
        adam_params,
        subtomo_size,
        lambda_=2.0,
    ):
        super().__init__()
        self.unet_params = unet_params
        self.adam_params = adam_params
        self.subtomo_size = subtomo_size
        self.lambda_ = lambda_
        self.unet = Unet3D(**self.unet_params)
        # self.ema = ExponentialMovingAverage(self.unet.parameters(), decay=0.995)
        self.save_hyperparameters()

    def forward(self, x):
        return self.unet(x.unsqueeze(1)).squeeze(
            1
        )  # unsqueeze to add channel dimension, squeeze to remove it

    def _sample_rotations(self, indices, deterministic):
        """
        Samples one grid rotation per volume (see ddw.utils.rotation.get_grid_rotations).
        Callers that need to later undo the same rotation (via _rotate_batch's 'inverse')
        must reuse the returned list rather than re-sampling by 'index': when 'deterministic'
        is False, sample_grid_rotation draws from the shared global 'random' state, so two
        separate calls - even with the same 'index' - are not guaranteed to agree.
        """
        return [sample_grid_rotation(int(index), deterministic) for index in indices]

    def _rotate_batch(self, vol_batch, rot_mats, inverse=False):
        """
        Rotates each volume in 'vol_batch' by the corresponding matrix in 'rot_mats' (from
        _sample_rotations). This is exact (no interpolation), so the output has exactly the
        same shape as the input and needs no cropping. If 'inverse' is True, applies the
        inverse rotation instead (the transpose of the signed permutation matrix, which
        exactly undoes rotate_vol for these grid-aligned rotations) - pass the *same*
        'rot_mats' used for the corresponding forward call so the two exactly cancel.
        """
        rotated = []
        for vol, rot_mat in zip(vol_batch, rot_mats):
            rotated.append(rotate_vol(vol, rot_mat.T if inverse else rot_mat))
        return torch.stack(rotated)

    def _step(self, batch, batch_idx, deterministic):
        subtomo0 = batch["subtomo0"]
        subtomo1 = batch["subtomo1"]
        ctf = batch["ctf"]

        # alternate which estimate is rotated + re-degraded to build the equivariance term's
        # model input ("x_hat2") vs. which (the *other*, independent-noise) estimate serves as
        # its target ("x_hat1"). The same choice also picks which of the two cross-wise
        # data-consistency pairings dc_loss evaluates this step (rather than always summing
        # both): only the "source" branch needs gradients for that, so the "target" branch -
        # only ever used detached, both here and for the equivariance loss below - is computed
        # under torch.no_grad() to skip unneeded gradient bookkeeping and speed things up
        # slightly.
        use_branch0_as_source = (
            (batch_idx % 2 == 0) if deterministic else (random.random() < 0.5)
        )
        if use_branch0_as_source:
            x_hat_source = self(subtomo0)
            with torch.no_grad():
                x_hat_target = self(subtomo1)
        else:
            x_hat_source = self(subtomo1)
            with torch.no_grad():
                x_hat_target = self(subtomo0)

        y_cross = subtomo1 if use_branch0_as_source else subtomo0
        dc_loss = data_consistency_loss(x_hat_source, y_cross, ctf)

        # sampled once and reused for both the forward and inverse rotation below, so they
        # are guaranteed to exactly cancel (see _sample_rotations)
        rot_mats = self._sample_rotations(batch["index"], deterministic)

        # rotate_vol/_rotate_batch is differentiable, but both estimates are detached here
        # anyway by design (standard equivariant-imaging stop-gradient): the gradient of
        # eq_loss should only flow through the second application of self() below
        x_hat_source_rot = self._rotate_batch(x_hat_source.detach(), rot_mats)

        # re-inject noise before re-masking, so the second pass's input isn't unrealistically
        # clean (out-of-distribution vs. what the model normally sees) - added pre-ctf so it's
        # zeroed in the missing wedge like real noise, not left as a tell. 'delta' makes up the
        # gap between the raw noise level and what the model's own two estimates still disagree
        # by (variances add) - recomputed every step, detached (a fixed scale, not learnable).
        # Same trick as IsoNet2's isonet2-n2n mode. x_hat_source/target are full-band (already
        # deconvolved/filled) estimates, unlike the band-limited subtomo0/1, so their raw diff
        # isn't comparable to noise_std - it'd be inflated by missing-wedge disagreement that's
        # genuine reconstruction uncertainty, not noise. Re-masking the diff by ctf first keeps
        # both sides of the subtraction on the same (observed) footing.
        noise_std = torch.std(subtomo0 - subtomo1) / 2**0.5
        masked_diff = apply_fourier_mask_to_tomo(x_hat_source.detach() - x_hat_target, ctf)
        new_noise_std = torch.std(masked_diff) / 2**0.5
        delta_noise_std = torch.sqrt(torch.abs(noise_std**2 - new_noise_std**2))
        # 'delta' is a post-mask target, but injection happens pre-mask (above) - ctf will
        # attenuate whatever we inject here too, so scale up by how much it attenuates a
        # flat-spectrum signal (Parseval: mean(ctf^2)) to compensate.
        ctf_attenuation = torch.sqrt((ctf**2).mean())
        pre_mask_noise_std = delta_noise_std / (ctf_attenuation + 1e-6)
        noisy_input = x_hat_source_rot + pre_mask_noise_std * torch.randn_like(x_hat_source_rot)

        z = apply_fourier_mask_to_tomo(noisy_input, ctf)
        x_double_hat = self(z)
        x_double_hat_unrot = self._rotate_batch(x_double_hat, rot_mats, inverse=True)
        eq_loss = equivariance_loss(x_double_hat_unrot, x_hat_target.detach(), ctf)

        loss = dc_loss + self.lambda_ * eq_loss
        return loss, dc_loss, eq_loss

    def training_step(self, batch, batch_idx):
        loss, dc_loss, eq_loss = self._step(batch, batch_idx, deterministic=False)
        # sync_dist=True: under multi-GPU DDP each rank only sees its own shard, so without
        # this the ModelCheckpoint callbacks that monitor "fitting_loss"/"val_loss" (see
        # fit_model.py) would select checkpoints based on a single rank's partial view of
        # the loss instead of the true value averaged across all ranks
        self.log("fitting_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("fitting_dc_loss", dc_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("fitting_eq_loss", eq_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, dc_loss, eq_loss = self._step(batch, batch_idx, deterministic=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_dc_loss", dc_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log("val_eq_loss", eq_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    # def on_before_zero_grad(self, optimizer) -> None:
    #     self.ema.update()

    def on_train_start(self) -> None:
        if self.current_epoch == 0:
            self.update_normalization()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.adam_params)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        return [optimizer]  # , [scheduler]

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric) -> None:
    #     if scheduler is not None:
    #         scheduler.step()

    def update_normalization(self):
        """
        Updates the average model input mean and standard deviation used to normalize the sub-tomograms.
        """
        loc, scale = get_avg_model_input_mean_and_std_from_dataloader(
            dataloader=self.trainer.train_dataloader, verbose=True
        )

        # update normalization in unet
        self.unet.normalization_loc = loc
        self.unet.normalization_scale = scale
        # update normalization in hparams
        self.unet_params["normalization_loc"] = loc
        self.unet_params["normalization_scale"] = scale
        self.update_hparam("unet_params", self.unet_params)
        self.log("normalization/loc", loc)
        self.log("normalization/scale", scale)

    def update_hparam(self, hparam, value):
        """
        Update a hyperparameter in the hparams.yaml file.
        """
        if not self.trainer.is_global_zero:
            # under DDP this hook runs on every rank, but the logger only writes
            # hparams.yaml on rank 0 (its log_hyperparams is @rank_zero_only), so on
            # other ranks the file is empty/not yet written - skip them here
            return
        logger = self.trainer.logger
        logdir = f"{logger.save_dir}/{logger.name}/version_{logger.version}"
        hparams_file = f"{logdir}/hparams.yaml"
        hparams = yaml.safe_load(open(hparams_file, "r"))
        hparams[hparam] = value
        with open(hparams_file, "w") as f:
            yaml.dump(hparams, f)


class Unet3D(torch.nn.Module):
    """
    PyTorch implementation of a 3D U-Net, which was inspired by the one used in the IsoNet software package (https://github.com/IsoNet-cryoET/IsoNet/tree/master/models/unet)
    """

    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        chans: int = 32,
        num_downsample_layers: int = 3,
        drop_prob: float = 0.0,
        residual: bool = True,
        normalization_loc: float = 0.0,
        normalization_scale: float = 1.0,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_downsample_layers = num_downsample_layers
        self.drop_prob = drop_prob
        self.residual = residual
        self.normalization_loc = normalization_loc
        self.normalization_scale = normalization_scale
        self.__init_layers__()

    @property
    def normalization_loc(self):
        return self._normalization_loc

    @normalization_loc.setter
    def normalization_loc(self, normalization_loc):
        self._normalization_loc = nn.parameter.Parameter(
            torch.tensor(normalization_loc), requires_grad=False
        )

    @property
    def normalization_scale(self):
        return self._normalization_scale

    @normalization_scale.setter
    def normalization_scale(self, normalization_scale):
        self._normalization_scale = nn.parameter.Parameter(
            torch.tensor(normalization_scale), requires_grad=False
        )

    def __init_layers__(self):
        self.down_blocks = nn.ModuleList(
            [DownConvBlock(self.in_chans, self.chans, self.drop_prob)]
        )
        self.down_samplers = nn.ModuleList([SpatialDownSampling(self.chans)])

        ch = self.chans
        for _ in range(self.num_downsample_layers - 1):
            self.down_blocks.append(DownConvBlock(ch, ch * 2, self.drop_prob))
            self.down_samplers.append(SpatialDownSampling(ch * 2))
            ch *= 2

        self.bottleneck = nn.Sequential(
            nn.Conv3d(ch, ch * 2, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(ch * 2, ch, kernel_size=(3, 3, 3), padding=1, bias=False),
        )

        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList([SpatialUpSampling(in_chans=ch, out_chans=ch)])

        for _ in range(self.num_downsample_layers - 1):
            self.up_blocks.append(UpConvBlock(2 * ch, ch, self.drop_prob))
            self.upsamplers.append(SpatialUpSampling(in_chans=ch, out_chans=ch // 2))
            ch //= 2
        self.up_blocks.append(UpConvBlock(2 * ch, ch, self.drop_prob))

        self.final_conv = nn.Conv3d(
            ch, self.out_chans, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False
        )

    def normalize(self, volume: torch.Tensor) -> torch.Tensor:
        return (volume - self.normalization_loc) / (self.normalization_scale + 1e-6)

    def denormalize(self, volume: torch.Tensor) -> torch.Tensor:
        return volume * (self.normalization_scale + 1e-6) + self.normalization_loc

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        volume = self.normalize(volume)

        stack = []
        output = volume

        # apply down-sampling layers
        for block, downsampler in zip(self.down_blocks, self.down_samplers):
            output = block(output)
            stack.append(output)  # save intermediate outputs for skip connections
            output = downsampler(output)

        output = self.bottleneck(output)

        # apply up-sampling layers
        for upsampler, block in zip(self.upsamplers, self.up_blocks):
            output = upsampler(output, cat=stack.pop())
            output = block(output)

        output = self.final_conv(output)
        if self.residual:
            output = output + volume

        output = self.denormalize(output)
        return output


class DownConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(out_chans, out_chans, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(out_chans, out_chans, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.layers(volume)


class UpConvBlock(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv3d(in_chans, in_chans // 2, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(in_chans // 2, in_chans // 2, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv3d(in_chans // 2, out_chans, kernel_size=(3, 3, 3), padding=1, bias=False),
            nn.Dropout3d(drop_prob),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        return self.layers(volume)


class SpatialDownSampling(nn.Module):
    def __init__(self, chans: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(chans, chans, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
        )

    def forward(self, volume):
        return self.layers(volume)


class SpatialUpSampling(nn.Module):
    def __init__(self, in_chans: int, out_chans: int, drop_prob=0.0):
        super().__init__()
        # Nearest-neighbor upsampling followed by a stride-1 conv, instead of a
        # strided ConvTranspose3d: with kernel_size=3 not divisible by stride=2, the
        # transposed conv gives uneven kernel overlap across output voxels, a
        # deterministic period-2 "checkerboard" artifact (see Odena et al., "Deconvolution
        # and Checkerboard Artifacts"). Resize+conv has no stride mismatch to produce
        # that unevenness.
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv3d(
            in_chans, out_chans, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, bias=False
        )
        self.activation = nn.LeakyReLU(negative_slope=0.05, inplace=True)

    def forward(self, volume: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        output = self.conv(self.upsample(volume))
        output = torch.cat([output, cat], dim=1)
        output = self.activation(output)
        return output
