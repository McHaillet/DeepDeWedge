import torch
import tqdm
from torch.utils.data import DataLoader, TensorDataset

from .mrctools import load_mrc_data
from .subtomos import extract_subtomos


def get_avg_model_input_mean_and_std(tomo_file, subtomo_size, subtomo_extraction_strides, standardize, batch_size, num_workers, batches=None, verbose=False):
    """
    Computes the average mean and standard deviation of raw sub-tomograms extracted from
    'tomo_file' - i.e. of the model input distribution, since model inputs are the raw
    sub-tomograms themselves (no synthetic masking is applied to them). These values are
    used to normalize sub-tomograms during model fitting and to normalize full tomograms in
    the final refinement step.
    """
    tomo = load_mrc_data(tomo_file).float()
    if standardize:
        tomo = (tomo - tomo.mean()) / tomo.std()
    subtomos, _ = extract_subtomos(
        tomo=tomo,
        subtomo_size=subtomo_size,
        subtomo_extraction_strides=subtomo_extraction_strides,
        pad_before_subtomo_extraction=True,
    )
    dataset = TensorDataset(torch.stack(subtomos))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)
    mean, std = get_avg_model_input_mean_and_std_from_dataloader(dataloader, batches=batches, verbose=verbose)
    return mean, std


def get_avg_model_input_mean_and_std_from_dataloader(dataloader, batches=None, verbose=False):
    """
    See above. Accepts either a dataloader yielding raw sub-tomogram tensors directly (as
    produced by get_avg_model_input_mean_and_std above), or one yielding SubtomoDataset-style
    batches with "subtomo0"/"subtomo1" keys (both are model inputs) - as used during model
    fitting.

    Under multi-GPU (DDP) fitting, 'dataloader' is only this rank's shard of the data (each
    rank gets an equal-sized, disjoint shard - see PyTorch Lightning's automatic
    DistributedSampler replacement), so the mean/std computed from it alone would be a
    per-rank statistic, not a global one - different ranks would end up applying different
    normalization in their forward pass, silently breaking the data-parallel training
    invariant that every replica computes the same function. If a process group is active,
    this averages the local statistics across all ranks so every rank ends up with the
    identical values.
    """
    if batches is None:
        batches = 1 * len(dataloader)
    means, vars = [], []
    bar = (
        tqdm.tqdm(range(batches), desc="Computing model-input normalization statistics")
        if verbose
        else range(batches)
    )
    iter_loader = iter(dataloader)
    for _ in bar:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            batch = next(iter_loader)
        if isinstance(batch, dict):
            inputs = torch.cat([batch["subtomo0"], batch["subtomo1"]], dim=0)
        else:
            inputs = batch[0]
        means.append(inputs.mean(dim=(-1, -2, -3)))
        vars.append(inputs.var(dim=(-1, -2, -3)))
    stats = torch.stack([torch.concat(means, 0).mean(), torch.concat(vars, 0).mean()])
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # the nccl backend (the default for multi-GPU fitting) only supports collectives on
        # CUDA tensors
        if torch.cuda.is_available():
            stats = stats.cuda()
        # each rank's shard is the same size (DistributedSampler pads to make it so), so a
        # plain average of the per-rank statistics equals the global statistic
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        stats /= torch.distributed.get_world_size()
    mean, var = stats[0].cpu().item(), stats[1].cpu().item()
    return mean, var**0.5


def get_avg_dc_loss_eps_from_dataloader(dataloader, batches=None, verbose=False):
    """
    Estimates the data's own noise scale, to calibrate data_consistency_loss's Charbonnier
    'eps' threshold (see ddw.utils.losses.data_consistency_loss) - called once from
    LitUnet3D.update_dc_loss_eps.

    'subtomo0' and 'subtomo1' observe the same physical region with the same 'ctf' but
    independent noise, so 'subtomo0 - subtomo1' cancels both the signal and 'ctf' and leaves
    pure noise (n0 - n1); its rfftn magnitude is therefore a direct, assumption-free estimate
    of the data's actual per-voxel noise scale, without needing to know 'ctf' or the true
    signal at all. Dividing by sqrt(2) accounts for 'n0 - n1' being the difference of two
    independent, identically-distributed noise sources rather than one.

    Same dataloader/DDP-averaging conventions as
    get_avg_model_input_mean_and_std_from_dataloader (see its docstring) - only accepts
    SubtomoDataset-style batches with "subtomo0"/"subtomo1" keys, since the calibration is
    meaningless without the paired independent-noise observations.
    """
    if batches is None:
        batches = 1 * len(dataloader)
    sq_means = []
    bar = (
        tqdm.tqdm(range(batches), desc="Computing data_consistency_loss eps")
        if verbose
        else range(batches)
    )
    iter_loader = iter(dataloader)
    for _ in bar:
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(dataloader)
            batch = next(iter_loader)
        noise = batch["subtomo0"] - batch["subtomo1"]
        noise_ft = torch.fft.rfftn(noise, dim=(-3, -2, -1), norm="ortho")
        sq_means.append(noise_ft.abs().pow(2).mean())
    mean_sq = torch.stack(sq_means).mean()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.cuda.is_available():
            mean_sq = mean_sq.cuda()
        torch.distributed.all_reduce(mean_sq, op=torch.distributed.ReduceOp.SUM)
        mean_sq /= torch.distributed.get_world_size()
    return (mean_sq.cpu().item() ** 0.5) / (2**0.5)
