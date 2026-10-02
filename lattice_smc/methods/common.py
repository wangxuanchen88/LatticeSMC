"""Shared pieces of the inference-time methods (SPEC 4): the per-(seed, method) resampling
generator, the reward evaluator for a prompt, and one-chunk generation for N particle slots."""
import torch

from lattice_smc.model.sampler import chunk_noise, sample_chunk, window_music
from lattice_smc.rewards.rep import RepEvaluator
from lattice_smc.rewards.steering import RewardEvaluator

METHOD_INDEX = {"base": 0, "bon_argmax": 1, "bon_is": 2, "fk_noise": 3, "greedy_chunk": 4,
                "lattice_smc": 5, "lattice_smc_notwist": 5, "lattice_smc_adaptive": 7}  # the ablation shares lattice_smc's stream


def resample_generator(seed, method):
    """One CPU generator per (sequence seed, method); every resampling event and the final draw
    consume it in a fixed order, so the sequence is reproducible."""
    return torch.Generator(device="cpu").manual_seed(int(seed) * 131 + METHOD_INDEX[method])


def evaluator(ctx, music):
    """The steering reward of the run: R_BA (default) or R_rep (Amendment M)."""
    if getattr(ctx, "reward", "ba") == "rep":
        return RepEvaluator(music, ctx.diffusion.smpl, ctx.scale, ctx.mn, ctx.counters)
    return RewardEvaluator(music, ctx.diffusion.smpl, ctx.scale, ctx.mn, ctx.counters)


def gen_chunk(ctx, prev, music, k, seed, N, on_x0_pred=None):
    """Chunk k for N particle slots (slot p uses the noise stream (seed, k, p)) given the
    particles' own previous chunks prev [N, 150, 151] (None for chunk 1)."""
    noise = chunk_noise(seed, k, range(N), device=ctx.device)
    return sample_chunk(ctx.model, ctx.diffusion, prev, window_music(music, k, N), noise, ctx.counters,
                        on_x0_pred=on_x0_pred, cfg=ctx.loss_cfg)


def alpha_of(ctx):
    assert ctx.alpha is not None, "alpha is not set (run the pilot first, data/alpha.json)"
    return float(ctx.alpha)
