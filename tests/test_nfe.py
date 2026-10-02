"""NFE accounting (project rule 3): base at K chunks costs exactly K x 50 x 2 per particle, and
the sampler is deterministic on one device (same seed -> identical chunk). Random-init model."""
import torch

from lattice_smc.generate import Context, generate_sequence, set_deterministic
from lattice_smc.model.sampler import DDIM_STEPS, chunk_noise, ddim_time_pairs, denoise_chunk, sample_chunk, window_music, window_prefix
from lattice_smc.model.small import build_diffusion, build_model

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_ctx():
    set_deterministic()
    torch.manual_seed(5)
    model = build_model({}).to(DEV).eval()
    return Context(model, build_diffusion(model, DEV).eval(), DEV)


def test_time_grid_matches_edge():
    pairs = ddim_time_pairs()
    assert len(pairs) == DDIM_STEPS and pairs[0] == (999, 979) and pairs[-1] == (19, -1)


def test_base_nfe_and_determinism():
    ctx = make_ctx()
    K = 2
    music = torch.randn(K * 150, 35, generator=torch.Generator().manual_seed(0))
    m1, log1 = generate_sequence(ctx, music, K, 2000, "base", 1)
    m2, log2 = generate_sequence(ctx, music, K, 2000, "base", 1)
    m3, _ = generate_sequence(ctx, music, K, 2001, "base", 1)
    assert log1["nfe"] == K * DDIM_STEPS * 2 == 200 and log1["reward_evals"] == 0
    assert log1["chunk_nfe"] == [100, 100]
    assert torch.equal(m1, m2)
    assert not torch.equal(m1, m3)
    assert m1.shape == (K * 150, 151)


def test_particle_batch_counts_rows():
    ctx = make_ctx()
    N = 3
    music = torch.randn(150, 35, generator=torch.Generator().manual_seed(0)).to(DEV)
    prefix, is_prefix = window_prefix(None, N, DEV)
    noise = chunk_noise(2000, 1, range(N), device=DEV)
    ctx.counters.reset()
    x = denoise_chunk(ctx.model, ctx.diffusion, prefix, is_prefix, window_music(music, 1, N), noise, ctx.counters)
    assert ctx.counters.nfe.n == N * DDIM_STEPS * 2
    assert x.shape == (N, 150, 151)
    # particle 0 of the batch uses the same noise stream as the single-particle run
    ctx.counters.reset()
    x0 = denoise_chunk(ctx.model, ctx.diffusion, prefix[:1], is_prefix[:1], window_music(music, 1, 1),
                       chunk_noise(2000, 1, range(1), device=DEV), ctx.counters)
    assert torch.equal(noise[0], chunk_noise(2000, 1, range(1), device=DEV)[0])
    assert torch.allclose(x[:1], x0, atol=1e-4)


def test_sample_chunk_shift_is_applied_only_with_a_prefix():
    """Amendment C in the sampler: with a prefix, the chunk is denoised in the root-relative frame
    and un-shifted; shifting the prefix by a constant root offset shifts the output by exactly
    that offset (the model never sees the absolute root)."""
    ctx = make_ctx()
    g = torch.Generator().manual_seed(0)
    music = torch.randn(300, 35, generator=g).to(DEV)
    prev = (torch.rand(1, 150, 151, generator=g) * 2 - 1).to(DEV)
    noise = chunk_noise(2000, 2, range(1), device=DEV)
    x_a = sample_chunk(ctx.model, ctx.diffusion, prev, window_music(music, 2, 1), noise, ctx.counters)
    prev_b = prev.clone()
    prev_b[..., 4:7] += 0.25
    x_b = sample_chunk(ctx.model, ctx.diffusion, prev_b, window_music(music, 2, 1), noise, ctx.counters)
    assert torch.equal(x_a[..., :4], x_b[..., :4]) and torch.equal(x_a[..., 7:], x_b[..., 7:])
    assert torch.allclose(x_b[..., 4:7] - x_a[..., 4:7], torch.full_like(x_a[..., 4:7], 0.25), atol=1e-6)
    assert ctx.counters.nfe.n == 2 * DDIM_STEPS * 2


def test_sample_chunk_disp_representation():
    """Amendment E in the sampler: the returned chunk is in position form, NFE is unchanged, and
    a prefix root offset shifts the output by exactly that offset."""
    ctx = make_ctx()
    cfg = {"repr": "disp", "root_weight": 1.0, "seam_lambda": 0.0}
    g = torch.Generator().manual_seed(0)
    music = torch.randn(300, 35, generator=g).to(DEV)
    prev = (torch.rand(1, 150, 151, generator=g) * 2 - 1).to(DEV)
    noise = chunk_noise(2000, 2, range(1), device=DEV)
    ctx.counters.reset()
    x_a = sample_chunk(ctx.model, ctx.diffusion, prev, window_music(music, 2, 1), noise, ctx.counters, cfg=cfg)
    assert ctx.counters.nfe.n == DDIM_STEPS * 2 and x_a.shape == (1, 150, 151)
    prev_b = prev.clone()
    prev_b[..., 4:7] += 0.25
    x_b = sample_chunk(ctx.model, ctx.diffusion, prev_b, window_music(music, 2, 1), noise, ctx.counters, cfg=cfg)
    assert torch.allclose(x_b[..., 4:7] - x_a[..., 4:7], torch.full_like(x_a[..., 4:7], 0.25), atol=1e-6)
    # chunk 1 integrates from the origin
    x_1 = sample_chunk(ctx.model, ctx.diffusion, None, window_music(music, 1, 1), chunk_noise(2000, 1, range(1), device=DEV), ctx.counters, cfg=cfg)
    assert x_1.shape == (1, 150, 151) and torch.isfinite(x_1).all()
