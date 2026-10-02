# Copied from the earlier harness: <earlier-harness>/model/small.py (working tree;
# that repo has no git commit). sha256 of the source file:
# d74806ade90e79530e5dc29cd8c8e821742636b8cf24e3570d8db87d77d838cf; EDGE pin
# 17c3428669ed6733edd9d8c66f7dc62060b8e46d.
# Modifications (SPEC 2.1): ChunkDanceDecoder subclass with a 300-frame window, an is_prefix input
# channel (152 input dims, 151 output dims) and an explicit per-row conditioning keep mask so that
# the conditional and unconditional halves of classifier-free guidance run in one batch; default
# size fixed at 192/384/4/4; P_UNCOND added.
"""Chunk-conditioned DanceDecoder plus EDGE's GaussianDiffusion for the schedule buffers and
q_sample. Diffusion settings are EDGE's own (EDGE.py): cosine schedule, 1000 steps, x0
prediction, l2, no p2 weighting, cond_drop_prob 0.25, guidance weight 2."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from lattice_smc.edge_import import EDGE_DIR  # noqa: F401
from model.diffusion import GaussianDiffusion  # noqa: E402
from model.model import DanceDecoder  # noqa: E402
from model.utils import prob_mask_like  # noqa: E402
from vis import SMPLSkeleton  # noqa: E402

REPR_DIM = 151
CHUNK_LEN = 150
WINDOW = 300
MUSIC_DIM = 35
COND_DROP_PROB = 0.25
GUIDANCE_WEIGHT = 2
P_UNCOND = 0.2

DEFAULT_MODEL = dict(latent_dim=192, ff_size=384, num_layers=4, num_heads=4, dropout=0.1)


class ChunkDanceDecoder(DanceDecoder):
    """EDGE's DanceDecoder over a 300-frame window. The motion input gets one extra channel,
    is_prefix (1 on the clean prefix frames, 0 on the chunk being denoised); the output is the
    151-dim x0 prediction for all 300 frames (only frames 150..299 are used)."""

    def __init__(self, latent_dim=192, ff_size=384, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__(nfeats=REPR_DIM, seq_len=WINDOW, latent_dim=latent_dim, ff_size=ff_size,
                         num_layers=num_layers, num_heads=num_heads, dropout=dropout,
                         cond_feature_dim=MUSIC_DIM, activation=F.gelu)
        self.input_projection = nn.Linear(REPR_DIM + 1, latent_dim)

    def forward(self, x, is_prefix, cond_embed, times, cond_drop_prob=0.0, keep_mask=None):
        """x [B, 300, 151], is_prefix [B, 300], cond_embed [B, 300, 35], times [B] long.
        keep_mask [B] bool overrides the Bernoulli(1 - cond_drop_prob) music-conditioning mask
        (needed so one batch can hold conditional and unconditional rows). This is
        DanceDecoder.forward with those two changes; the arithmetic is unchanged."""
        batch_size, device = x.shape[0], x.device
        x = torch.cat([x, is_prefix[..., None].to(x.dtype)], dim=-1)
        x = self.input_projection(x)
        x = self.abs_pos_encoding(x)

        if keep_mask is None:
            keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")

        cond_tokens = self.cond_projection(cond_embed)
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)
        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed)

        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

        t_hidden = self.time_mlp(times)
        t = self.to_time_cond(t_hidden)
        t_tokens = self.to_time_tokens(t_hidden)

        null_cond_hidden = self.null_cond_hidden.to(t.dtype)
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        t = t + cond_hidden

        c = torch.cat((cond_tokens, t_tokens), dim=-2)
        cond_tokens = self.norm_cond(c)
        output = self.seqTransDecoder(x, cond_tokens, t)
        return self.final_layer(output)

    def guided_forward(self, *args, **kwargs):
        raise NotImplementedError("guidance lives in lattice_smc.model.sampler so that NFE is counted")


def build_model(model_cfg):
    cfg = dict(DEFAULT_MODEL)
    cfg.update(model_cfg or {})
    return ChunkDanceDecoder(**cfg)


def build_diffusion(model, device):
    smpl = SMPLSkeleton(device)
    diffusion = GaussianDiffusion(model, WINDOW, REPR_DIM, smpl, schedule="cosine",
                                  n_timestep=1000, predict_epsilon=False, loss_type="l2",
                                  use_p2=False, cond_drop_prob=COND_DROP_PROB,
                                  guidance_weight=GUIDANCE_WEIGHT)
    # GaussianDiffusion deep-copies the model as an EMA "master_model"; we never use it.
    diffusion.master_model = None
    return diffusion.to(device)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["config"].get("model")).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    diffusion = build_diffusion(model, device).eval()
    return model, diffusion, ck


if __name__ == "__main__":
    m = build_model({})
    print(DEFAULT_MODEL, count_params(m))
