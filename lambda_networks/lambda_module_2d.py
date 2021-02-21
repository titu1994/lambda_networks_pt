# Ported from https://github.com/lucidrains/lambda-networks
import torch
import torch.nn as nn
import einops

from typing import Optional


def compute_relative_positions(n):
    pos = torch.meshgrid(torch.arange(n), torch.arange(n))
    pos = einops.rearrange(torch.stack(pos), "n i j -> (i j) n")  # [n*n, 2] pos[n] = (i, j)
    rel_pos = pos[None, :] - pos[:, None]  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
    rel_pos += n - 1  # shift value range from [-n+1, n-1] to [0, 2n-2]
    return rel_pos


class LambdaLayer2D(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int],
        dim_k: int = 16,
        m: Optional[int] = None,
        r: Optional[int] = None,
        heads: int = 4,
        dim_interdimension: int = 1,
        implementation: int = 0,
    ):
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim
        self.k = dim_k
        self.u = dim_interdimension  # intra-depth dimension
        self.heads = heads

        VALID_IMPLEMENTATIONS = [0, 1]
        assert implementation in VALID_IMPLEMENTATIONS, f"Implementation must be one of {VALID_IMPLEMENTATIONS}"
        self.implementation = implementation

        assert (dim_out % heads) == 0, "values dimension must be divisible by number of heads for multi-head query"
        dim_v = dim_out // heads
        self.v = dim_v

        self.to_q = nn.Conv2d(dim, dim_k * heads, 1, bias=False)
        self.to_k = nn.Conv2d(dim, dim_k * dim_interdimension, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim_v * dim_interdimension, 1, bias=False)

        # initialize Q, K and V
        nn.init.normal_(self.to_q.weight, std=(dim_k * dim_out) ** (-0.5))
        nn.init.normal_(self.to_k.weight, std=(dim_out) ** (-0.5))
        nn.init.normal_(self.to_v.weight, std=(dim_out) ** (-0.5))

        self.norm_q = nn.BatchNorm2d(dim_k * heads)
        self.norm_v = nn.BatchNorm2d(dim_v * dim_interdimension)

        self.local_context = r is not None

        if m is not None and r is not None:
            raise ValueError("Either one of  `m` or `r` should be provided for global or local context respectively.")

        if m is None and r is None:
            raise ValueError("Either one of `m` or `r` should be provided for global or local context respectively.")

        if r is not None:
            assert (r % 2) == 1, "Receptive kernel size should be odd"
            if self.implementation == 0:
                self.pos_conv = nn.Conv3d(dim_interdimension, dim_k, (1, r, r), padding=(0, r // 2, r // 2))
            elif self.implementation == 1:
                self.pos_conv = nn.Conv2d(dim_interdimension, dim_k, (r, r), padding=(r // 2, r // 2))
        else:
            assert m is not None, "You must specify the window size (m = h = w)"
            rel_lengths = 2 * m - 1
            self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, dim_k, dim_interdimension))
            self.rel_pos = compute_relative_positions(m)

            nn.init.uniform_(self.rel_pos_emb)

    def forward(self, x):
        b, c, hh, ww = x.shape
        u = self.u
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = einops.rearrange(q, "b (h k) hh ww -> b h k (hh ww)", h=h)
        k = einops.rearrange(k, "b (u k) hh ww -> b u k (hh ww)", u=u)
        v = einops.rearrange(v, "b (u v) hh ww -> b u v (hh ww)", u=u)

        k = k.softmax(dim=-1)  # [b, u, k, hh * ww]

        lambda_c = torch.einsum("b u k m, b u v m -> b k v", k, v)
        y_c = torch.einsum("b h k n, b k v -> b h v n", q, lambda_c)

        if self.local_context:
            if self.implementation == 0:
                v = einops.rearrange(v, "b u v (hh ww) -> b u v hh ww", hh=hh, ww=ww)
                lambda_p = self.pos_conv(v)
                y_p = torch.einsum("b h k n, b k v n -> b h v n", q, lambda_p.flatten(3))

            elif self.implementation == 1:
                v = einops.rearrange(v, "b u v (hh ww) -> b u v hh ww", hh=hh, ww=ww)
                v_stack = []
                for v_idx in range(self.v):
                    v_stack.append(self.pos_conv(v[:, :, v_idx, :, :]))
                lambda_p = torch.stack(v_stack, dim=2)
                y_p = torch.einsum("b h k n, b k v n -> b h v n", q, lambda_p.flatten(3))

        else:
            n, m = self.rel_pos.unbind(dim=-1)
            rel_pos_emb = self.rel_pos_emb[n, m]
            lambda_p = torch.einsum("n m k u, b u v m -> b n k v", rel_pos_emb, v)
            y_p = torch.einsum("b h k n, b n k v -> b h v n", q, lambda_p)

        Y = y_c + y_p
        out = einops.rearrange(Y, "b h v (hh ww) -> b (h v) hh ww", hh=hh, ww=ww)
        return out
