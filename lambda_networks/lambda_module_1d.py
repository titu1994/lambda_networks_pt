# Ported from https://github.com/lucidrains/lambda-networks
import torch
import torch.nn as nn

from typing import Optional


def compute_relative_positions(n):
    pos = torch.meshgrid(torch.arange(n))
    pos = torch.stack(pos).transpose(1, 0)  # [n*n, 2] pos[n] = (i, j)
    rel_pos = pos[None, :] - pos[:, None]  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
    rel_pos += n - 1  # shift value range from [-n+1, n-1] to [0, 2n-2]
    return rel_pos


class LambdaLayer1D(nn.Module):
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
        self.h = self.heads = heads

        VALID_IMPLEMENTATIONS = [0, 1]
        assert implementation in VALID_IMPLEMENTATIONS, f"Implementation must be one of {VALID_IMPLEMENTATIONS}"
        self.implementation = implementation

        assert (dim_out % heads) == 0, "values dimension must be divisible by number of heads for multi-head query"
        dim_v = dim_out // heads
        self.v = dim_v

        self.to_q = nn.Conv1d(dim, dim_k * heads, 1, bias=False)
        self.to_k = nn.Conv1d(dim, dim_k * dim_interdimension, 1, bias=False)
        self.to_v = nn.Conv1d(dim, dim_v * dim_interdimension, 1, bias=False)

        # initialize Q, K and V
        nn.init.normal_(self.to_q.weight, std=(dim_k * dim_out) ** (-0.5))
        nn.init.normal_(self.to_k.weight, std=(dim_out) ** (-0.5))
        nn.init.normal_(self.to_v.weight, std=(dim_out) ** (-0.5))

        self.norm_q = nn.BatchNorm1d(dim_k * heads)
        self.norm_v = nn.BatchNorm1d(dim_v * dim_interdimension)

        self.local_context = r is not None

        if m is not None and r is not None:
            raise ValueError("Either `m` or `r` should be provided for global or local context respectively.")

        if r is not None:
            assert (r % 2) == 1, "Receptive kernel size should be odd"
            if self.implementation == 0:
                self.pos_conv = nn.Conv2d(dim_interdimension, dim_k, (1, r), padding=(0, r // 2))
            elif self.implementation == 1:
                self.pos_conv = nn.Conv1d(dim_interdimension, dim_k, r, padding=(r // 2))
        else:
            assert m is not None, "You must specify the window size (m = t)"
            rel_lengths = 2 * m - 1
            self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, dim_k, dim_interdimension))
            self.rel_pos = compute_relative_positions(m)

            nn.init.uniform_(self.rel_pos_emb)

    def forward(self, x):
        b, c, t = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = q.view([b, self.h, self.k, t])
        k = k.view([b, self.u, self.k, t])
        v = v.view([b, self.u, self.v, t])

        k = k.softmax(dim=-1)  # [b, u, k, t]

        lambda_c = torch.einsum("b u k m, b u v m -> b k v", k, v)
        y_c = torch.einsum("b h k n, b k v -> b h v n", q, lambda_c)

        if self.local_context:
            if self.implementation == 0:
                # v = [b, u, v, t]
                lambda_p = self.pos_conv(v)
                y_p = torch.einsum("b h k n, b k v n -> b h v n", q, lambda_p)

            elif self.implementation == 1:
                # v = [b, u, v, t]
                v_stack = []
                for v_idx in range(self.v):
                    v_stack.append(self.pos_conv(v[:, :, v_idx, :]))
                lambda_p = torch.stack(v_stack, dim=2)
                y_p = torch.einsum("b h k n, b k v n -> b h v n", q, lambda_p)

        else:
            # n = m: [t, t]
            n = m = self.rel_pos[:, :, -1]
            rel_pos_emb = self.rel_pos_emb[n, m]
            lambda_p = torch.einsum("n m k u, b u v m -> b n k v", rel_pos_emb, v)
            y_p = torch.einsum("b h k n, b n k v -> b h v n", q, lambda_p)

        Y = y_c + y_p
        out = Y.reshape([b, self.h * self.v, t])
        return out
