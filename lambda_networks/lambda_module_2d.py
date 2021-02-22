# Ported from https://github.com/lucidrains/lambda-networks
import torch
import torch.nn as nn
import einops

from typing import Optional


class LambdaLayer2D(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int],
        m: Optional[int] = None,
        r: Optional[int] = None,
        dim_k: int = 16,
        dim_intra: int = 1,
        heads: int = 4,
        implementation: int = 0,
    ):
        """
        Lambda Networks module implemented for 4D input tensor (B, C, H, W).

        References:
            - [LambdaNetworks: Modeling Long-Range Interactions Without Attention](https://arxiv.org/abs/2102.08602)

        Args:
            dim: Dimension of the channel axis in the input tensor.
            dim_out: Output dimension of the channel axis.
            m: (Optional) Global Context size. If provided, the spatial dimensions (H, W) must match `m` exactly.
            r: (Optional) Local Context convolutional receptive field. Should be used to reduce memory / compute requirements,
                as well as apply Lambda Module on dimensions which change per batch (i.e. when (H, W) is not constant).
            dim_k: Key / Query dimension. Defaults to 16.
            dim_intra: Intra-depth dimension. Corresponds to `u` in the paper. `u` > 1 computes multi-query
                lambdas over both the context positions and the intra-depth dimension.
            heads: Number of heads in multi-query lambda layer. Corresponds to `h` in the paper.
            implementation: (Optional) Integer flag representing which implementation should be utilized.
                Implementation 0: Implementation from the paper, constructing a n-D Lambda Module utilizing a (n+1)-D
                    Convolutional operator.
                Implementation 1: Equivalent implementation of the paper, constructing a n-D Lambda Module utilizing a
                    n-D Convolutional operator, and then looping through the Key (K) dimension, applying the n-D conv to
                    to each K_i, finally concatenating all the values to map `u` -> `k`. Equivalent to Impl 0 for fp64,
                    minor loss of floating point precision at fp32. May cause issues at fp16 (untested).
        """
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim
        self.dim_in = dim
        self.dim_out = dim_out

        self.k = dim_k
        self.u = dim_intra  # intra-depth dimension
        self.h = self.heads = heads
        self.m = m
        self.r = r

        VALID_IMPLEMENTATIONS = [0, 1]
        assert implementation in VALID_IMPLEMENTATIONS, f"Implementation must be one of {VALID_IMPLEMENTATIONS}"
        self.implementation = implementation

        assert (dim_out % heads) == 0, "values dimension must be divisible by number of heads for multi-head query"
        dim_v = dim_out // heads
        self.v = dim_v

        self.to_q = nn.Conv2d(dim, dim_k * heads, 1, bias=False)
        self.to_k = nn.Conv2d(dim, dim_k * dim_intra, 1, bias=False)
        self.to_v = nn.Conv2d(dim, dim_v * dim_intra, 1, bias=False)

        # initialize Q, K and V
        nn.init.normal_(self.to_q.weight, std=(dim_k * dim_out) ** (-0.5))
        nn.init.normal_(self.to_k.weight, std=(dim_out) ** (-0.5))
        nn.init.normal_(self.to_v.weight, std=(dim_out) ** (-0.5))

        self.norm_q = nn.BatchNorm2d(dim_k * heads)
        self.norm_v = nn.BatchNorm2d(dim_v * dim_intra)

        self.local_context = r is not None

        if m is not None and r is not None:
            raise ValueError("Either one of  `m` or `r` should be provided for global or local context respectively.")

        if m is None and r is None:
            raise ValueError("Either one of `m` or `r` should be provided for global or local context respectively.")

        if r is not None:
            assert (r % 2) == 1, "Receptive kernel size should be odd"
            if self.implementation == 0:
                self.pos_conv = nn.Conv3d(dim_intra, dim_k, (1, r, r), padding=(0, r // 2, r // 2))
            elif self.implementation == 1:
                self.pos_conv = nn.Conv2d(dim_intra, dim_k, (r, r), padding=(r // 2, r // 2))
        else:
            assert m is not None, "You must specify the window size (m = h = w)"
            rel_lengths = 2 * m - 1
            self.rel_pos_emb = nn.Parameter(torch.randn(rel_lengths, rel_lengths, dim_k, dim_intra))
            self.rel_pos = self.compute_relative_positions(m, m)

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
                del v_stack
                y_p = torch.einsum("b h k n, b k v n -> b h v n", q, lambda_p.flatten(3))

        else:
            if hh == self.m and ww == self.m:
                h_, w_ = self.rel_pos.unbind(dim=-1)
            else:
                if hh > self.m or ww > self.m:
                    raise ValueError(
                        f"Current spatial dimension ({hh}, {ww}) cannot be larger than maximum context size "
                        f"({self.m}, {self.m})"
                    )

                pos_ = self.compute_relative_positions(hh, ww, device=x.device)
                h_, w_ = pos_.unbind(dim=-1)

            rel_pos_emb = self.rel_pos_emb[h_, w_]
            lambda_p = torch.einsum("n m k u, b u v m -> b n k v", rel_pos_emb, v)
            y_p = torch.einsum("b h k n, b n k v -> b h v n", q, lambda_p)

        Y = y_c + y_p
        out = einops.rearrange(Y, "b h v (hh ww) -> b (h v) hh ww", hh=hh, ww=ww)
        return out

    def compute_relative_positions(self, h, w, device=None):
        pos = torch.meshgrid(torch.arange(h), torch.arange(w))
        pos = einops.rearrange(torch.stack(pos), "n i j -> (i j) n")  # [n*n, 2] pos[n] = (i, j)

        if device is not None:
            pos = pos.to(device)

        rel_pos = pos[None, :] - pos[:, None]  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
        rel_pos = torch.clamp(rel_pos, -self.m, self.m)
        rel_pos += self.m - 1  # n - 1  # shift value range from [-n+1, n-1] to [0, 2n-2]
        return rel_pos

    def extra_repr(self):
        return 'input_dim={dim_in}, output_dim={dim_out}, m={m}, r={r}, k={k}, h={h}, u={u},'.format(**self.__dict__)
