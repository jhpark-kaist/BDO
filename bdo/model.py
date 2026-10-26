import torch
import torch.nn as nn
import torch.nn.functional as F


def time_encoding(times: torch.Tensor, dim: int, base: float = 10000.0) -> torch.Tensor:
    """Sinusoidal encoding of continuous time stamps, sin and cos interleaved. [B, T] -> [B, T, dim]."""
    i = torch.arange(dim // 2, device=times.device)
    freqs = base ** (2 * i / dim)
    x = times[..., None].float() / freqs
    return torch.stack([torch.sin(x), torch.cos(x)], dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class BDOEncoder(nn.Module):
    """Pre-trained BDO encoder.

    Maps an fMRI ROI time series and its time stamps to one latent vector per time step.
    The time stamps can be irregular. Each ROI vector is embedded linearly, its time stamp is
    added as a sinusoidal encoding, and the sequence goes through a transformer over time.
    """

    def __init__(self, input_dim: int = 450, embed_dim: int = 768, depth: int = 12,
                 num_heads: int = 12, mlp_ratio: float = 4.0, time_scale: float = 0.1):
        super().__init__()
        self.config = dict(input_dim=input_dim, embed_dim=embed_dim, depth=depth,
                           num_heads=num_heads, mlp_ratio=mlp_ratio, time_scale=time_scale)
        self.embed_dim = embed_dim
        self.time_scale = time_scale
        self.embed = nn.Linear(input_dim, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        x:     [B, T, input_dim] ROI time series, normalized with `bdo.RobustScaler`.
        times: [B, T] acquisition time of each step in seconds, e.g. `arange(T) * TR`.
        Returns the latent sequence, [B, T, embed_dim].
        """
        x = self.embed(x) + time_encoding(self.time_scale * times, self.embed_dim)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    @torch.no_grad()
    def embed_subject(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Subject-level representation used in the paper's downstream evaluation: the time average
        of the latent sequence. [B, T, input_dim], [B, T] -> [B, embed_dim]."""
        return self.forward(x, times).mean(dim=1)

    @classmethod
    def from_pretrained(cls, path: str, map_location="cpu") -> "BDOEncoder":
        """Load a released checkpoint (a dict with `config` and `state_dict`)."""
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model.eval()
