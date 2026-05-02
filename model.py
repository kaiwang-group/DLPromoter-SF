import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        if L > self.pe.size(1):
            max_len = max(L, int(self.pe.size(1) * 2))
            pe = torch.zeros(max_len, self.pe.size(-1), device=x.device, dtype=torch.float32)
            position = torch.arange(0, max_len, dtype=torch.float32, device=x.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, self.pe.size(-1), 2, dtype=torch.float32, device=x.device)
                * (-math.log(10000.0) / self.pe.size(-1))
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.pe = pe.unsqueeze(0)
        return x + self.pe[:, :L].to(dtype=x.dtype, device=x.device)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y


class Identity1D(nn.Module):
    def forward(self, x):
        return x


def _same_padding_1d(kernel_size: int) -> int:
    return int(kernel_size // 2)


def _make_cnn_block(in_ch, out_ch, kernel_size, padding, use_se, se_reduction):
    layers = [
        nn.BatchNorm1d(in_ch),
        nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.MaxPool1d(2),
    ]
    if use_se:
        layers.append(SEBlock(out_ch, se_reduction))
    return nn.Sequential(*layers)


class TransformerHybridSEFusionModel(nn.Module):


    # 可选消融开关：
    # - use_se=False
    # - use_extra=False
    # - use_transformer=False

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 256,
        output_size: int = 1,
        dropout_rate: float = 0.2,
        se_reduction: int = 16,
        extra_feat_dim: int = 76,
        use_extra: bool = True,
        conv_kernels=(13, 11, 9),
        use_se: bool = True,
        use_transformer: bool = True,
        transformer_layers: int = 2,   # 这里默认改成2层
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.extra_feat_dim = int(extra_feat_dim)
        self.use_extra = bool(use_extra)
        self.use_se = bool(use_se)
        self.use_transformer = bool(use_transformer)

        k1, k2, k3 = [int(k) for k in conv_kernels]
        p1, p2, p3 = _same_padding_1d(k1), _same_padding_1d(k2), _same_padding_1d(k3)

        # ---------- CNN ----------
        self.cnn1 = _make_cnn_block(input_size, hidden_size, k1, p1, self.use_se, se_reduction)
        self.cnn2 = _make_cnn_block(hidden_size, hidden_size, k2, p2, self.use_se, se_reduction)
        self.cnn3 = _make_cnn_block(hidden_size, hidden_size, k3, p3, self.use_se, se_reduction)

        # ---------- Transformer ----------
        self.input_proj = nn.Conv1d(hidden_size, hidden_size, 1)
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=2048)

        if self.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=512,
                dropout=dropout_rate,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(transformer_layers)
            )
        else:
            self.transformer = nn.Identity()

        self.transformer_se = SEBlock(hidden_size, se_reduction) if self.use_se else Identity1D()

        # ---------- Pool ----------
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ---------- Extra branch ----------
        self.extra_mlp = nn.Sequential(
            nn.LayerNorm(extra_feat_dim),
            nn.Linear(extra_feat_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.extra_gate = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.extra_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

        self.film = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.Tanh()
        )
        self.film_scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

        # ---------- Head ----------
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
        )

        self.conv_kernels = (k1, k2, k3)
        self.transformer_layers = int(transformer_layers)

    def forward(self, x: torch.Tensor, extra_feat: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 4:
            x_c = x
        elif x.size(-1) == 4:
            x_c = x.permute(0, 2, 1)
        else:
            raise ValueError(f"x should be (B,4,L) or (B,L,4), got {tuple(x.shape)}")

        x_c = self.cnn1(x_c)
        x_c = self.cnn2(x_c)
        x_c = self.cnn3(x_c)

        if self.use_transformer:
            x_t = self.input_proj(x_c).permute(0, 2, 1)   # (B, L, C)
            x_t = self.pos_encoder(x_t)
            x_t = self.transformer(x_t)
            x_se = x_t.permute(0, 2, 1)                   # (B, C, L)
            x_se = self.transformer_se(x_se)
        else:
            x_se = x_c
            x_se = self.transformer_se(x_se)

        seq_vec = self.global_pool(x_se).squeeze(-1)

        if self.use_extra:
            extra_hidden = self.extra_mlp(extra_feat)
            extra_vec = extra_hidden * self.extra_gate(extra_hidden) * self.extra_scale

            gb = self.film(extra_hidden)
            gamma, beta = gb.chunk(2, dim=-1)
            seq_mod = seq_vec * (1.0 + self.film_scale * gamma) + (self.film_scale * beta)
        else:
            extra_vec = torch.zeros_like(seq_vec)
            seq_mod = seq_vec

        z = torch.cat([seq_mod, extra_vec], dim=-1)
        y = self.head(z).squeeze(-1)
        return y

    def backbone_parameters(self):
        for m in [
            self.cnn1, self.cnn2, self.cnn3,
            self.input_proj, self.transformer,
            self.transformer_se, self.pos_encoder, self.global_pool
        ]:
            for p in m.parameters():
                yield p

    def head_parameters(self):
        for m in [self.extra_mlp, self.extra_gate, self.film, self.head]:
            for p in m.parameters():
                yield p
        yield self.extra_scale
        yield self.film_scale
