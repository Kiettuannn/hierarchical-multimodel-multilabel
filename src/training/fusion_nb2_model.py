"""
=============================================================================
FUSION NB-2: Fusion Model Architecture — Simple Concat + MLP
=============================================================================
Kiến trúc:
  Vision [1536] ─┐
  ASR    [768]  ─┤ Concat → [3584]
  OCR    [768]  ─┤         + Tabular [12] → [3596]
  CLAP   [512]  ─┘
                  ↓
          MLP Block 1: Linear(3596, 1024) + LayerNorm + GELU + Dropout
          MLP Block 2: Linear(1024,  256) + LayerNorm + GELU + Dropout
          Head      : Linear(256, out_size)  ← 1 (Stage1) hoặc 7 (Stage2)

Được import bởi fusion_nb3_stage1_train.py và fusion_nb4_stage2_train.py.
=============================================================================
"""

# ── Cell 1: Imports ───────────────────────────────────────────────────────────
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
VISION_DIM  = 1536
ASR_DIM     = 768
OCR_DIM     = 768
CLAP_DIM    = 512
TABULAR_DIM = 12

CONCAT_DIM  = VISION_DIM + ASR_DIM + OCR_DIM + CLAP_DIM  # 3584
INPUT_DIM   = CONCAT_DIM + TABULAR_DIM                    # 3596


# ── Cell 2: Model ─────────────────────────────────────────────────────────────
class FusionMLP(nn.Module):
    """
    Simple Concat + MLP Fusion Model.

    Args:
        out_size  : 1 cho Stage 1 (binary), 7 cho Stage 2 (multi-label)
        dropout   : dropout rate (default 0.3)
        hidden_1  : dim của MLP block 1 (default 1024)
        hidden_2  : dim của MLP block 2 (default 256)
    """

    def __init__(self, out_size: int = 7, dropout: float = 0.3,
                 hidden_1: int = 1024, hidden_2: int = 256):
        super().__init__()
        self.out_size = out_size

        self.mlp = nn.Sequential(
            # Block 1
            nn.Linear(INPUT_DIM, hidden_1),
            nn.LayerNorm(hidden_1),
            nn.GELU(),
            nn.Dropout(dropout),
            # Block 2
            nn.Linear(hidden_1, hidden_2),
            nn.LayerNorm(hidden_2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Linear(hidden_2, out_size)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, vision, asr, ocr, clap, tabular):
        """
        Args:
            vision  : (B, 1536)
            asr     : (B, 768)
            ocr     : (B, 768)
            clap    : (B, 512)
            tabular : (B, 12)
        Returns:
            logits  : (B, out_size)  — raw logits, CHƯA qua Sigmoid
        """
        x = torch.cat([vision, asr, ocr, clap, tabular], dim=-1)  # (B, 3596)
        x = self.mlp(x)                                            # (B, 256)
        return self.head(x)                                        # (B, out_size)

    def replace_head(self, new_out_size: int):
        """
        Thay thế head layer để chuyển từ Stage 1 (out=1) sang Stage 2 (out=7).
        Các layer MLP trước đó giữ nguyên weights (warm-start).
        """
        in_features  = self.head.in_features
        self.head    = nn.Linear(in_features, new_out_size)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.out_size = new_out_size
        print(f"✅ Head replaced: {in_features} → {new_out_size}")

    def freeze_mlp(self):
        """Freeze MLP layers, chỉ train head (dùng đầu Stage 2)."""
        for p in self.mlp.parameters():
            p.requires_grad_(False)
        print("🔒 MLP frozen — chỉ train head")

    def unfreeze_all(self):
        """Unfreeze toàn bộ model (dùng sau vài epoch warm-up của Stage 2)."""
        for p in self.parameters():
            p.requires_grad_(True)
        print("🔓 All layers unfrozen")


# ── Cell 3: Smoke test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    B = 4
    model = FusionMLP(out_size=7)
    print(model)

    dummy = {
        "vision" : torch.randn(B, VISION_DIM),
        "asr"    : torch.randn(B, ASR_DIM),
        "ocr"    : torch.randn(B, OCR_DIM),
        "clap"   : torch.randn(B, CLAP_DIM),
        "tabular": torch.randn(B, TABULAR_DIM),
    }
    out = model(**dummy)
    print(f"✅ Output shape: {out.shape}")  # Expected: (4, 7)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
