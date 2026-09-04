"""
ECG Classification Models
Three architectures to choose from:
  1. CNN1D         – fast baseline
  2. ResNet1D      – deeper, residual connections (recommended)
  3. CNNLSTM       – CNN feature extractor + BiLSTM temporal modelling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# 1.  Simple 1-D CNN  (good quick baseline)
# ──────────────────────────────────────────────────────────────
class CNN1D(nn.Module):
    """
    4-layer 1-D CNN with batch normalisation and dropout.
    Input : (batch, 1, 200)
    Output: (batch, num_classes)
    """
    def __init__(self, num_classes=5, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            # block 1
            nn.Conv1d(1, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            # block 2
            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            # block 3
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            # block 4
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ──────────────────────────────────────────────────────────────
# 2.  1-D ResNet  (recommended – best accuracy / speed trade-off)
# ──────────────────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    """Pre-activation residual block for 1-D signals."""
    def __init__(self, channels, kernel_size=7, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNet1D(nn.Module):
    """
    1-D ResNet with 4 stages and global average pooling.
    Input : (batch, 1, 200)
    Output: (batch, num_classes)
    """
    def __init__(self, num_classes=5, base_filters=64, n_blocks=3, dropout=0.3):
        super().__init__()

        # stem
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_filters, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_filters), nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # residual stages
        stages = []
        ch = base_filters
        for stage in range(4):
            for _ in range(n_blocks):
                stages.append(ResBlock1D(ch, dropout=dropout))
            if stage < 3:                               # downsample between stages
                next_ch = ch * 2
                stages.append(nn.Sequential(
                    nn.Conv1d(ch, next_ch, 1, bias=False),
                    nn.BatchNorm1d(next_ch), nn.ReLU(),
                    nn.MaxPool1d(2),
                ))
                ch = next_ch

        self.stages = nn.Sequential(*stages)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(ch, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)


# ──────────────────────────────────────────────────────────────
# 3.  CNN + BiLSTM  (best for capturing sequential dependencies)
# ──────────────────────────────────────────────────────────────
class CNNLSTM(nn.Module):
    """
    CNN extracts local morphology features,
    BiLSTM captures temporal context across the beat.
    Input : (batch, 1, 200)
    Output: (batch, num_classes)
    """
    def __init__(self, num_classes=5, cnn_channels=64, lstm_hidden=128,
                 lstm_layers=2, dropout=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, 7, padding=3),
            nn.BatchNorm1d(cnn_channels), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(cnn_channels, cnn_channels * 2, 5, padding=2),
            nn.BatchNorm1d(cnn_channels * 2), nn.ReLU(), nn.MaxPool1d(2),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)                          # (B, C, L)
        x = x.permute(0, 2, 1)                  # (B, L, C) for LSTM
        out, _ = self.lstm(x)
        x = out[:, -1, :]                        # last time step
        return self.classifier(x)


# ──────────────────────────────────────────────────────────────
# Factory helper
# ──────────────────────────────────────────────────────────────
def get_model(name: str, num_classes: int = 5, **kwargs) -> nn.Module:
    """
    Instantiate a model by name.

    Parameters
    ----------
    name        : 'cnn1d' | 'resnet1d' | 'cnnlstm'
    num_classes : int (default 5 for AAMI)
    **kwargs    : forwarded to the model constructor
    """
    name = name.lower()
    if name == 'cnn1d':
        return CNN1D(num_classes=num_classes, **kwargs)
    elif name == 'resnet1d':
        return ResNet1D(num_classes=num_classes, **kwargs)
    elif name == 'cnnlstm':
        return CNNLSTM(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose: cnn1d | resnet1d | cnnlstm")
