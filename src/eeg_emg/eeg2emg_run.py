#!/usr/bin/env python3
"""
eeg2emg_run.py - adapted for continuous or trial-based EEG/EMG .npz datasets.

Features:
- Accepts .npz with various key names (case-insensitive).
- Accepts trial-based (trials, channels, time) or continuous (samples, channels) arrays.
- Converts continuous -> single-trial (1, channels, time) and builds sliding windows.
- Optional resampling of EEG -> EMG sampling rate.
- CNN+LSTM regression EEG -> EMG (window -> window).
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Utilities: robust NPZ loading
# -----------------------------
def load_npz_anycase(path):
    d = np.load(path, allow_pickle=True)
    # map lowercase->original key
    keys = {k.lower(): k for k in d.files}
    eeg_key = None
    emg_key = None
    for cand in [
        "eeg",
        "eeg_data",
        "eeg_signal",
        "eeg_signals",
        "eegarray",
        "eeg_array",
        "eegraw",
        "signal_eeg",
    ]:
        if cand in keys:
            eeg_key = keys[cand]
            break
    for cand in [
        "emg",
        "emg_data",
        "emg_signal",
        "emg_signals",
        "emgarray",
        "emg_array",
        "emgraw",
        "signal_emg",
    ]:
        if cand in keys:
            emg_key = keys[cand]
            break
    # fallback: if two arrays present, assume first=EEG, second=EMG
    if eeg_key is None or emg_key is None:
        files = d.files
        if len(files) >= 2:
            if eeg_key is None:
                eeg_key = files[0]
            if emg_key is None:
                # choose second key that's not eeg_key
                for f in files:
                    if f != eeg_key:
                        emg_key = f
                        break
        else:
            raise KeyError(f"Could not find EEG/EMG in {path}. Keys found: {d.files}")
    return d[eeg_key], d[emg_key]


# -----------------------------
# Convert arrays to trial-format
# -----------------------------
def to_trials_format(arr):
    """
    Convert array to shape (n_trials, channels, time).
    Accepts:
      - 3D (trials, channels, time) -> returned unchanged
      - 2D (samples, channels) -> (1, channels, samples)
      - 2D (channels, samples) -> (1, channels, samples)
      - 1D -> (1,1,time)
    Heuristics:
      if arr.ndim==2 and arr.shape[0] < arr.shape[1]: assume (channels, samples)
      else assume (samples, channels) and transpose.
    """
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr.astype(np.float32)
    if arr.ndim == 2:
        r0, r1 = arr.shape
        if r0 < r1:
            # likely (channels, samples)
            channels, samples = r0, r1
            return arr.reshape(1, channels, samples).astype(np.float32)
        else:
            # likely (samples, channels)
            samples, channels = r0, r1
            return arr.T.reshape(1, channels, samples).astype(np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, 1, -1).astype(np.float32)
    raise ValueError(f"Unsupported array shape: {arr.shape}")


# -----------------------------
# Simple normalization
# -----------------------------
def zscore(x, axis=None, eps=1e-8):
    mu = np.mean(x, axis=axis, keepdims=True)
    sigma = np.std(x, axis=axis, keepdims=True)
    return (x - mu) / (sigma + eps)


# -----------------------------
# Dataset with sliding windows
# -----------------------------
class EEGEMGWindowDataset(Dataset):
    def __init__(
        self, eeg_data, emg_data, window_size=256, step=128, normalize=True, pre_windowed=False
    ):
        assert eeg_data.shape[0] == emg_data.shape[0], "trials/windows count mismatch"
        self.eeg = eeg_data
        self.emg = emg_data
        self.normalize = normalize
        self.pre_windowed = pre_windowed
        self.index = []

        if self.normalize:
            mean = np.mean(self.eeg, axis=-1, keepdims=True)
            std = np.std(self.eeg, axis=-1, keepdims=True)
            self.eeg = (self.eeg - mean) / (std + 1e-8)

            mean_emg = np.mean(self.emg, axis=-1, keepdims=True)
            std_emg = np.std(self.emg, axis=-1, keepdims=True)
            self.emg = (self.emg - mean_emg) / (std_emg + 1e-8)

        if self.pre_windowed:
            current_len = self.eeg.shape[-1]
            if current_len < window_size:
                print(
                    f"WARNING: augmented windows ({current_len}) are smaller than the required window_size ({window_size})."
                )
                self.window_size = current_len
            else:
                self.window_size = window_size
            for i in range(self.eeg.shape[0]):
                self.index.append((i, 0))

        else:
            self.window_size = window_size
            self.step = step
            for t in range(self.eeg.shape[0]):
                T = self.eeg.shape[-1]
                if self.window_size > T:
                    continue
                for s in range(0, T - self.window_size + 1, self.step):
                    self.index.append((t, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        t, s = self.index[idx]
        if self.pre_windowed:
            e = self.eeg[t]
            m = self.emg[t]
            if e.shape[-1] > self.window_size:
                e = e[:, : self.window_size]
                m = m[:, : self.window_size]
        else:
            e = self.eeg[t, :, s : s + self.window_size].copy()
            m = self.emg[t, :, s : s + self.window_size].copy()
        return e, m


# -----------------------------
# Model (CNN1D + LSTM -> EMG)
# -----------------------------
class CNNLSTM_EEG2EMG(nn.Module):
    def __init__(
        self,
        n_eeg_channels,
        n_emg_channels,
        cnn_channels=64,
        lstm_hidden=128,
        lstm_layers=2,
        bidirectional=False,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=n_eeg_channels, out_channels=cnn_channels, kernel_size=5, padding=2
        )
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out = lstm_hidden * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out, n_emg_channels)

    def forward(self, x):
        # x: (batch, channels, time)
        c = self.conv1(x)
        c = self.bn1(c)
        c = self.relu(c)
        c = self.conv2(c)
        c = self.bn2(c)
        c = self.relu(c)
        c = c.permute(0, 2, 1).contiguous()  # (batch, time, feats)
        lstm_out, _ = self.lstm(c)
        out = self.fc(lstm_out)  # (batch, time, emg_ch)
        return out.permute(0, 2, 1).contiguous()  # (batch, emg_ch, time)


# -----------------------------
# Training and eval helpers
# -----------------------------
def train_epoch(model, loader, optim, criterion_mse, criterion_corr, device):
    model.train()
    total = 0.0
    n = 0
    for X, Y in loader:
        X = X.to(device).float()
        Y = Y.to(device).float()
        optim.zero_grad()
        pred = model(X)
        loss_mse = criterion_mse(pred, Y)
        # loss_mse = 0
        loss_corr = criterion_corr(pred, Y)
        loss = loss_mse + (0.5 * loss_corr)
        loss.backward()
        optim.step()
        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / max(1, n)


def evaluate(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device).float()
            Y = Y.to(device).float()
            p = model(X)
            preds.append(p.cpu().numpy())
            trues.append(Y.cpu().numpy())
    if len(preds) == 0:
        return np.zeros((0,)), np.zeros((0,))
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return preds, trues


def mse_metric(pred, true):
    return float(np.mean((pred - true) ** 2))


def r2_score_np(pred, true):
    pred = np.array(pred)
    true = np.array(true)
    if pred.size == 0:
        return 0.0
    if pred.ndim == 3:
        vals = []
        for i in range(pred.shape[0]):
            p = pred[i].reshape(pred.shape[1], -1)
            t = true[i].reshape(true.shape[1], -1)
            for ch in range(p.shape[0]):
                ss_res = np.sum((t[ch] - p[ch]) ** 2)
                ss_tot = np.sum((t[ch] - np.mean(t[ch])) ** 2) + 1e-8
                vals.append(1 - ss_res / ss_tot)
        return float(np.mean(vals))
    else:
        p = pred.reshape(pred.shape[0], -1)
        t = true.reshape(true.shape[0], -1)
        vals = []
        for ch in range(p.shape[0]):
            ss_res = np.sum((t[ch] - p[ch]) ** 2)
            ss_tot = np.sum((t[ch] - np.mean(t[ch])) ** 2) + 1e-8
            vals.append(1 - ss_res / ss_tot)
        return float(np.mean(vals))


class NegPearsonLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds, targets):
        # preds/targets shape: (Batch, Channel, Time)
        preds_mean = preds - preds.mean(dim=2, keepdim=True)
        targets_mean = targets - targets.mean(dim=2, keepdim=True)
        cov = (preds_mean * targets_mean).sum(dim=2)
        preds_std = preds_mean.pow(2).sum(dim=2).sqrt()
        targets_std = targets_mean.pow(2).sum(dim=2).sqrt()
        pearson = cov / (preds_std * targets_std + self.eps)

        return 1 - pearson.mean()


# -----------------------------
# Main runner
# -----------------------------
def main(args):
    eeg_raw, emg_raw = load_npz_anycase(args.data_path)
    print("Raw shapes:", eeg_raw.shape, emg_raw.shape)
    # convert to trials
    eeg_trials = to_trials_format(eeg_raw)
    emg_trials = to_trials_format(emg_raw)
    print("Trials format:", eeg_trials.shape, emg_trials.shape)

    if eeg_raw.ndim == 3 and eeg_raw.shape[1] > eeg_raw.shape[2]:
        print(">>> Détection format (N, Time, Channel). Transposition en (N, Channel, Time)...")
        eeg_trials = eeg_raw.transpose(0, 2, 1).astype(np.float32)
        emg_trials = emg_raw.transpose(0, 2, 1).astype(np.float32)
        is_pre_windowed = True
    else:
        eeg_trials = to_trials_format(eeg_raw)
        emg_trials = to_trials_format(emg_raw)
        is_pre_windowed = False

    print("Final Trials format (N, Ch, T):", eeg_trials.shape, emg_trials.shape)

    # optional resample: resample EEG to EMG fs
    if args.resample and args.eeg_fs and args.emg_fs and args.eeg_fs != args.emg_fs:
        factor = args.emg_fs / args.eeg_fs
        newlen = int(np.round(eeg_trials.shape[-1] * factor))
        print(
            f"Resampling EEG from {eeg_trials.shape[-1]} -> {newlen} samples (factor {factor:.3f})"
        )
        eeg_trials = np.stack(
            [np.stack([resample(ch, newlen, axis=-1) for ch in trial]) for trial in eeg_trials]
        )
        eeg_trials = eeg_trials.astype(np.float32)

    # align length by trimming to min length across trials
    min_len = min(eeg_trials.shape[-1], emg_trials.shape[-1])
    if eeg_trials.shape[-1] != min_len or emg_trials.shape[-1] != min_len:
        print("Trimming to min length:", min_len)
        eeg_trials = eeg_trials[..., :min_len]
        emg_trials = emg_trials[..., :min_len]

    # build dataset windows
    dataset = EEGEMGWindowDataset(
        eeg_trials,
        emg_trials,
        window_size=args.window_size,
        step=args.step,
        normalize=args.normalize,
        pre_windowed=is_pre_windowed,
    )
    print("Total windows:", len(dataset))
    if len(dataset) == 0:
        raise RuntimeError(
            "No windows created: check window_size/step relative to recording length."
        )

    n_train = int(len(dataset) * (1 - args.val_ratio))
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # infer channel counts if not provided
    n_eeg_ch = eeg_trials.shape[1]
    n_emg_ch = emg_trials.shape[1]
    print("Channels (EEG, EMG):", n_eeg_ch, n_emg_ch)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = CNNLSTM_EEG2EMG(
        n_eeg_ch,
        n_emg_ch,
        cnn_channels=args.cnn_channels,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        bidirectional=args.bidirectional,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_mse = nn.MSELoss()
    criterion_corr = NegPearsonLoss()

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion_mse, criterion_corr, device)
        preds, trues = evaluate(model, val_loader, device)
        val_mse = mse_metric(preds, trues)
        val_r2 = r2_score_np(preds, trues)
        t1 = time.time()
        print(
            f"Epoch {epoch}/{args.epochs} train_loss={tr_loss:.6f} val_mse={val_mse:.6f} val_r2={val_r2:.4f} time={t1 - t0:.1f}s"
        )
        if val_mse < best_val:
            best_val = val_mse
            config = {
                "n_eeg_channels": n_eeg_ch,
                "n_emg_channels": n_emg_ch,
                "cnn_channels": args.cnn_channels,
                "lstm_hidden": args.lstm_hidden,
                "lstm_layers": args.lstm_layers,
                "bidirectional": args.bidirectional,
                "window_size": args.window_size,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "best_val_mse": best_val,
                },
                args.save_path,
            )
            print("Saved best model:", args.save_path)

    preds, trues = evaluate(model, val_loader, device)
    np.savez(args.output_npz, preds=preds, trues=trues)
    print("Saved predictions to", args.output_npz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to .npz (contains EEG+EMG arrays)"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--step", type=int, default=128)
    parser.add_argument("--cnn_channels", type=int, default=64)
    parser.add_argument("--lstm_hidden", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--save_path", type=str, default="eeg2emg_best.pth")
    parser.add_argument("--output_npz", type=str, default="eeg2emg_preds.npz")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--resample", action="store_true", help="Resample EEG to EMG fs")
    parser.add_argument("--eeg_fs", type=float, default=None)
    parser.add_argument("--emg_fs", type=float, default=None)
    args = parser.parse_args()
    main(args)
