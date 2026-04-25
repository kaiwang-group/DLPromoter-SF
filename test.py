import os
import re
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    median_absolute_error, max_error, explained_variance_score
)
from scipy.stats import pearsonr, spearmanr

from model import TransformerHybridSEFusionModel


# -----------------------
# Utils (保持与训练代码一致)
# -----------------------
def clean_seq(s: str) -> str:
    s = s.strip().lower()
    s = re.sub("[^acgt]", "z", s)
    return s


def one_hot_encode(seq: str):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    arr = np.array(list(seq), dtype="<U1")
    idx = np.array([mapping.get(c, 4) for c in arr], dtype=np.int64)
    oh = np.zeros((len(idx), 4), dtype=np.float32)
    mask = idx < 4
    oh[np.arange(len(idx))[mask], idx[mask]] = 1.0
    return oh


def pad_or_trunc(mat: np.ndarray, max_len: int):
    L = mat.shape[0]
    if L > max_len:
        return mat[:max_len]
    if L < max_len:
        pad_len = max_len - L
        return np.pad(mat, ((0, pad_len), (0, 0)))
    return mat


def load_csv(path: str):
    seqs, ys = [], []
    with open(path, "r", encoding="utf-8") as f:
        _ = next(f, None)
        for line in f:
            if not line.strip():
                continue
            s, y = line.strip().split(",")
            seqs.append(s)
            ys.append(float(y))
    return seqs, np.asarray(ys, dtype=np.float32)


def torch_load_state_dict(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


# -----------------------
# Extra features (保持与训练代码一致)
# -----------------------
_BASES = ["a", "c", "g", "t"]
_KMER3 = [a + b + c for a in _BASES for b in _BASES for c in _BASES]
_KMER3_TO_IDX = {k: i for i, k in enumerate(_KMER3)}


def kmer3_freq(seq: str):
    seq = seq.lower()
    counts = np.zeros((64,), dtype=np.float32)
    denom = 0
    for i in range(len(seq) - 2):
        k = seq[i:i + 3]
        if "z" in k:
            continue
        counts[_KMER3_TO_IDX[k]] += 1.0
        denom += 1
    if denom > 0:
        counts /= denom
    return counts


def gc_profile(seq: str, n_bins: int = 8):
    seq = seq.lower()
    L = len(seq)
    out = np.zeros((n_bins,), dtype=np.float32)
    for b in range(n_bins):
        l = int(round(b * L / n_bins))
        r = int(round((b + 1) * L / n_bins))
        chunk = seq[l:r]
        valid = [c for c in chunk if c in "acgt"]
        if len(valid) == 0:
            out[b] = 0.0
            continue
        gc = sum(1 for c in valid if c in "gc") / float(len(valid))
        out[b] = gc
    return out


def global_comp(seq: str):
    seq = seq.lower()
    valid = [c for c in seq if c in "acgt"]
    if len(valid) == 0:
        return np.zeros((2,), dtype=np.float32)
    gc = sum(1 for c in valid if c in "gc") / float(len(valid))
    at = 1.0 - gc
    return np.asarray([gc, at], dtype=np.float32)


def longest_homopolymer(seq: str, base: str):
    seq = seq.lower()
    best = cur = 0
    for c in seq:
        if c == base:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return float(best)


def build_extra_features(seq: str, n_bins: int = 8):
    seq = clean_seq(seq)
    f1 = kmer3_freq(seq)
    f2 = gc_profile(seq, n_bins=n_bins)
    f3 = global_comp(seq)
    f4 = np.asarray([longest_homopolymer(seq, "a"), longest_homopolymer(seq, "t")], dtype=np.float32)
    return np.concatenate([f1, f2, f3, f4], axis=0).astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(np.float64).ravel()
    y_pred = y_pred.astype(np.float64).ravel()
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mxerr = float(max_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    ev = float(explained_variance_score(y_true, y_pred))
    pr = float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else 0.0
    sr = float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else 0.0

    return {
        "R2": r2, "RMSE": rmse, "MAE": mae, "MedAE": medae,
        "MaxError": mxerr, "MSE": float(mse), "ExplainedVar": ev,
        "Pearson": pr, "Spearman": sr,
    }


# -----------------------
# Testing Script
# -----------------------
def run_test(run_dir: str, test_csv: str, gpu_id: int = 1, batch_size: int = 256, num_workers: int = 2):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    # 1. 恢复配置文件
    cfg_path = os.path.join(run_dir, "run_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    max_len = cfg["max_len"]
    gc_bins = cfg.get("gc_bins", 8)
    normalize_extra = cfg.get("normalize_extra", True)

    # 2. 读取测试数据
    seq_test, y_test = load_csv(test_csv)
    test_oh_list = [one_hot_encode(clean_seq(s)) for s in seq_test]

    # 根据训练集确定的 max_len 进行 padding 或 truncate
    X_test = np.stack([pad_or_trunc(m, max_len) for m in test_oh_list], axis=0)
    F_test = np.stack([build_extra_features(s, n_bins=gc_bins) for s in seq_test], axis=0)

    # 3. 恢复归一化参数并应用
    if normalize_extra:
        norm_path = os.path.join(run_dir, "extra_norm.json")
        with open(norm_path, "r", encoding="utf-8") as f:
            norm = json.load(f)
        mu = np.array(norm["mu"]).reshape(1, -1)
        sd = np.array(norm["sd"]).reshape(1, -1)
        F_test = (F_test - mu) / sd

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    F_test_t = torch.tensor(F_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    pin = bool(torch.cuda.is_available())
    test_loader = DataLoader(
        TensorDataset(X_test_t, F_test_t, y_test_t),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin
    )

    # 4. 初始化模型
    model = TransformerHybridSEFusionModel(
        input_size=4,
        hidden_size=cfg["hidden_size"],
        output_size=1,
        dropout_rate=cfg["dropout_rate"],
        extra_feat_dim=cfg["extra_feat_dim"],
        use_extra=cfg["use_extra"],
        conv_kernels=tuple(cfg["conv_kernels"]),
    ).to(device)

    best_path = os.path.join(run_dir, "best_model.pth")
    model.load_state_dict(torch_load_state_dict(best_path, map_location=device))
    model.eval()

    # 5. 开始推断
    preds, trues = [], []
    with torch.no_grad():
        for x, feat, y in test_loader:
            x = x.to(device, non_blocking=True)
            feat = feat.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1)

            out = model(x, feat).view(-1)
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())

    y_true = np.concatenate(trues).ravel()
    y_pred = np.concatenate(preds).ravel()

    # 6. 计算 Metrics 并保存预测
    test_metrics = compute_metrics(y_true, y_pred)

    pred_path = os.path.join(run_dir, "test_predictions.csv")
    with open(pred_path, "w", encoding="utf-8") as f:
        f.write("y_true,y_pred\n")
        for yt, yp in zip(y_true, y_pred):
            f.write(f"{yt},{yp}\n")

    metrics_path = os.path.join(run_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    print("\n===== TEST METRICS =====")
    print(f"Test Run Dir: {run_dir}")
    print("TEST METRICS:", json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"Predictions saved to: {pred_path}")

    return test_metrics


if __name__ == "__main__":
    TEST_CSV = "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Data/SC/wrcprocess_test_data162982.csv"

    # 请在这里填入你刚刚训练代码生成保存的具体的 run_dir 路径

    RUN_DIR = "在此处填入训练日志里生成的实际路径"

    # 运行测试
    if os.path.exists(RUN_DIR):
        run_test(run_dir=RUN_DIR, test_csv=TEST_CSV, gpu_id=1)
    else:
        print(f"Error: 请确保 RUN_DIR 设置正确，路径 {RUN_DIR} 不存在。")