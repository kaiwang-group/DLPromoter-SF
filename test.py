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

# 导入你的模型架构 (确保 model.py 在同级目录或 Python 环境变量中)
from model import TransformerHybridSEFusionModel

# ==========================================
# 1. 路径与硬件配置 (请确认这里的路径完全正确)
# ==========================================
GPU_ID = 1
MODEL_PATH = "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/wrc_63468huigui/CNN_TF222/results/combine162982_ema_k13_11_9_20260314_141632/best_model.pth"
CONFIG_PATH = "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/wrc_63468huigui/CNN_TF222/results/combine162982_ema_k13_11_9_20260314_141632/run_config.json"
NORM_PATH = "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/wrc_63468huigui/CNN_TF222/results/combine162982_ema_k13_11_9_20260314_141632/extra_norm.json"

# 注意替换为真实的测试集 CSV 路径
TEST_CSV = "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/wrc_63468huigui/DLPromoter-SF/dataset/wrcprocess_test_data162982.csv"
# TEST_CSV = "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/wrc_63468huigui/DLPromoter-SF/dataset/medium_similarity.csv"
# TEST_CSV = "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/wrc_63468huigui/DLPromoter-SF/dataset/high_similarity.csv"
# TEST_CSV = "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/wrc_63468huigui/DLPromoter-SF/dataset/low_similarity.csv"
BATCH_SIZE = 256
NUM_WORKERS = 2


# ==========================================
# 2. 数据处理工具函数 (与训练集保持绝对一致)
# ==========================================
def clean_seq(s: str) -> str:
    s = s.strip().lower()
    return re.sub("[^acgt]", "z", s)


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
        _ = next(f, None)  # 跳过表头
        for line in f:
            if not line.strip(): continue
            s, y = line.strip().split(",")
            seqs.append(s)
            ys.append(float(y))
    return seqs, np.asarray(ys, dtype=np.float32)


# ---- 统计特征计算 (76维) ----
_BASES = ["a", "c", "g", "t"]
_KMER3 = [a + b + c for a in _BASES for b in _BASES for c in _BASES]
_KMER3_TO_IDX = {k: i for i, k in enumerate(_KMER3)}


def kmer3_freq(seq: str):
    counts = np.zeros((64,), dtype=np.float32)
    denom = 0
    for i in range(len(seq) - 2):
        k = seq[i:i + 3]
        if "z" in k: continue
        counts[_KMER3_TO_IDX[k]] += 1.0
        denom += 1
    if denom > 0: counts /= denom
    return counts


def gc_profile(seq: str, n_bins: int = 8):
    L = len(seq)
    out = np.zeros((n_bins,), dtype=np.float32)
    for b in range(n_bins):
        l, r = int(round(b * L / n_bins)), int(round((b + 1) * L / n_bins))
        valid = [c for c in seq[l:r] if c in "acgt"]
        if not valid: continue
        out[b] = sum(1 for c in valid if c in "gc") / float(len(valid))
    return out


def global_comp(seq: str):
    valid = [c for c in seq if c in "acgt"]
    if not valid: return np.zeros((2,), dtype=np.float32)
    gc = sum(1 for c in valid if c in "gc") / float(len(valid))
    return np.asarray([gc, 1.0 - gc], dtype=np.float32)


def longest_homopolymer(seq: str, base: str):
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
    return np.concatenate([
        kmer3_freq(seq), gc_profile(seq, n_bins),
        global_comp(seq), np.asarray([longest_homopolymer(seq, "a"), longest_homopolymer(seq, "t")], dtype=np.float32)
    ], axis=0).astype(np.float32)


def compute_metrics(y_true, y_pred):
    y_true, y_pred = y_true.astype(np.float64).ravel(), y_pred.astype(np.float64).ravel()
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Pearson": float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else 0.0,
        "Spearman": float(spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else 0.0,
    }


# ==========================================
# 3. 主测试逻辑
# ==========================================
def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")

    # 1. 加载运行配置文件
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        run_cfg = json.load(f)
    print(f"[*] 加载配置文件成功 (max_len={run_cfg['max_len']}, conv_kernels={run_cfg['conv_kernels']})")

    # 2. 加载归一化参数
    with open(NORM_PATH, "r", encoding="utf-8") as f:
        extra_norm = json.load(f)
    mu = np.array(extra_norm["mu"], dtype=np.float32)
    sd = np.array(extra_norm["sd"], dtype=np.float32)
    print(f"[*] 加载统计特征归一化参数成功")

    # 3. 读取并处理测试集数据
    print(f"[*] 正在处理测试集数据...")
    seq_test, y_test = load_csv(TEST_CSV)

    # 构建输入矩阵
    X_test_list = [one_hot_encode(clean_seq(s)) for s in seq_test]
    # 【关键】必须使用训练时的 max_len
    X_test = np.stack([pad_or_trunc(m, run_cfg["max_len"]) for m in X_test_list], axis=0)

    # 构建并归一化统计特征
    F_test = np.stack([build_extra_features(s, n_bins=run_cfg["gc_bins"]) for s in seq_test], axis=0)
    # 【关键】使用训练集的 mu 和 sd 进行标准化
    F_test = (F_test - mu) / sd

    # 转换为 Tensor 并创建 DataLoader
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(F_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
        ),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # 4. 初始化模型 (参数从 run_cfg 读取，保证结构百分百一致)
    model = TransformerHybridSEFusionModel(
        input_size=4,
        hidden_size=run_cfg["hidden_size"],
        output_size=1,
        dropout_rate=0.0,  # 推理阶段关闭 dropout
        extra_feat_dim=run_cfg["extra_feat_dim"],
        use_extra=run_cfg["use_extra"],
        conv_kernels=tuple(run_cfg["conv_kernels"]),
    ).to(device)

    # 5. 加载最佳权重
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    print(f"[*] 成功加载模型权重: {MODEL_PATH}\n")

    # 6. 开始预测与评估
    preds, trues = [], []
    with torch.no_grad():
        for x, feat, y in test_loader:
            x, feat, y = x.to(device), feat.to(device), y.to(device).view(-1)
            # 根据是否使用混合精度推理
            if hasattr(torch, "amp"):
                with torch.amp.autocast("cuda"):
                    out = model(x, feat).view(-1)
            else:
                with torch.cuda.amp.autocast():
                    out = model(x, feat).view(-1)

            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())

    y_true = np.concatenate(trues).ravel()
    y_pred = np.concatenate(preds).ravel()

    # 计算并打印指标
    metrics = compute_metrics(y_true, y_pred)

    print("========================================")
    print("           测试集评估结果 (TEST SET)      ")
    print("========================================")
    print(f"R² Score : {metrics['R2']:.4f}")
    print(f"RMSE     : {metrics['RMSE']:.4f}")
    print(f"MAE      : {metrics['MAE']:.4f}")
    print(f"Pearson  : {metrics['Pearson']:.4f}")
    print(f"Spearman : {metrics['Spearman']:.4f}")
    print("========================================")


if __name__ == "__main__":
    main()
