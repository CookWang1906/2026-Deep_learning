import copy
import random
import time
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)

import common.silence
from common.config import Config
from common.data import Standardizer
from common.model import MLP

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TensorPairDataset(Dataset):
    # Dataset lưu cặp đầu vào và nhãn dưới dạng tensor.

    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def set_seed(seed):
    # Cố định seed cho các thư viện dùng trong quá trình huấn luyện.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _make_loader(X, y, batch_size, shuffle, seed=0):
    # Tạo DataLoader có thứ tự lấy mẫu xác định.
    ds = TensorPairDataset(X, y)
    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)  # Giữ thứ tự xáo trộn có thể tái lập.
        base = RandomSampler(ds, generator=g)
    else:
        base = SequentialSampler(ds)
    batch_sampler = BatchSampler(base, batch_size=batch_size, drop_last=False)
    return DataLoader(ds, sampler=batch_sampler, batch_size=None)

def run_epoch(model, loader, loss_fn, optimizer=None):
    # Chạy một epoch huấn luyện hoặc đánh giá.
    is_train = optimizer is not None
    model.train(is_train)
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()

    total = torch.zeros((), device=DEVICE)
    n = 0
    with grad_ctx:
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)  # Lan truyền xuôi.
            loss = loss_fn(pred, yb)
            if is_train:
                optimizer.zero_grad(set_to_none=True)  # Xóa gradient cũ.
                loss.backward()  # Lan truyền ngược.
                optimizer.step()  # Cập nhật trọng số.
            total += loss.detach() * xb.size(0)
            n += xb.size(0)
    return total.item() / n

def train_model(cfg: Config, seed: int, data: dict):
    # Huấn luyện MLP và trả về checkpoint tốt nhất theo validation loss.
    set_seed(seed)

    X_tr, y_tr = data["train"][0][: cfg.n_train], data["train"][1][: cfg.n_train]
    X_va, y_va = data["val"]

    # Chỉ khớp bộ chuẩn hóa trên tập train.
    scaler = Standardizer().fit(X_tr, y_tr)
    train_loader = _make_loader(
        scaler.transform_x(X_tr),
        scaler.transform_y(y_tr),
        cfg.batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = _make_loader(
        scaler.transform_x(X_va),
        scaler.transform_y(y_va),
        batch_size=256,
        shuffle=False,
    )

    model = MLP(
        in_features=2,
        hidden_sizes=cfg.hidden_sizes,
        out_features=1,
        activation=cfg.activation,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = nn.MSELoss()

    best_val, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = []
    t0 = time.perf_counter()

    epoch = 0
    for epoch in range(1, cfg.max_epochs + 1):
        tr_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        va_loss = run_epoch(model, val_loader, loss_fn)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})

        if va_loss < best_val - 1e-8:  # Chỉ lưu khi validation loss giảm.
            best_val, best_epoch, wait = va_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    train_time = time.perf_counter() - t0
    model.load_state_dict(best_state)  # Khôi phục trọng số tốt nhất.
    model.eval()

    info = {
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_val_loss_std": best_val,
        "train_time_s": train_time,
    }
    return model, scaler, pd.DataFrame(history), info

@torch.no_grad()
def predict(model, scaler, X):
    # Dự đoán đầu ra trên thang đo ban đầu.
    model.eval()
    xs = torch.as_tensor(scaler.transform_x(X), dtype=torch.float32, device=DEVICE)
    y_scaled = model(xs).cpu().numpy().ravel()
    return scaler.inverse_y(y_scaled)

def regression_metrics(y_true, y_pred):
    """MSE, MAE, RMSE."""
    err = np.asarray(y_true) - np.asarray(y_pred)
    mse = float(np.mean(err**2))
    return {"MSE": mse, "MAE": float(np.mean(np.abs(err))), "RMSE": float(np.sqrt(mse))}

def save_checkpoint(path, model, scaler, cfg, seed, info):
    # Lưu trọng số, cấu hình và tham số chuẩn hóa.
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": cfg.to_dict(),
            "scaler": scaler.to_dict(),
            "seed": seed,
            "info": info,
        },
        path,
    )

def load_checkpoint(path):
    # Nạp checkpoint và trả về mô hình, bộ chuẩn hóa, cấu hình.
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    cfg = Config(**ckpt["config"])
    model = MLP(
        in_features=2,
        hidden_sizes=cfg.hidden_sizes,
        out_features=1,
        activation=cfg.activation,
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    scaler = Standardizer.from_dict(ckpt["scaler"])
    return model, scaler, cfg
