import numpy as np
import pandas as pd
from common.config import (
    C_MAT,
    LOGS_DIR,
    MU,
    N_TEST,
    N_TRAIN_POOL,
    N_VAL,
    SEED_DATA,
    SEED_TEST,
    X1_RANGE,
    X2_RANGE,
)

def gaussian_2d(x, mu=MU, cov=C_MAT):
    # Tính mật độ Gaussian cho một hoặc nhiều điểm đầu vào.
    x = np.atleast_2d(x)
    diff = x - mu  # Mỗi hàng là một vector sai khác.
    cov_inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    # Tính dạng toàn phương cho từng điểm.
    maha_sq = np.einsum("ni,ij,nj->n", diff, cov_inv, diff)
    return np.exp(-0.5 * maha_sq) / (2.0 * np.pi * np.sqrt(det))

def sample_inputs(n, rng):
    # Lấy mẫu đều trong miền xác định.
    x1 = rng.uniform(X1_RANGE[0], X1_RANGE[1], size=n)
    x2 = rng.uniform(X2_RANGE[0], X2_RANGE[1], size=n)
    return np.column_stack([x1, x2])

def make_dataset(save=True):
    # Tạo các tập train, validation và test độc lập.
    rng = np.random.default_rng(SEED_DATA)
    X_pool = sample_inputs(N_TRAIN_POOL + N_VAL, rng)
    X_train, X_val = X_pool[:N_TRAIN_POOL], X_pool[N_TRAIN_POOL:]
    X_test = sample_inputs(N_TEST, np.random.default_rng(SEED_TEST))

    data = {
        "train": (X_train, gaussian_2d(X_train)),
        "val": (X_val, gaussian_2d(X_val)),
        "test": (X_test, gaussian_2d(X_test)),
    }

    if save:
        out_dir = LOGS_DIR / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        for split, (X, y) in data.items():
            pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y}).to_csv(
                out_dir / f"{split}.csv", index=False
            )
    return data

def describe_dataset(data):
    # Tính thống kê mô tả cho từng tập dữ liệu.
    rows = []
    for split, (X, y) in data.items():
        for name, v in (("x1", X[:, 0]), ("x2", X[:, 1]), ("y", y)):
            rows.append(
                {
                    "split": split,
                    "variable": name,
                    "n": len(v),
                    "mean": v.mean(),
                    "std": v.std(),
                    "min": v.min(),
                    "max": v.max(),
                }
            )
    return pd.DataFrame(rows)


class Standardizer:
    # Chuẩn hóa đầu vào và đầu ra, đồng thời lưu thống kê để đảo chuẩn hóa.

    def __init__(self):
        self.x_mean = self.x_std = None
        self.y_mean = self.y_std = None

    def fit(self, X, y):
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-8
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) + 1e-8
        return self

    def transform_x(self, X):
        return (X - self.x_mean) / self.x_std

    def transform_y(self, y):
        return (y - self.y_mean) / self.y_std

    def inverse_y(self, y_scaled):
        return y_scaled * self.y_std + self.y_mean

    def to_dict(self):
        return {
            "x_mean": self.x_mean.tolist(),
            "x_std": self.x_std.tolist(),
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.x_mean = np.array(d["x_mean"])
        obj.x_std = np.array(d["x_std"])
        obj.y_mean = float(d["y_mean"])
        obj.y_std = float(d["y_std"])
        return obj
