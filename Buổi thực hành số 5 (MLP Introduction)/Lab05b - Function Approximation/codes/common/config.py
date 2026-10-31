import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

# Đường dẫn thư mục
def _find_root():
    # Tìm thư mục gốc dựa trên tệp cấu hình dự án.
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "requirements.txt").exists() or (p / "README.md").exists():
            return p
    return here.parents[2]

ROOT = _find_root()
CODES_DIR = ROOT / "codes"
GRAPHS_DIR = ROOT / "graphs"
LOGS_DIR = ROOT / "logs"
BEST_MODEL_DIR = CODES_DIR / "best model"

# Chế độ nhanh (--quick)
QUICK = "--quick" in sys.argv  # Bật chế độ nhanh khi có tham số --quick.
QUICK_EPOCHS = 20  # Số epoch tối đa ở chế độ nhanh.


def results_name(name):
    # Thêm tiền tố riêng cho kết quả chạy nhanh.
    return f"quick_{name}" if QUICK else name

def ensure_dirs():
    # Tạo các thư mục đầu ra.
    for d in (GRAPHS_DIR, LOGS_DIR, BEST_MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

# Bài toán Gaussian hai chiều
MU = np.array([0.0, 0.0])
C_MAT = np.array([[0.3, -0.5], [-0.5, 2.0]])
X1_RANGE = (-2.0, 2.0)
X2_RANGE = (-4.0, 4.0)

# Dữ liệu và seed
SEED_DATA = 42  # Tạo dữ liệu train và validation.
SEED_TEST = 2024  # Tạo tập test độc lập.
N_TRAIN_POOL = 2000  # Cỡ pool train lớn nhất.
N_VAL = 400  # Tập validation cố định.
N_TEST = 100  # Chỉ dùng để đánh giá cuối.

SEEDS = [0, 1, 2]  # Chạy mỗi cấu hình với ba seed.

# Giới hạn số tiến trình huấn luyện song song.
MAX_WORKERS = max(1, min(4, os.cpu_count() or 1))

# Cấu hình huấn luyện MLP
@dataclass
class Config:
    n_train: int = 1000
    n_layers: int = 2
    width: int = 8
    activation: str = "tanh"
    lr: float = 1e-2
    batch_size: int = 32
    max_epochs: int = 500
    patience: int = 40
    weight_decay: float = 0.0

    @property
    def hidden_sizes(self):
        # Tạo kích thước cho từng tầng ẩn.
        return tuple([self.width] * self.n_layers)

    def to_dict(self):
        return asdict(self)

# Mỗi nhóm thực nghiệm chỉ thay đổi một yếu tố so với cấu hình này.
BASELINE = Config()
