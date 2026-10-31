"""Nhóm A: khảo sát ảnh hưởng của kích thước tập train."""

import sys
from pathlib import Path

def _bootstrap():
    # Thêm thư mục mã nguồn và thí nghiệm vào sys.path.
    here = Path(__file__).resolve()
    codes = next(
        (
            p
            for p in [here.parent, *here.parents]
            if (p / "common" / "config.py").exists()
        ),
        None,
    )
    if codes is None:
        raise RuntimeError("Could not find codes/common/config.py")
    sys.path.insert(0, str(codes))
    exp_file = next(codes.rglob("experiment.py"), None)
    if exp_file is None:
        raise RuntimeError("Could not find experiment.py under codes/")
    sys.path.insert(0, str(exp_file.parent))

_bootstrap()

import common.silence  # noqa: F401
from common.config import BASELINE
from common.data import make_dataset
from experiment import config_with, run_sweep

SIZES = list(range(200, 2001, 200))  # Mười kích thước từ 200 đến 2000.

def main():
    data = make_dataset(save=False)
    run_sweep(
        exp_name = "exp_A_data_size",
        param_name = "n_train",
        values = SIZES,
        make_cfg = lambda n: config_with(BASELINE, n_train=n),
        data = data,
        xlabel = "Training samples (A)",
    )
    print("\nGroup A complete. See logs/exp_A_data_size/ and graphs/exp_A_data_size/.")

if __name__ == "__main__":
    main()
