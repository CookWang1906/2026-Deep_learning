"""Nhóm B: khảo sát độ rộng MLP một tầng ẩn."""

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

WIDTHS = list(range(1, 11))  # 1..10

def main():
    data = make_dataset(save=False)
    run_sweep(
        exp_name = "exp_B_width",
        param_name = "width",
        values = WIDTHS,
        make_cfg = lambda b: config_with(BASELINE, n_layers=1, width=b),
        data = data,
        xlabel = "Hidden-layer width (B)",
    )
    print("\nGroup B complete. See logs/exp_B_width/ and graphs/exp_B_width/.")

if __name__ == "__main__":
    main()
