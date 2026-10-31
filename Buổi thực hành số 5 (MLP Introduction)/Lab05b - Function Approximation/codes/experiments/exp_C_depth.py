"""Nhóm C: khảo sát số tầng ẩn."""

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

DEPTHS = [1, 2, 3, 4]

def main():
    data = make_dataset(save=False)
    run_sweep(
        exp_name = "exp_C_depth",
        param_name = "n_layers",
        values = DEPTHS,
        make_cfg = lambda c: config_with(BASELINE, n_layers=c),
        data = data,
        xlabel = "Number of hidden layers (C)",
    )
    print("\nGroup C complete. See logs/exp_C_depth/ and graphs/exp_C_depth/.")

if __name__ == "__main__":
    main()
