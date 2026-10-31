import sys
from dataclasses import replace
from pathlib import Path
import pandas as pd

def _bootstrap():
    # Thêm thư mục mã nguồn vào sys.path.
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

_bootstrap()

import common.silence
from common.config import (
    BEST_MODEL_DIR,
    Config,
    GRAPHS_DIR,
    LOGS_DIR,
    QUICK,
    QUICK_EPOCHS,
    SEEDS,
    ensure_dirs,
    results_name,
)
from common.data import make_dataset
from common.plots import plot_approx_surface, plot_loss_curve, plot_pred_vs_true
from common.train import (
    predict,
    regression_metrics,
    save_checkpoint,
    train_model,
)

# Các nhóm dùng để chọn cấu hình.
EXP_GROUPS = ["exp_A_data_size", "exp_B_width", "exp_C_depth"]

def select_best_config():
    # Chọn cấu hình có validation MSE trung bình thấp nhất.
    frames = []
    for g in EXP_GROUPS:
        f = LOGS_DIR / results_name(g) / "raw_runs.csv"
        if not f.exists():
            raise FileNotFoundError(
                f"Missing experiment results: {f}\n"
                "Run the corresponding experiment first."
            )
        frames.append(pd.read_csv(f))
    raw = pd.concat(frames, ignore_index=True)

    cfg_cols = list(Config().to_dict().keys())
    table = (
        raw.groupby(cfg_cols)["val_MSE"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean")
        .reset_index(drop=True)
    )

    best_row = table.iloc[0]
    # Khôi phục đúng kiểu dữ liệu của cấu hình.
    best = Config(**{c: type(getattr(Config(), c))(best_row[c]) for c in cfg_cols})
    return best, table.head(5)

def main():
    ensure_dirs()
    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Chọn cấu hình dựa trên validation.
    best_cfg, top5 = select_best_config()
    if QUICK:
        best_cfg = replace(best_cfg, max_epochs=QUICK_EPOCHS)
    seeds = SEEDS[:1] if QUICK else SEEDS

    print("Top 5 configurations by mean validation MSE:")
    print(top5.round(6).to_string(index=False))
    print(f"\nSelected configuration: {best_cfg}\n")

    out_logs = LOGS_DIR / results_name("best_model")
    out_graphs = GRAPHS_DIR / results_name("best_model")
    out_logs.mkdir(parents=True, exist_ok=True)
    out_graphs.mkdir(parents=True, exist_ok=True)

    data = make_dataset(save=False)
    X_va, y_va = data["val"]

    # Chọn seed dựa trên validation MSE.
    best = None  # Lưu metric, seed, mô hình và lịch sử tốt nhất.
    for seed in seeds:
        model, scaler, history, info = train_model(best_cfg, seed, data)
        val_mse = regression_metrics(y_va, predict(model, scaler, X_va))["MSE"]
        print(
            f"seed={seed}: validation MSE={val_mse:.6e}, "
            f"best epoch={info['best_epoch']}"
        )
        if best is None or val_mse < best[0]:
            best = (val_mse, seed, model, scaler, history, info)

    val_mse, seed, model, scaler, history, info = best
    print(f"\nSelected seed: {seed} (lowest validation MSE).")

    # Lưu checkpoint trước khi đánh giá test.
    ckpt_name = "quick_best_mlp.pt" if QUICK else "best_mlp.pt"
    ckpt_path = BEST_MODEL_DIR / ckpt_name
    save_checkpoint(ckpt_path, model, scaler, best_cfg, seed, info)
    print(f"Checkpoint saved: {ckpt_path}")

    X_te, y_te = data["test"]
    y_pred = predict(model, scaler, X_te)
    metrics = regression_metrics(y_te, y_pred)
    print("\nTest results (independent set, 100 samples):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6e}")

    pd.DataFrame([best_cfg.to_dict()]).assign(seed=seed).to_csv(
        out_logs / "selected_config.csv", index=False
    )
    top5.to_csv(out_logs / "top5_configs.csv", index=False)
    pd.DataFrame([{"seed": seed, **best_cfg.to_dict(), **metrics}]).to_csv(
        out_logs / "test_metrics.csv", index=False
    )
    pd.DataFrame(
        {"x1": X_te[:, 0], "x2": X_te[:, 1], "y_true": y_te, "y_pred": y_pred}
    ).to_csv(out_logs / "test_predictions.csv", index=False)
    history.to_csv(out_logs / "history.csv", index=False)

    plot_pred_vs_true(y_te, y_pred, out_graphs / "pred_vs_true_test.png")
    plot_approx_surface(model, scaler, out_graphs / "approx_surface.png")
    plot_loss_curve(history, out_graphs / "loss_curve.png")
    print("\nComplete. See codes/best model/ and graphs/best_model/.")

if __name__ == "__main__":
    main()
