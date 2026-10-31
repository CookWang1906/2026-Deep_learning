import common.silence
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from common.config import (
    GRAPHS_DIR,
    LOGS_DIR,
    MAX_WORKERS,
    QUICK,
    QUICK_EPOCHS,
    SEEDS,
    ensure_dirs,
    results_name,
)
from common.plots import plot_sweep, plot_learning_curves
from common.train import train_model, predict, regression_metrics

# Các metric được tổng hợp vào summary.csv.
METRIC_COLS = [
    "train_MSE",
    "train_MAE",
    "train_RMSE",
    "val_MSE",
    "val_MAE",
    "val_RMSE",
    "epochs_run",
    "train_time_s",
]

def _init_worker():
    # Giới hạn mỗi tiến trình ở một luồng CPU.
    import torch

    torch.set_num_threads(1)

def evaluate_run(cfg, seed, data):
    # Huấn luyện một lần và chỉ đánh giá trên train/validation.
    model, scaler, history, info = train_model(cfg, seed, data)

    row = {"seed": seed, **cfg.to_dict(), **info}
    for split in ("train", "val"):
        X, y = data[split]
        if split == "train":  # Dùng đúng các mẫu đã huấn luyện.
            X, y = X[: cfg.n_train], y[: cfg.n_train]
        metrics = regression_metrics(y, predict(model, scaler, X))
        for k, v in metrics.items():
            row[f"{split}_{k}"] = v
    return row, history

def run_sweep(exp_name, param_name, values, make_cfg, data, xlabel, seeds=SEEDS):
    # Chạy các cấu hình của một nhóm thực nghiệm.
    ensure_dirs()
    out_name = results_name(exp_name)
    log_dir = LOGS_DIR / out_name
    graph_dir = GRAPHS_DIR / out_name
    log_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    if QUICK:
        seeds = seeds[:1]

    # Tạo danh sách cấu hình và seed.
    jobs = []
    for v in values:
        cfg = make_cfg(v)
        if QUICK:
            cfg = replace(cfg, max_epochs=QUICK_EPOCHS)
        for s in seeds:
            jobs.append((v, s, cfg))

    total = len(jobs)
    workers = max(1, min(MAX_WORKERS, total))
    sweep_data = {"train": data["train"], "val": data["val"]}
    mode = "QUICK" if QUICK else "FULL"
    print(
        f"[{out_name}] {mode} mode: {total} runs, " f"{workers} parallel workers",
        flush=True,
    )

    results = {}
    if workers == 1:
        for k, (v, s, cfg) in enumerate(jobs, 1):
            print(f"  ({k}/{total}) {param_name}={v}, seed={s}", flush=True)
            results[(v, s)] = evaluate_run(cfg, s, sweep_data)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            futures = {
                ex.submit(evaluate_run, cfg, s, sweep_data): (v, s)
                for v, s, cfg in jobs
            }
            for k, fut in enumerate(as_completed(futures), 1):
                v, s = futures[fut]
                results[(v, s)] = fut.result()
                print(
                    f"  ({k}/{total}) completed {param_name}={v}, seed={s}", flush=True
                )

    # Lưu kết quả theo thứ tự tham số và seed.
    rows, first_seed_histories = [], {}
    for v in values:
        for s in seeds:
            row, history = results[(v, s)]
            row[param_name] = v
            rows.append(row)
            history.to_csv(
                log_dir / f"history_{param_name}{v}_seed{s}.csv", index=False
            )
            if s == seeds[0]:
                first_seed_histories[v] = history

    raw = pd.DataFrame(rows)
    raw.to_csv(log_dir / "raw_runs.csv", index=False)

    summary = raw.groupby(param_name)[METRIC_COLS].agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary.to_csv(log_dir / "summary.csv")

    plot_sweep(summary, xlabel, graph_dir)
    plot_learning_curves(
        first_seed_histories, xlabel, graph_dir / "learning_curves.png"
    )

    print(
        f"\n[{out_name}] Summary (mean and standard deviation across "
        f"{len(seeds)} seeds):"
    )
    show = summary[
        [
            "train_MSE_mean",
            "val_MSE_mean",
            "val_MSE_std",
            "val_MAE_mean",
            "val_RMSE_mean",
            "epochs_run_mean",
            "train_time_s_mean",
        ]
    ]
    print(show.round(6).to_string())
    return raw, summary

def config_with(base, **overrides):
    # Tạo cấu hình mới với các tham số được ghi đè.
    return replace(base, **overrides)
