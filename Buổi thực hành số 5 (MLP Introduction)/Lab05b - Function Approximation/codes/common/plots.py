import matplotlib
matplotlib.use("Agg")  # Chọn backend không cần giao diện đồ họa.
import matplotlib.pyplot as plt
import numpy as np
from common.config import X1_RANGE, X2_RANGE
from common.data import gaussian_2d
from common.train import predict


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved figure: {path}")

def _grid(n1=120, n2=240):
    # Tạo lưới điểm đều để vẽ bề mặt.
    x1 = np.linspace(X1_RANGE[0], X1_RANGE[1], n1)
    x2 = np.linspace(X2_RANGE[0], X2_RANGE[1], n2)
    G1, G2 = np.meshgrid(x1, x2)
    return G1, G2, np.column_stack([G1.ravel(), G2.ravel()])

def plot_scatter3d(X, y, path, title="Training data: (x1, x2, f(x))"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    p = ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap="viridis", s=8)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y = f(x)")
    ax.set_title(title)
    fig.colorbar(p, ax=ax, shrink=0.6, label="y")
    _save(fig, path)

def plot_true_surface(path, title="True function surface f(x1, x2)"):
    G1, G2, grid = _grid()
    Z = gaussian_2d(grid).reshape(G1.shape)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(G1, G2, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")
    ax.set_title(title)
    _save(fig, path)

def plot_distributions(X, y, path):
    # Vẽ histogram của các đặc trưng và nhãn trên tập train.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, v, name in zip(axes, [X[:, 0], X[:, 1], y], ["x1", "x2", "y = f(x)"]):
        ax.hist(v, bins=40, color="steelblue", alpha=0.85)
        ax.set_title(f"Distribution of {name}")
        ax.set_xlabel(name)
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)
    _save(fig, path)

def plot_pred_vs_true(
    y_true, y_pred, path, title = "True vs. predicted values (test set)"
):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=18, alpha=0.7)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", label="Predicted = true (ideal)")
    ax.set_xlabel("True y")
    ax.set_ylabel("Predicted y")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)

def plot_approx_surface(model, scaler, path, title = "MLP approximation surface"):
    G1, G2, grid = _grid()
    Z = predict(model, scaler, grid).reshape(G1.shape)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(G1, G2, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("Predicted y")
    ax.set_title(title)
    _save(fig, path)

def plot_loss_curve(history, path, title = "Normalized loss convergence"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss")
    ax.plot(history["epoch"], history["val_loss"], label="Validation loss")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized MSE (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)

def plot_sweep(summary, xlabel, out_dir):
    # Vẽ metric train/validation cho một nhóm thí nghiệm.
    x = summary.index.values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for metric, label in [("train_MSE", "Train MSE"), ("val_MSE", "Validation MSE")]:
        axes[0].errorbar(
            x,
            summary[f"{metric}_mean"],
            yerr=summary[f"{metric}_std"].fillna(0),
            marker="o",
            capsize=3,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("MSE (original y scale, log)")
    axes[0].set_title(f"Error by {xlabel.lower()}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for metric, label in [
        ("val_MAE", "Validation MAE"),
        ("val_RMSE", "Validation RMSE"),
    ]:
        axes[1].errorbar(
            x,
            summary[f"{metric}_mean"],
            yerr=summary[f"{metric}_std"].fillna(0),
            marker="s",
            capsize=3,
            label=label,
        )
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Validation error (original y scale)")
    axes[1].set_title("Validation MAE and RMSE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    _save(fig, out_dir / "sweep_errors.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].errorbar(
        x,
        summary["train_time_s_mean"],
        yerr=summary["train_time_s_std"].fillna(0),
        marker="o",
        capsize=3,
        color="tab:green",
    )
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Training time (seconds)")
    axes[0].set_title("Training time")
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(
        x,
        summary["epochs_run_mean"],
        yerr=summary["epochs_run_std"].fillna(0),
        marker="o",
        capsize=3,
        color="tab:purple",
    )
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Number of epochs")
    axes[1].set_title("Epochs before stopping")
    axes[1].grid(alpha=0.3)
    _save(fig, out_dir / "sweep_time_epochs.png")

def plot_learning_curves(histories, xlabel, path):
    # Vẽ loss train và validation cho từng cấu hình.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for v, hist in histories.items():
        axes[0].plot(hist["epoch"], hist["train_loss"], label=f"{xlabel} = {v}")
        axes[1].plot(hist["epoch"], hist["val_loss"], label=f"{xlabel} = {v}")
    for ax, t in zip(axes, ["Train loss", "Validation loss"]):
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized MSE (log scale)")
        ax.set_title(f"{t} by epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    _save(fig, path)
