# Bài tập lớn: Xấp xỉ hàm số bằng MLP (PyTorch)

Xấp xỉ hàm mật độ Gaussian 2 chiều

$$
f(\mathbf{x}) = \frac{1}{2\pi\sqrt{\det C}}\exp\!\Big(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T C^{-1}(\mathbf{x}-\boldsymbol{\mu})\Big),
\quad \boldsymbol{\mu}=\begin{bmatrix}0\\0\end{bmatrix},\;
C=\begin{bmatrix}0.3 & -0.5\\ -0.5 & 2\end{bmatrix}
$$

trong miền $-2\le x_1\le 2$, $-4\le x_2\le 4$, bằng mạng perceptron đa tầng (MLP) tự xây dựng bằng `torch.nn.Module`.

---

## 1. Cấu trúc thư mục

```
.
├── codes/
│   ├── common/                  # Thư viện dùng chung
│   │   ├── config.py            # Đường dẫn, seed, miền dữ liệu, cấu hình baseline
│   │   ├── data.py              # Hàm Gaussian, lấy mẫu, chia tập, chuẩn hóa
│   │   ├── model.py             # Lớp MLP (kế thừa nn.Module)
│   │   ├── train.py             # Vòng lặp train/validation, early stopping, checkpoint
│   │   └── plots.py             # Các hàm vẽ biểu đồ
│   ├── experiments/             # Các thí nghiệm (testing)
│   │   ├── experiment.py        # Khung chạy nhóm A/B/C (song song, ghi log, vẽ hình)
│   │   ├── exp_A_data_size.py   # Nhóm A: số mẫu train
│   │   ├── exp_B_width.py       # Nhóm B: số nơ-ron tầng ẩn (1 tầng)
│   │   └── exp_C_depth.py       # Nhóm C: số tầng ẩn
│   └── best model/
│       └── train_best.py        # Huấn luyện & lưu mô hình tốt nhất, đánh giá test
├── graphs/                      # Hình ảnh (sinh tự động)
│   └── visualize_data.ipynb     # Bước 1: sinh dữ liệu, thống kê, vẽ và lưu hình
├── logs/                        # File CSV: dữ liệu, log từng lần chạy, bảng tổng hợp
├── reports/                     # Báo cáo Word/PDF
├── slides/                      # Slide thuyết trình
├── requirements.txt
└── README.md
```

Các thư mục `graphs/`, `logs/` và file trọng số `codes/best model/best_mlp.pt` được tạo khi chạy code.

---

## 2. Cài đặt môi trường

Yêu cầu: Python 3.10 trở lên. GPU không bắt buộc, code tự chọn CPU nếu không có CUDA.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> Nếu máy bạn dùng phiên bản khác, có thể cập nhật `requirements.txt` bằng lệnh `pip freeze` sau khi cài đặt.

---

## 3. Cách chạy (theo thứ tự)

Chạy tất cả các lệnh **từ thư mục gốc của dự án**. Không cần sửa đường dẫn tuyệt đối.

```bash
# Bước 1: sinh dữ liệu, kiểm tra công thức Gaussian, vẽ scatter 3D và bề mặt hàm thật
jupyter nbconvert --to notebook --execute --inplace graphs/visualize_data.ipynb

# Bước 2: ba nhóm thực nghiệm (có thể chạy độc lập)
python codes/experiments/exp_A_data_size.py
python codes/experiments/exp_B_width.py
python codes/experiments/exp_C_depth.py

# Bước 3: chọn cấu hình tốt nhất theo validation, huấn luyện, lưu checkpoint, đánh giá test
python "codes/best model/train_best.py"
```

Sau khi chạy xong, bạn sẽ có:

| Đầu ra | Vị trí |
|---|---|
| Dữ liệu train/val/test (CSV) | `logs/data/` |
| Log từng lần chạy (CSV, train/validation metrics) | `logs/exp_*/raw_runs.csv` |
| Lịch sử loss từng epoch (CSV) | `logs/exp_*/history_*.csv` |
| Bảng tổng hợp mean ± std (CSV, train/validation metrics) | `logs/exp_*/summary.csv` |
| Hình biểu đồ | `graphs/data/`, `graphs/exp_*/`, `graphs/best_model/` |
| Trọng số mô hình tốt nhất | `codes/best model/best_mlp.pt` |
| Kết quả test của mô hình tốt nhất | `logs/best_model/test_metrics.csv` |

**Thời gian chạy tham khảo:** trên CPU thông thường, mỗi lần huấn luyện mất vài giây đến vài chục giây. Toàn bộ nhóm A, B, C (mỗi cấu hình 3 seed) mất khoảng 15–30 phút.

**Chạy trong Jupyter Notebook:** nếu cần file `.ipynb`, tạo một notebook ở thư mục gốc và chạy trong từng ô, ví dụ:

```python
%run codes/experiments/exp_A_data_size.py
```

---

## 4. Cấu hình baseline và các nhóm thực nghiệm

Baseline (`codes/common/config.py`) dùng chung cho mọi nhóm, mỗi nhóm chỉ thay đổi đúng một yếu tố.

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Seed dữ liệu | 42 | Sinh pool train (2000) và validation (400) |
| Seed test | 2024 | Tập test 100 mẫu, độc lập với train/validation |
| Seed huấn luyện | 0, 1, 2 | Mỗi cấu hình chạy 3 seed, báo cáo mean ± std |
| Số mẫu train | 1000 | Nhóm A thay đổi |
| Số tầng ẩn | 2 | Nhóm C thay đổi; nhóm B cố định 1 tầng |
| Số nơ-ron/tầng | 8 | Nhóm B thay đổi (1 tầng) |
| Hàm kích hoạt | tanh | Phù hợp với hàm trơn như Gaussian |
| Optimizer | Adam | |
| Learning rate | 1e-2 | |
| Batch size | 32 | |
| Số epoch tối đa | 500 | Điều kiện dừng 1 |
| Early stopping (patience) | 40 | Điều kiện dừng 2: validation loss không giảm sau 40 epoch |
| Loss | MSELoss | Huấn luyện trên dữ liệu đã chuẩn hóa |

Các nhóm thực nghiệm:

| Nhóm | Tham số khảo sát | Giá trị | Script |
|---|---|---|---|
| A | Số mẫu train $A$ | 200, 400, …, 2000 | `exp_A_data_size.py` |
| B | Số nơ-ron tầng ẩn $B$ (1 tầng) | 1, 2, …, 10 | `exp_B_width.py` |
| C | Số tầng ẩn $C$ (8 nơ-ron/tầng) | 1, 2, 3, 4 | `exp_C_depth.py` |

---

## 5. Quy trình đánh giá công bằng

- **Chia dữ liệu cố định:** train và validation sinh từ `SEED_DATA = 42`, test sinh từ `SEED_TEST = 2024`. Mọi cấu hình dùng đúng cùng một bộ dữ liệu.
- **Nhóm A dùng tập con lồng nhau:** train $A$ mẫu là $A$ mẫu đầu tiên của pool 2000 mẫu, nên các tập nhỏ là tập con của tập lớn.
- **Chuẩn hóa không rò rỉ:** trung bình và độ lệch chuẩn của $x$ và $y$ chỉ tính trên tập train, rồi áp dụng cho validation và test.
- **Chọn mô hình bằng validation, không bằng test:** các sweep A/B/C chỉ tính metric train/validation. Tập test chỉ được đánh giá cho mô hình cuối cùng trong `train_best.py`.
- **Checkpoint tốt nhất:** trong mỗi lần huấn luyện, trọng số có validation loss thấp nhất được giữ lại.

---

## 6. Tái lập kết quả

- Mọi seed (dữ liệu, khởi tạo trọng số, thứ tự xáo trộn batch) đều được cố định.
- Đường dẫn được tính từ vị trí file, không phụ thuộc máy cá nhân.
- Kết quả có thể hơi khác giữa CPU và GPU hoặc giữa các phiên bản PyTorch do tính toán số thực. Nếu cần kết quả khớp tuyệt đối, hãy dùng đúng phiên bản trong `requirements.txt`.

---

## 7. Tải và nạp lại mô hình đã lưu

```python
import sys
import numpy as np
sys.path.insert(0, "codes")
from common.train import load_checkpoint, predict

model, scaler, cfg = load_checkpoint("codes/best model/best_mlp.pt")
print(predict(model, scaler, np.array([[0.0, 0.0], [1.0, -2.0]])))
```
