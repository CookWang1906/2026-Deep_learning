@echo off
REM ============================================================
REM  Cách dùng:
REM    run.bat         : chạy toàn bộ quy trình.
REM    run.bat quick   : chạy thử nhanh (1 seed, 20 epoch).
REM                      Kết quả được lưu trong các thư mục quick_*.
REM
REM  Các bước:
REM    [0/5] Kiểm tra và cài đặt thư viện.
REM    [1/5] Tạo dữ liệu và biểu đồ từ notebook.
REM    [2/5] Nhóm A: ảnh hưởng của kích thước tập huấn luyện.
REM    [3/5] Nhóm B: ảnh hưởng của độ rộng tầng ẩn.
REM    [4/5] Nhóm C: ảnh hưởng của số tầng ẩn.
REM    [5/5] Chọn mô hình theo validation và đánh giá trên test.
REM
REM  Sau mỗi nhóm, tự động kiểm tra summary.csv và dừng lại nếu bị thiếu.
REM ============================================================

chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONWARNINGS=ignore

REM Chuyển về thư mục gốc của dự án.
cd /d "%~dp0"

REM - Xử lý tham số quick -
set EXTRA=
set PREFIX=
if /i "%~1"=="quick" (
    set EXTRA=--quick
    set PREFIX=quick_
)

echo.
echo ============================================================
echo  2D Gaussian Function Approximation with an MLP (PyTorch)
echo  Directory: %CD%
if defined EXTRA (
    echo  MODE: Quick Test (results are saved in quick_* folders)
) else (
    echo  MODE: Full
)
echo ============================================================

REM - Tạo thư mục đầu ra nếu còn thiếu -
if not exist logs mkdir logs
if not exist graphs mkdir graphs
if not exist "codes\best model" mkdir "codes\best model"

REM - [0/5] Tự động kiểm tra và cài thư viện -  
set STEP=[0/5] Check and install dependencies
echo.
echo [0/5] Checking dependencies...
python -c "import torch, pandas, matplotlib, scipy, nbconvert, ipykernel" >nul 2>&1
if errorlevel 1 (
    echo     Dependencies are missing. Installing from requirements.txt...
    python -m pip install -r requirements.txt
    if errorlevel 1 goto :error
) else (
    echo     All dependencies are installed.
)

REM - [1/5] Tạo dữ liệu và biểu đồ bằng Notebook -
set STEP=[1/5] Generate data and plots (Notebook)
echo.
echo [1/5] Generating data and plots (Notebook graphs\visualize_data.ipynb)...
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 graphs\visualize_data.ipynb
if errorlevel 1 goto :error
if not exist "graphs\data\true_surface.png" (
    echo [ERROR] Notebook finished, but graphs\data\true_surface.png was not created.
    goto :error
)

REM - [2/5] Nhóm A -
set STEP=[2/5] Group A - Training set size
echo.
echo [2/5] Group A - Training set size...
python codes\experiments\exp_A_data_size.py %EXTRA%
if errorlevel 1 goto :error
if not exist "logs\%PREFIX%exp_A_data_size\summary.csv" (
    echo [ERROR] Group A finished, but logs\%PREFIX%exp_A_data_size\summary.csv was not created.
    goto :error
)

REM - [3/5] Nhóm B -
set STEP=[3/5] Group B - Hidden-layer width
echo.
echo [3/5] Group B - Hidden-layer width...
python codes\experiments\exp_B_width.py %EXTRA%
if errorlevel 1 goto :error
if not exist "logs\%PREFIX%exp_B_width\summary.csv" (
    echo [ERROR] Group B finished, but logs\%PREFIX%exp_B_width\summary.csv was not created.
    goto :error
)

REM - [4/5] Nhóm C -
set STEP=[4/5] Group C - Number of hidden layers
echo.
echo [4/5] Group C - Number of hidden layers...
python codes\experiments\exp_C_depth.py %EXTRA%
if errorlevel 1 goto :error
if not exist "logs\%PREFIX%exp_C_depth\summary.csv" (
    echo [ERROR] Group C finished, but logs\%PREFIX%exp_C_depth\summary.csv was not created.
    goto :error
)

REM - [5/5] Chọn mô hình tốt nhất và đánh giá trên test -
set STEP=[5/5] Select the best model and evaluate on the test set
echo.
echo [5/5] Selecting the best model using validation, training it, and evaluating on the test set...
python "codes\best model\train_best.py" %EXTRA%
if errorlevel 1 goto :error

echo.
echo ============================================================
echo  ALL STEPS COMPLETED.
echo    graphs\           : plots
echo    logs\             : result CSV files
echo    codes\best model\ : best model weights
echo ============================================================
pause
exit /b 0

:error
echo.
echo ============================================================
echo  [ERROR] Stopped at step: %STEP%
echo  Check the error message above for details.
echo ============================================================
pause
exit /b 1
