# Rental Price Prediction API — Da Nang

Backend dự đoán giá thuê phòng trọ tại Đà Nẵng bằng Ridge Regression, cung cấp API bằng FastAPI và giải thích kết quả bằng SHAP. Dự án cũng có script huấn luyện model và giao diện Streamlit chạy riêng.

> **Phạm vi triển khai:** Render được cấu hình để chạy **FastAPI backend**. Giao diện Streamlit (`app.py`) hiện gọi `http://localhost:8000`, vì vậy chỉ dùng trực tiếp khi backend chạy trên cùng máy. Muốn đưa giao diện lên hosting riêng, cần cấu hình URL backend cho giao diện ở một thay đổi riêng.

## Mục lục

- [Tính năng](#tính-năng)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Model và dữ liệu](#model-và-dữ-liệu)
- [Chạy local](#chạy-local)
- [API](#api)
- [Triển khai backend lên Render](#triển-khai-backend-lên-render)
- [Huấn luyện lại model](#huấn-luyện-lại-model)
- [Đẩy dự án lên GitHub](#đẩy-dự-án-lên-github)
- [Lưu ý](#lưu-ý)

## Tính năng

- Dự đoán giá thuê theo VND/tháng từ đặc điểm phòng trọ.
- Trả về khoảng giá tham khảo dựa trên sai số median APE của tập test.
- Giải thích các yếu tố ảnh hưởng đến dự đoán bằng SHAP.
- Cung cấp metadata, danh sách lựa chọn và trạng thái dịch vụ qua API.
- Hỗ trợ huấn luyện lại model từ CSV bằng `train.py`.

## Cấu trúc dự án

```text
.
├── app.py                       # Giao diện Streamlit (chạy riêng)
├── main.py                      # FastAPI backend và các endpoint
├── train.py                     # Huấn luyện, đánh giá và xuất model
├── rental_price_model_tro.pkl   # Model đã huấn luyện, được backend nạp
├── model_metadata.json          # Metadata, đặc trưng và phiên bản model
├── rental_room_rows1.csv        # Dữ liệu; backend dùng làm nền SHAP
├── pyproject.toml               # Khai báo project và dependencies phát triển
├── uv.lock                      # Lockfile cho môi trường uv
├── requirements-render.txt      # Dependencies runtime ghim theo model hiện tại
├── render.yaml                  # Render Blueprint cho FastAPI
├── .gitignore                   # Loại môi trường local và file sinh khi train
└── README.md
```

Các file Python, model và dữ liệu được giữ ở thư mục gốc vì ứng dụng đang tham chiếu tới chúng bằng đường dẫn tương đối. Không chuyển chúng sang thư mục con nếu chưa cập nhật các đường dẫn tương ứng trong code.

## Model và dữ liệu

Backend nạp các artifact sau khi khởi động:

| File | Mục đích |
|---|---|
| `rental_price_model_tro.pkl` | Pipeline tiền xử lý và model Ridge đã fit |
| `model_metadata.json` | Tên model, đặc trưng đầu vào, danh sách category, metric và phiên bản thư viện |
| `rental_room_rows1.csv` | Dữ liệu nền tùy chọn để khởi tạo phần giải thích SHAP |

Model hiện tại được tạo với `scikit-learn 1.9.1`, `pandas 3.1.0`, `numpy 2.5.4` và `joblib 1.6.0`. Các phiên bản phục vụ backend được ghim trong `requirements-render.txt` để giảm rủi ro không tương thích khi nạp file pickle.

Nếu không có CSV nền, backend vẫn có thể chạy với cách tạo SHAP fallback đã được triển khai trong `main.py`; kết quả giải thích sẽ dùng baseline thay thế. Có thể đặt đường dẫn artifact khác bằng các biến môi trường `MODEL_PATH`, `META_PATH` và `CSV_PATH`.

Metric holdout được lưu trong metadata của model hiện tại:

| Metric | Giá trị |
|---|---:|
| R² trên log(giá) | 0.5124 |
| MAE | 713,852 VND |
| Median APE | 18.41% |
| Dự đoán trong ±20% | 53.13% |

Đây là kết quả đánh giá của model đã lưu, không phải cam kết hiệu năng trên dữ liệu mới.

## Chạy local

Yêu cầu Python **3.14**. Có thể dùng `uv` để đồng bộ đúng lockfile:

```powershell
python -m pip install uv
uv sync --frozen
```

Khởi chạy FastAPI:

```powershell
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Mở tài liệu API tại <http://localhost:8000/docs>, hoặc kiểm tra trạng thái tại <http://localhost:8000/health>.

Để chạy giao diện Streamlit trong một terminal khác:

```powershell
uv run streamlit run app.py
```

Giao diện mặc định kết nối tới backend local tại `http://localhost:8000`.

## API

| Method | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/health` | Trạng thái server và model |
| `GET` | `/meta` | Metadata model, metric và danh sách giá trị hợp lệ |
| `POST` | `/predict` | Giá dự đoán, khoảng tham khảo và giải thích SHAP |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc |

Ví dụ request `POST /predict`:

```json
{
  "size_m2": 25,
  "dist_to_cbd": 4.0,
  "dist_to_beach": 2.0,
  "dist_to_nearest_uni": 1.5,
  "has_balcony": 1,
  "has_elevator": 0,
  "has_window": 0,
  "furnishing_status": "Đầy đủ nội thất",
  "ward_name": "Hòa Khánh",
  "nearest_beach_code": "XuanThieu",
  "nearest_uni_code": "DUT"
}
```

Các category hợp lệ được trả về bởi `GET /meta`. Response dự đoán gồm `prediction`, `explanation` và thông tin model.

## Triển khai backend lên Render

Blueprint trong `render.yaml` tạo một **Web Service** chạy `main:app`, cài các dependency runtime từ `requirements-render.txt` và dùng endpoint `/health` làm health check.

1. Đẩy toàn bộ dự án lên một repository GitHub. Đảm bảo repository có các file `main.py`, `rental_price_model_tro.pkl`, `model_metadata.json`, `rental_room_rows1.csv`, `requirements-render.txt` và `render.yaml`.
2. Đăng nhập Render, chọn **New + → Blueprint** và kết nối repository.
3. Xác nhận cấu hình được đọc từ `render.yaml`, sau đó tạo service và chờ build/deploy hoàn tất.
4. Mở URL Render cấp cho service:
   - `/health` để xác nhận backend sẵn sàng.
   - `/docs` để thử API.
   - `/meta` để xem danh sách category và metric.
5. Gọi `POST /predict` bằng payload mẫu ở phần [API](#api).

Các thiết lập cốt lõi đã khai báo trong Blueprint:

- Python `3.14.2`.
- Build command: `pip install -r requirements-render.txt`.
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`.
- Health check: `/health`.

Nếu deploy thủ công thay vì dùng Blueprint, chọn runtime Python, đặt build/start command như trên và đặt health check path là `/health`. Giữ các file model và metadata ở thư mục làm việc của service. Có thể cấu hình `MODEL_PATH`, `META_PATH`, `CSV_PATH` nếu lưu artifact ở vị trí khác.

`ALLOWED_ORIGINS` mặc định là `*` theo cấu hình hiện tại của backend. Nếu frontend chạy ở domain riêng, hãy đặt biến này thành các origin được phép, phân cách bằng dấu phẩy và không có khoảng trắng, ví dụ `https://example.com,https://www.example.com`.

## Huấn luyện lại model

Trong môi trường đã cài dependencies của project (`uv sync --frozen`):

```powershell
uv run python train.py rental_room_rows1.csv
```

Script ghi đè `rental_price_model_tro.pkl` và `model_metadata.json`, đồng thời tạo các file kết quả huấn luyện/biểu đồ. Sau khi tạo model mới:

1. Kiểm tra metric và phiên bản thư viện trong `model_metadata.json`.
2. Đảm bảo các phiên bản runtime trong `requirements-render.txt` tương thích với phiên bản đã dùng để train, đặc biệt là scikit-learn.
3. Chạy thử API local trước khi cập nhật các artifact lên GitHub và deploy lại Render.

## Đẩy dự án lên GitHub

Tạo repository trên GitHub, sau đó chạy tại thư mục dự án:

```powershell
git init
git add .
git status
git commit -m "Prepare rental price API for Render deployment"
git branch -M main
git remote add origin https://github.com/<username>/<repository>.git
git push -u origin main
```

Thay `<username>` và `<repository>` bằng thông tin repository của bạn. Kiểm tra `git status` trước khi commit; `.venv`, cache Python và file đầu ra sinh ra khi train đã được loại khỏi Git bởi `.gitignore`.

## Lưu ý

- Giữ nguyên tên và vị trí tương đối của các file artifact khi chạy backend với cấu hình mặc định.
- Không commit thông tin bí mật, token hoặc file môi trường chứa secrets.
- Dataset cần được chia sẻ/đẩy lên GitHub chỉ khi bạn có quyền sử dụng và phân phối dữ liệu đó.
- `train.py` là script huấn luyện ngoại tuyến; Render chỉ chạy API, không huấn luyện model trong quá trình khởi động.
