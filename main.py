"""
Backend FastAPI: dự đoán giá thuê phòng trọ (Đà Nẵng) + giải thích bằng SHAP.

Chạy:
    pip install fastapi uvicorn shap scikit-learn pandas numpy scipy joblib
    uvicorn main:app --reload --port 8000

Cần có cùng thư mục (hoặc đặt qua biến môi trường):
    rental_price_model_tro.pkl   (MODEL_PATH)
    model_metadata.json          (META_PATH)
    rental_room_rows2.csv        (CSV_PATH, tuỳ chọn - dùng làm "nền" cho SHAP)

Endpoints:
    GET  /health      kiểm tra server
    GET  /meta        danh sách phường, nội thất, metric model (để frontend dựng form)
    POST /predict     dự đoán giá + giải thích SHAP
"""
import json
import logging
import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy import sparse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rental-api")

# Nhóm hiếm/chưa gặp được gộp vào "infrequent" theo chủ đích (giống train.py)
warnings.filterwarnings("ignore", message="Found unknown categories")

MODEL_PATH = os.getenv("MODEL_PATH", "rental_price_model_tro.pkl")
META_PATH = os.getenv("META_PATH", "model_metadata.json")
CSV_PATH = os.getenv("CSV_PATH", "rental_room_rows1.csv")
BACKGROUND_SIZE = 500
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

FURN_ALLOWED = ["Không nội thất", "Nội thất cơ bản", "Đầy đủ nội thất",
                "Nội thất cao cấp", "Không xác định"]

FEATURE_LABELS = {
    "size_m2": "Diện tích (m²)",
    "dist_to_cbd": "Khoảng cách tới trung tâm (km)",
    "dist_to_beach": "Khoảng cách tới biển (km)",
    "dist_to_nearest_uni": "Khoảng cách tới trường ĐH gần nhất (km)",
    "has_balcony": "Ban công",
    "has_elevator": "Thang máy",
    "has_window": "Cửa sổ",
    "furnishing_status": "Nội thất",
    "ward_name": "Phường/Xã",
    "nearest_beach_code": "Bãi biển gần nhất",
    "nearest_uni_code": "Trường ĐH gần nhất",
}


# ----------------------------------------------------------------------------
# Schema request
# ----------------------------------------------------------------------------
class RoomInput(BaseModel):
    size_m2: float = Field(..., gt=0, description="Diện tích (m²)")
    dist_to_cbd: float = Field(..., ge=0, description="Khoảng cách tới trung tâm (km)")
    dist_to_beach: float = Field(..., ge=0, description="Khoảng cách tới biển (km)")
    dist_to_nearest_uni: float = Field(..., ge=0, description="Khoảng cách tới ĐH gần nhất (km)")
    has_balcony: int = Field(0, ge=0, le=1)
    has_elevator: int = Field(0, ge=0, le=1)
    has_window: int = Field(0, ge=0, le=1)
    furnishing_status: str = Field(..., description=" | ".join(FURN_ALLOWED))
    ward_name: str = Field(..., description="Tên phường/xã (xem GET /meta)")
    nearest_beach_code: str = Field(..., description="Mã bãi biển gần nhất, vd XuanThieu (xem GET /meta)")
    nearest_uni_code: str = Field(..., description="Mã trường ĐH gần nhất, vd DUT (xem GET /meta)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "size_m2": 25, "dist_to_cbd": 4.0, "dist_to_beach": 2.0,
                "dist_to_nearest_uni": 1.5, "has_balcony": 1, "has_elevator": 0,
                "has_window": 0, "furnishing_status": "Đầy đủ nội thất",
                "ward_name": "Hòa Khánh",
                "nearest_beach_code": "XuanThieu", "nearest_uni_code": "DUT",
            }
        }
    }


# ----------------------------------------------------------------------------
# Trạng thái toàn cục (nạp 1 lần khi khởi động)
# ----------------------------------------------------------------------------
class State:
    model = None            # TransformedTargetRegressor (nhận dữ liệu thô -> VND)
    prep = None             # ColumnTransformer đã fit
    explainer = None        # shap.LinearExplainer (thang log giá)
    meta: dict = {}
    raw_features: list = []
    feature_names: list = []
    group_idx: dict = {}    # tên biến gốc -> chỉ số cột sau transform
    cat_cols: list = []
    has_background = False
    global_importance: list = []


S = State()


def _dense(a):
    return a.toarray() if sparse.issparse(a) else np.asarray(a)


def _iqr_mask(s: pd.Series, k: float = 3) -> pd.Series:
    # giống hệt train.py
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return s.between(q1 - k * iqr, q3 + k * iqr)


def _load_background(raw_features: list):
    """Dựng lại tập dữ liệu train (cùng logic làm sạch) để làm nền cho SHAP."""
    path = Path(CSV_PATH)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df = df[df["category_name"] == "Phòng trọ"].copy()
        df["posted_date"] = pd.to_datetime(df["posted_date"], format="mixed")
        df = df[df["posted_date"].dt.year.isin([2026])].copy()
        df = df[df["ward_name"] != "Hải Vân"]
        df = df.drop_duplicates(
            ["latitude", "longitude", "price_vnd", "size_m2", "furnishing_status"])
        df["log_price"] = np.log(df["price_vnd"])
        df = df[_iqr_mask(df["log_price"]) & _iqr_mask(df["size_m2"])]
        df = df[raw_features].dropna()
        if len(df) > BACKGROUND_SIZE:
            df = df.sample(BACKGROUND_SIZE, random_state=42)
        return df.reset_index(drop=True)
    except Exception as e:  # noqa: BLE001
        log.warning("Không dựng được background từ CSV: %s", e)
        return None


def load_artifacts():
    S.model = joblib.load(MODEL_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        S.meta = json.load(f)

    # cảnh báo lệch phiên bản sklearn (nguyên nhân phổ biến khi load pkl lỗi)
    trained_ver = S.meta.get("versions", {}).get("sklearn")
    if trained_ver and trained_ver != sklearn.__version__:
        log.warning("sklearn lúc train=%s, hiện tại=%s -> có thể lỗi/lệch kết quả",
                    trained_ver, sklearn.__version__)

    S.raw_features = S.meta["input_columns"]

    # TransformedTargetRegressor -> regressor_ là Pipeline(prep, model) đã fit
    reg = S.model.regressor_
    S.prep = reg.named_steps["prep"]
    linear = reg.named_steps["model"]
    S.feature_names = list(S.prep.get_feature_names_out())

    # gom các cột one-hot (<cột>_<giá trị>, kể cả <cột>_infrequent_sklearn) về biến gốc
    # các cột categorical lấy từ chính OneHotEncoder đã fit
    try:
        cat_cols = list(S.prep.named_transformers_["ward"].feature_names_in_)
    except Exception:  # noqa: BLE001
        cat_cols = ["ward_name", "nearest_beach_code", "nearest_uni_code"]
    S.cat_cols = cat_cols
    cat_sorted = sorted(cat_cols, key=len, reverse=True)

    groups: dict = {}
    for i, name in enumerate(S.feature_names):
        g = next((c for c in cat_sorted if name.startswith(c + "_")), name)
        groups.setdefault(g, []).append(i)
    S.group_idx = groups

    # SHAP cho mô hình tuyến tính: phi_i = coef_i * (x_i - E[x_i]), thang log(giá)
    bg_raw = _load_background(S.raw_features)
    if bg_raw is not None and len(bg_raw) > 0:
        bg = _dense(S.prep.transform(bg_raw))
        S.has_background = True
        log.info("SHAP background: %d dòng từ %s", len(bg), CSV_PATH)
    else:
        bg = np.zeros((1, len(S.feature_names)))
        S.has_background = False
        log.warning("Không có CSV nền -> dùng nền = 0 (biến số đã chuẩn hoá thì đúng, "
                    "biến phân loại so với nhóm cơ sở). Nên đặt CSV_PATH.")

    # max_samples=len(bg): dùng toàn bộ nền, không để SHAP tự cắt xuống 100 mẫu
    S.explainer = shap.LinearExplainer(
        linear, shap.maskers.Independent(bg, max_samples=len(bg)))

    # độ quan trọng toàn cục (mean |SHAP|, theo biến gốc)
    S.global_importance = []
    if S.has_background:
        sv = np.asarray(S.explainer.shap_values(bg))
        imp = {g: float(np.abs(sv[:, idx].sum(axis=1)).mean()) for g, idx in groups.items()}
        S.global_importance = [
            {"feature": g, "label": FEATURE_LABELS.get(g, g), "mean_abs_shap_log": v}
            for g, v in sorted(imp.items(), key=lambda kv: -kv[1])
        ]


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    log.info("Đã nạp model: %s", S.meta.get("model"))
    yield


app = FastAPI(title="Rental Price API - Phòng trọ Đà Nẵng", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------
# Logic giải thích
# ----------------------------------------------------------------------------
def explain_row(row_df: pd.DataFrame, pred_vnd: float) -> dict:
    """
    Giải thích 1 dự đoán. SHAP tính trên log(giá):
        log(giá) = base_log + sum(shap_i)   =>   giá = exp(base_log) * prod(exp(shap_i))
    Mỗi biến có:
      - shap_log      : đóng góp vào log(giá)
      - pct_effect    : % thay đổi giá do biến đó (nhân tính) = exp(shap)-1
      - vnd_effect    : đóng góp quy ra VND, tổng các biến = giá dự đoán - giá nền
    """
    Xt = _dense(S.prep.transform(row_df))
    sv = np.asarray(S.explainer.shap_values(Xt)).reshape(-1)
    base_log = float(np.ravel(S.explainer.expected_value)[0])
    base_vnd = float(np.exp(base_log))

    total_shap = float(sv.sum())
    delta_vnd = pred_vnd - base_vnd
    raw = row_df.iloc[0].to_dict()

    contribs = []
    for g, idx in S.group_idx.items():
        s = float(sv[idx].sum())
        item = {
            "feature": g,
            "label": FEATURE_LABELS.get(g, g),
            "value": raw[g].item() if hasattr(raw[g], "item") else raw[g],
            "shap_log": s,
            "pct_effect": float(np.expm1(s) * 100),
            "vnd_effect": float(delta_vnd * s / total_shap) if abs(total_shap) > 1e-12 else 0.0,
            "direction": "tăng giá" if s > 0 else ("giảm giá" if s < 0 else "trung lập"),
        }
        contribs.append(item)
    contribs.sort(key=lambda c: abs(c["shap_log"]), reverse=True)

    recon_log = base_log + total_shap
    return {
        "method": "SHAP LinearExplainer (interventional) trên log(giá)",
        "baseline_note": (
            "So với trung bình của tập dữ liệu huấn luyện."
            if S.has_background else
            "Không có CSV nền: biến số so với trung bình huấn luyện, biến phân loại so với nhóm cơ sở (drop first)."),
        "base_value_log": base_log,
        "base_value_vnd": base_vnd,
        "prediction_log": float(np.log(pred_vnd)),
        "additivity_error_log": float(abs(recon_log - np.log(pred_vnd))),
        "contributions": contribs,   # đã sắp theo mức ảnh hưởng giảm dần (dùng vẽ waterfall/bar)
    }


# ----------------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": S.model is not None}


@app.get("/meta")
def meta():
    return {
        "model": S.meta.get("model"),
        "best_params": S.meta.get("best_params"),
        "input_columns": S.raw_features,
        "feature_labels": FEATURE_LABELS,
        "furnishing_values": FURN_ALLOWED,
        "ward_values": S.meta.get("ward_values", []),
        "nearest_beach_code_values": S.meta.get("nearest_beach_code_values", []),
        "nearest_uni_code_values": S.meta.get("nearest_uni_code_values", []),
        "output": S.meta.get("output"),
        "metrics": {
            "cv_r2_log": S.meta.get("cv_r2_log"),
            "nested_cv_r2_log": S.meta.get("nested_cv_r2_log"),
            "holdout_test": S.meta.get("holdout_test"),
        },
        "n_train_rows": S.meta.get("n_train_rows"),
        "global_importance": S.global_importance,
    }


@app.post("/predict")
def predict(payload: RoomInput):
    data = payload.model_dump()
    for c in ["furnishing_status", "ward_name", "nearest_beach_code", "nearest_uni_code"]:
        data[c] = data[c].strip()

    if data["furnishing_status"] not in FURN_ALLOWED:
        raise HTTPException(422, detail={
            "error": f"furnishing_status '{data['furnishing_status']}' không hợp lệ",
            "valid": FURN_ALLOWED})

    # kiểm tra các biến phân loại theo metadata (bỏ qua nếu metadata cũ không có khoá)
    for col, key in [("ward_name", "ward_values"),
                     ("nearest_beach_code", "nearest_beach_code_values"),
                     ("nearest_uni_code", "nearest_uni_code_values")]:
        valid = S.meta.get(key)
        if valid and data[col] not in valid:
            raise HTTPException(422, detail={
                "error": f"{col} '{data[col]}' không có trong dữ liệu huấn luyện",
                "valid_values": valid})
    row_df = pd.DataFrame([data])[S.raw_features]   # đúng thứ tự cột lúc train

    try:
        pred_vnd = float(S.model.predict(row_df)[0])  # pipeline: log->scale->encode->model->exp
        explanation = explain_row(row_df, pred_vnd)
    except Exception as e:  # noqa: BLE001
        log.exception("Lỗi dự đoán")
        raise HTTPException(500, detail=f"Lỗi khi dự đoán/giải thích: {e}") from e

    # khoảng ước lượng thô dựa trên sai số tương đối trung vị của tập test
    median_ape = (S.meta.get("holdout_test") or {}).get("median_ape")
    price_range = None
    if median_ape is not None:
        price_range = {"low": pred_vnd * (1 - median_ape),
                       "high": pred_vnd * (1 + median_ape),
                       "note": "Ước lượng thô theo median APE trên tập test, không phải khoảng tin cậy thống kê"}
    print(pred_vnd)
    return {
        "input": data,
        "prediction": {
            "price_vnd": pred_vnd,
            "price_million_vnd": round(pred_vnd / 1e6, 2),
            "unit": S.meta.get("output", "VND/tháng"),
            "range_estimate": price_range,
        },
        "explanation": explanation,
        "model": {
            "name": S.meta.get("model"),
            "best_params": S.meta.get("best_params"),
            "holdout_test": S.meta.get("holdout_test"),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)