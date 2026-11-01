"""
FastAPI backend for rental price prediction in Da Nang.

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

POST /predict selects the apartment pipeline when `rooms` or
`property_type: "apartment"` is present; otherwise it selects the room pipeline.
"""
import json
import logging
import os
import warnings
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from scipy import sparse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rental-api")

warnings.filterwarnings("ignore", message="Found unknown categories")

BACKGROUND_SIZE = 500
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
FURN_ALLOWED = [
    "Không nội thất",
    "Nội thất cơ bản",
    "Đầy đủ nội thất",
    "Nội thất cao cấp",
    "Không xác định",
]

ROOM_FEATURE_LABELS = {
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

APARTMENT_FEATURE_LABELS = {
    **ROOM_FEATURE_LABELS,
    "rooms": "Số phòng ngủ / Không gian",
}


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
    nearest_beach_code: str = Field(..., description="Mã bãi biển gần nhất (xem GET /meta)")
    nearest_uni_code: str = Field(..., description="Mã trường ĐH gần nhất (xem GET /meta)")

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "nearest_uni_code": "DUT",
            }
        }
    }


class ApartmentInput(BaseModel):
    size_m2: float = Field(..., gt=0, description="Diện tích (m²)")
    rooms: int = Field(1, ge=1, description="Số phòng")
    dist_to_cbd: float = Field(..., ge=0, description="Khoảng cách tới trung tâm (km)")
    dist_to_beach: float = Field(..., ge=0, description="Khoảng cách tới biển (km)")
    dist_to_nearest_uni: float = Field(..., ge=0, description="Khoảng cách tới ĐH gần nhất (km)")
    has_balcony: int = Field(0, ge=0, le=1)
    has_elevator: int = Field(0, ge=0, le=1)
    has_window: int = Field(0, ge=0, le=1)
    furnishing_status: str = Field(..., description=" | ".join(FURN_ALLOWED))
    ward_name: str = Field(..., description="Tên phường/xã (xem GET /meta)")
    nearest_beach_code: str = Field(..., description="Mã bãi biển gần nhất (xem GET /meta)")
    nearest_uni_code: str = Field(..., description="Mã trường ĐH gần nhất (xem GET /meta)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "size_m2": 60,
                "rooms": 2,
                "dist_to_cbd": 2.5,
                "dist_to_beach": 1.0,
                "dist_to_nearest_uni": 1.5,
                "has_balcony": 1,
                "has_elevator": 1,
                "has_window": 1,
                "furnishing_status": "Đầy đủ nội thất",
                "ward_name": "Phước Mỹ",
                "nearest_beach_code": "MyKhe",
                "nearest_uni_code": "DUE",
            }
        }
    }


@dataclass(frozen=True)
class ModelConfig:
    property_type: Literal["room", "apartment"]
    model_path: str
    meta_path: str
    csv_path: str
    category_name: str
    iqr_k: float
    deduplicate_columns: tuple[str, ...]
    feature_labels: dict[str, str]
    excluded_wards: tuple[str, ...] = ()
    rooms_default: int | None = None


@dataclass
class ModelState:
    config: ModelConfig
    model: Any = None
    prep: Any = None
    explainer: Any = None
    meta: dict[str, Any] = field(default_factory=dict)
    raw_features: list[str] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    group_idx: dict[str, list[int]] = field(default_factory=dict)
    cat_cols: list[str] = field(default_factory=list)
    has_background: bool = False
    global_importance: list[dict[str, Any]] = field(default_factory=list)
    explanation_method: str = ""


ROOM_STATE = ModelState(
    config=ModelConfig(
        property_type="room",
        model_path=os.getenv("ROOM_MODEL_PATH", os.getenv("MODEL_PATH", "rental_price_model_tro.pkl")),
        meta_path=os.getenv("ROOM_META_PATH", os.getenv("META_PATH", "model_metadata.json")),
        csv_path=os.getenv("ROOM_CSV_PATH", os.getenv("CSV_PATH", "rental_room_rows1.csv")),
        category_name="Phòng trọ",
        iqr_k=3,
        deduplicate_columns=("latitude", "longitude", "price_vnd", "size_m2", "furnishing_status"),
        feature_labels=ROOM_FEATURE_LABELS,
        excluded_wards=("Hải Vân",),
    )
)

APARTMENT_STATE = ModelState(
    config=ModelConfig(
        property_type="apartment",
        model_path=os.getenv("APARTMENT_MODEL_PATH", "rental_price_model_canho_hgb.pkl"),
        meta_path=os.getenv("APARTMENT_META_PATH", "model_metadata_canho_hgb.json"),
        csv_path=os.getenv("APARTMENT_CSV_PATH", os.getenv("CSV_PATH", "rental_room_rows1.csv")),
        category_name="Căn hộ/Chung cư",
        iqr_k=1.5,
        deduplicate_columns=(
            "latitude",
            "longitude",
            "price_vnd",
            "size_m2",
            "furnishing_status",
            "rooms",
        ),
        feature_labels=APARTMENT_FEATURE_LABELS,
        rooms_default=1,
    )
)


def _dense(values: Any) -> np.ndarray:
    return values.toarray() if sparse.issparse(values) else np.asarray(values)


def _iqr_mask(series: pd.Series, k: float) -> pd.Series:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return series.between(q1 - k * iqr, q3 + k * iqr)


def _load_background(state: ModelState) -> pd.DataFrame | None:
    config = state.config
    path = Path(config.csv_path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df = df[df["category_name"] == config.category_name].copy()
        df["posted_date"] = pd.to_datetime(df["posted_date"], format="mixed")
        df = df[df["posted_date"].dt.year.isin([2026])].copy()
        if config.excluded_wards:
            df = df[~df["ward_name"].isin(config.excluded_wards)]
        df = df.drop_duplicates(list(config.deduplicate_columns))
        if config.rooms_default is not None:
            df["rooms"] = df["rooms"].fillna(config.rooms_default)
        df["log_price"] = np.log(df["price_vnd"])
        df = df[_iqr_mask(df["log_price"], config.iqr_k) & _iqr_mask(df["size_m2"], config.iqr_k)]
        df = df[state.raw_features].dropna()
        if len(df) > BACKGROUND_SIZE:
            df = df.sample(BACKGROUND_SIZE, random_state=42)
        return df.reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Không dựng được background cho %s từ CSV: %s",
            config.property_type,
            exc,
        )
        return None


def load_artifacts(state: ModelState) -> None:
    config = state.config
    state.model = joblib.load(config.model_path)
    with open(config.meta_path, encoding="utf-8") as file:
        state.meta = json.load(file)

    trained_version = state.meta.get("versions", {}).get("sklearn")
    if trained_version and trained_version != sklearn.__version__:
        log.warning(
            "sklearn lúc train=%s, hiện tại=%s -> có thể lỗi/lệch kết quả",
            trained_version,
            sklearn.__version__,
        )

    state.raw_features = state.meta["input_columns"]
    reg = state.model.regressor_
    state.prep = reg.named_steps["prep"]
    estimator = reg.named_steps["model"]
    state.feature_names = list(state.prep.get_feature_names_out())

    try:
        cat_cols = list(state.prep.named_transformers_["ward"].feature_names_in_)
    except Exception:  # noqa: BLE001
        cat_cols = ["ward_name", "nearest_beach_code", "nearest_uni_code"]
    state.cat_cols = cat_cols
    cat_sorted = sorted(cat_cols, key=len, reverse=True)

    groups: dict[str, list[int]] = {}
    for index, name in enumerate(state.feature_names):
        group = next((col for col in cat_sorted if name.startswith(col + "_")), name)
        groups.setdefault(group, []).append(index)
    state.group_idx = groups

    bg_raw = _load_background(state)
    if bg_raw is not None and len(bg_raw) > 0:
        bg = _dense(state.prep.transform(bg_raw))
        state.has_background = True
        log.info(
            "SHAP background %s: %d dòng từ %s",
            config.property_type,
            len(bg),
            config.csv_path,
        )
    else:
        bg = np.zeros((1, len(state.feature_names)))
        state.has_background = False
        log.warning(
            "Không có CSV nền %s -> dùng nền = 0",
            config.property_type,
        )

    if hasattr(estimator, "coef_"):
        state.explainer = shap.LinearExplainer(
            estimator,
            shap.maskers.Independent(bg, max_samples=len(bg)),
        )
        state.explanation_method = "SHAP LinearExplainer (interventional) trên log(giá)"
    else:
        state.explainer = shap.PermutationExplainer(
            estimator.predict,
            shap.maskers.Independent(bg, max_samples=len(bg)),
            seed=42,
        )
        state.explanation_method = "SHAP PermutationExplainer (interventional) trên log(giá)"

    state.global_importance = []
    if state.has_background:
        if hasattr(state.explainer, "expected_value"):
            shap_values = np.asarray(state.explainer.shap_values(bg))
        else:
            importance_sample = bg[:min(len(bg), 100)]
            explanation = state.explainer(
                importance_sample,
                max_evals=2 * len(state.feature_names) + 1,
            )
            shap_values = np.asarray(explanation.values)
        importance = {
            group: float(np.abs(shap_values[:, indices].sum(axis=1)).mean())
            for group, indices in groups.items()
        }
        state.global_importance = [
            {
                "feature": group,
                "label": config.feature_labels.get(group, group),
                "mean_abs_shap_log": value,
            }
            for group, value in sorted(importance.items(), key=lambda item: -item[1])
        ]


@asynccontextmanager
async def lifespan(_: FastAPI):
    for state in (ROOM_STATE, APARTMENT_STATE):
        load_artifacts(state)
        log.info(
            "Đã nạp model %s: %s",
            state.config.property_type,
            state.meta.get("model"),
        )
    yield


app = FastAPI(title="Rental Price API - Phòng trọ và Căn hộ Đà Nẵng", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def explain_row(state: ModelState, row_df: pd.DataFrame, pred_vnd: float) -> dict[str, Any]:
    Xt = _dense(state.prep.transform(row_df))
    if hasattr(state.explainer, "expected_value"):
        shap_values = np.asarray(state.explainer.shap_values(Xt)).reshape(-1)
        base_log = float(np.ravel(state.explainer.expected_value)[0])
    else:
        explanation = state.explainer(
            Xt,
            max_evals=2 * len(state.feature_names) + 1,
        )
        shap_values = np.asarray(explanation.values).reshape(-1)
        base_log = float(np.ravel(explanation.base_values)[0])
    base_vnd = float(np.exp(base_log))

    total_shap = float(shap_values.sum())
    delta_vnd = pred_vnd - base_vnd
    raw = row_df.iloc[0].to_dict()

    contributions = []
    for group, indices in state.group_idx.items():
        value = float(shap_values[indices].sum())
        contributions.append(
            {
                "feature": group,
                "label": state.config.feature_labels.get(group, group),
                "value": raw[group].item() if hasattr(raw[group], "item") else raw[group],
                "shap_log": value,
                "pct_effect": float(np.expm1(value) * 100),
                "vnd_effect": (
                    float(delta_vnd * value / total_shap)
                    if abs(total_shap) > 1e-12
                    else 0.0
                ),
                "direction": "tăng giá" if value > 0 else ("giảm giá" if value < 0 else "trung lập"),
            }
        )
    contributions.sort(key=lambda item: abs(item["shap_log"]), reverse=True)

    prediction_log = float(np.log(pred_vnd))
    return {
        "method": state.explanation_method,
        "baseline_note": (
            f"So với trung bình của tập dữ liệu huấn luyện "
            f"{'căn hộ' if state.config.property_type == 'apartment' else 'phòng trọ'}."
            if state.has_background
            else "Không có CSV nền: biến số so với trung bình huấn luyện, "
            "biến phân loại so với nhóm cơ sở."
        ),
        "base_value_log": base_log,
        "base_value_vnd": base_vnd,
        "prediction_log": prediction_log,
        "additivity_error_log": float(abs(base_log + total_shap - prediction_log)),
        "contributions": contributions,
    }


def _metadata(state: ModelState) -> dict[str, Any]:
    meta = state.meta
    return {
        "model": meta.get("model"),
        "best_params": meta.get("best_params"),
        "input_columns": state.raw_features,
        "feature_labels": state.config.feature_labels,
        "furnishing_values": FURN_ALLOWED,
        "ward_values": meta.get("ward_values", []),
        "nearest_beach_code_values": meta.get("nearest_beach_code_values", []),
        "nearest_uni_code_values": meta.get("nearest_uni_code_values", []),
        "output": meta.get("output"),
        "metrics": {
            "cv_r2_log": meta.get("cv_r2_log"),
            "nested_cv_r2_log": meta.get("nested_cv_r2_log"),
            "holdout_test": meta.get("holdout_test"),
        },
        "n_train_rows": meta.get("n_train_rows"),
        "global_importance": state.global_importance,
    }


def _predict(state: ModelState, payload: BaseModel) -> dict[str, Any]:
    data = payload.model_dump()
    for column in ("furnishing_status", "ward_name", "nearest_beach_code", "nearest_uni_code"):
        data[column] = data[column].strip()

    if data["furnishing_status"] not in FURN_ALLOWED:
        raise HTTPException(
            422,
            detail={
                "error": f"furnishing_status '{data['furnishing_status']}' không hợp lệ",
                "valid": FURN_ALLOWED,
            },
        )

    for column, key in (
        ("ward_name", "ward_values"),
        ("nearest_beach_code", "nearest_beach_code_values"),
        ("nearest_uni_code", "nearest_uni_code_values"),
    ):
        valid = state.meta.get(key)
        if valid and data[column] not in valid:
            raise HTTPException(
                422,
                detail={
                    "error": f"{column} '{data[column]}' không có trong dữ liệu huấn luyện",
                    "valid_values": valid,
                },
            )

    row_df = pd.DataFrame([data])[state.raw_features]
    try:
        pred_vnd = float(state.model.predict(row_df)[0])
        explanation = explain_row(state, row_df, pred_vnd)
    except Exception as exc:  # noqa: BLE001
        log.exception("Lỗi dự đoán %s", state.config.property_type)
        raise HTTPException(500, detail=f"Lỗi khi dự đoán/giải thích: {exc}") from exc

    median_ape = (state.meta.get("holdout_test") or {}).get("median_ape")
    price_range = None
    if median_ape is not None:
        property_label = "căn hộ" if state.config.property_type == "apartment" else "phòng trọ"
        price_range = {
            "low": pred_vnd * (1 - median_ape),
            "high": pred_vnd * (1 + median_ape),
            "note": (
                "Ước lượng thô theo median APE trên tập test "
                f"{property_label}, không phải khoảng tin cậy thống kê"
            ),
        }

    return {
        "input": data,
        "prediction": {
            "price_vnd": pred_vnd,
            "price_million_vnd": round(pred_vnd / 1e6, 2),
            "unit": state.meta.get("output", "VND/tháng"),
            "range_estimate": price_range,
        },
        "explanation": explanation,
        "model": {
            "name": state.meta.get("model"),
            "best_params": state.meta.get("best_params"),
            "holdout_test": state.meta.get("holdout_test"),
        },
    }


def _resolve_property_type(payload: dict[str, Any]) -> Literal["room", "apartment"]:
    requested_type = payload.get("property_type")
    if requested_type is None:
        return "apartment" if "rooms" in payload else "room"
    if not isinstance(requested_type, str):
        raise HTTPException(422, detail="property_type phải là 'room' hoặc 'apartment'")

    normalized = " ".join(requested_type.casefold().split()).replace(" / ", "/")
    if normalized in {"room", "rental_room", "phòng trọ", "phong tro"}:
        return "room"
    if normalized in {
        "apartment",
        "canho",
        "căn hộ",
        "chung cư",
        "căn hộ/chung cư",
        "can ho/chung cu",
    }:
        return "apartment"
    raise HTTPException(422, detail="property_type phải là 'room' hoặc 'apartment'")


def _validate_payload(
    model_type: type[RoomInput] | type[ApartmentInput],
    payload: dict[str, Any],
) -> RoomInput | ApartmentInput:
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        errors = [
            {**error, "loc": ("body", *error["loc"])}
            for error in exc.errors()
        ]
        raise RequestValidationError(errors, body=payload) from exc


@app.get("/health")
def health() -> dict[str, Any]:
    models_loaded = {
        "room": ROOM_STATE.model is not None,
        "apartment": APARTMENT_STATE.model is not None,
    }
    return {
        "status": "ok",
        "model_loaded": all(models_loaded.values()),
        "models_loaded": models_loaded,
    }


@app.get("/meta")
def meta(property_type: Literal["room", "apartment"] = "room") -> dict[str, Any]:
    state = APARTMENT_STATE if property_type == "apartment" else ROOM_STATE
    return _metadata(state)


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    property_type = _resolve_property_type(data)
    data.pop("property_type", None)
    if property_type == "apartment":
        validated = _validate_payload(ApartmentInput, data)
        return _predict(APARTMENT_STATE, validated)
    validated = _validate_payload(RoomInput, data)
    return _predict(ROOM_STATE, validated)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
