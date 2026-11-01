"""
HistGradientBoostingRegressor dự đoán giá thuê - CĂN HỘ / CHUNG CƯ (Đà Nẵng)
Tối ưu siêu tham số (RandomizedSearchCV) + nested CV + permutation importance + xuất PKL.

Tiền xử lý dành riêng cho mô hình cây:
  - KHÔNG log / KHÔNG chuẩn hoá (cây bất biến với phép biến đổi đơn điệu)
  - Biến phân loại (nội thất, phường, bãi biển, trường ĐH) -> mã số nguyên + khai báo
    categorical_features để HGB tự tìm cách chia nhóm tối ưu (không one-hot)
  - Giữ gộp nhóm hiếm (MIN_FREQ); giá trị lạ -> NaN (HGB coi như thiếu)

Yêu cầu: scikit-learn >= 1.4
Chạy:  python train_canho_hgb.py [đường_dẫn_csv]
Đầu ra (nếu SAVE_MODEL=True): rental_price_model_canho_hgb.pkl, model_metadata_canho_hgb.json
"""
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform

warnings.filterwarnings("ignore", message="Found unknown categories")

from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (KFold, RandomizedSearchCV, RepeatedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "rental_room_rows1.csv"
RANDOM_STATE = 42
MODEL_PATH = "rental_price_model_canho_hgb.pkl"
META_PATH = "model_metadata_canho_hgb.json"

YEARS = [2026]
ROOMS_DEFAULT = 1          # giá trị điền khi thiếu 'rooms' (backend phải dùng cùng giá trị)
EXCLUDE_WARDS = []
IQR_K = 1.5
MIN_FREQ = 5               # nhóm có < 5 mẫu gộp thành "infrequent"

N_ITER_SEARCH = 80         # số tổ hợp siêu tham số thử ngẫu nhiên
RUN_NESTED_CV = True       # nested CV tốn thời gian hơn; đặt False để bỏ qua
N_ITER_NESTED = 25         # số tổ hợp cho mỗi vòng trong của nested CV
SAVE_MODEL = True          # False -> chỉ in console, không ghi file

# ----------------------------------------------------------------------------
# 1. LỌC & LÀM SẠCH  (giữ nguyên logic bản trước)
# ----------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

df = df[df["category_name"] == "Căn hộ/Chung cư"].copy()
n_raw = len(df)

df["posted_date"] = pd.to_datetime(df["posted_date"], format="mixed")
df = df[df["posted_date"].dt.year.isin(YEARS)].copy()
n_year = len(df)

if EXCLUDE_WARDS:
    df = df[~df["ward_name"].isin(EXCLUDE_WARDS)]

df = df.drop_duplicates(
    ["latitude", "longitude", "price_vnd", "size_m2", "furnishing_status", "rooms"]
)
n_dedup = len(df)

df["rooms"] = df["rooms"].fillna(ROOMS_DEFAULT)
df["log_price"] = np.log(df["price_vnd"])


def iqr_mask(s: pd.Series, k: float = IQR_K) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return s.between(q1 - k * iqr, q3 + k * iqr)


# Lưu ý: ngưỡng IQR tính trên toàn bộ dữ liệu trước khi chia train/test (rò rỉ rất nhẹ)
df = df[iqr_mask(df["log_price"]) & iqr_mask(df["size_m2"])].reset_index(drop=True)
print(f"Căn hộ/Chung cư: {n_raw} -> năm {YEARS}: {n_year} -> bỏ trùng {n_dedup} "
      f"-> bỏ outlier IQR {len(df)}")
print(f"  Giá      : {df.price_vnd.min():,.0f} - {df.price_vnd.max():,.0f} VND")
print(f"  Diện tích: {df.size_m2.min():.0f} - {df.size_m2.max():.0f} m2")
print(f"  Số phòng : {df.rooms.min():.0f} - {df.rooms.max():.0f}")

# ----------------------------------------------------------------------------
# 2. TIỀN XỬ LÝ CHO HistGradientBoosting
#    Cột đầu ra theo thứ tự: [số...] [nội thất] [ward, beach, uni]
#    -> CAT_MASK đánh dấu các cột phân loại (HGB sẽ xử lý native, không cần one-hot)
# ----------------------------------------------------------------------------
FURN_ORDER = ["Không nội thất", "Nội thất cơ bản", "Đầy đủ nội thất",
              "Nội thất cao cấp", "Không xác định"]
NUM_COLS = ["size_m2", "rooms", "dist_to_cbd", "dist_to_beach", "dist_to_nearest_uni",
            "has_balcony", "has_elevator", "has_window"]
FURN_COL = ["furnishing_status"]
CAT_COLS = ["ward_name", "nearest_beach_code", "nearest_uni_code"]
RAW_FEATURES = NUM_COLS + FURN_COL + CAT_COLS      # giống thứ tự input của các bản trước
CAT_MASK = [False] * len(NUM_COLS) + [True] * (len(FURN_COL) + len(CAT_COLS))


def make_hgb_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", "passthrough", NUM_COLS),
            # "Không xác định" là một nhóm bình thường: HGB không hiểu mã số này là thứ tự
            ("furn", OrdinalEncoder(categories=[FURN_ORDER], handle_unknown="use_encoded_value",
                                    unknown_value=np.nan), FURN_COL),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                   min_frequency=MIN_FREQ), CAT_COLS),
        ],
        verbose_feature_names_out=False,
    )


def make_hgb_pipe(**hgb_kwargs) -> Pipeline:
    return Pipeline([
        ("prep", make_hgb_preprocessor()),
        ("model", HistGradientBoostingRegressor(
            categorical_features=CAT_MASK, random_state=RANDOM_STATE, **hgb_kwargs)),
    ])


def make_linear_baseline() -> Pipeline:
    """Mô hình tuyến tính tham chiếu (tiền xử lý one-hot + scale như các bản trước)."""
    prep = ColumnTransformer(
        [
            ("log_size", Pipeline([
                ("log", FunctionTransformer(np.log, feature_names_out="one-to-one")),
                ("sc", StandardScaler())]), ["size_m2"]),
            ("num", StandardScaler(), NUM_COLS[1:]),
            ("furn", OneHotEncoder(categories=[FURN_ORDER], drop="first"), FURN_COL),
            ("cat", OneHotEncoder(drop="first", handle_unknown="infrequent_if_exist",
                                  min_frequency=MIN_FREQ), CAT_COLS),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("prep", prep), ("model", RidgeCV(alphas=np.logspace(-3, 3, 25)))])


X, y = df[RAW_FEATURES], df["log_price"]      # tối ưu trên thang log giá như các bản trước
# Cùng random_state/test_size với bản LazyPredict -> tập test giống hệt, so sánh được
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# ----------------------------------------------------------------------------
# 3. TỐI ƯU SIÊU THAM SỐ (RandomizedSearchCV, RepeatedKFold 5x2)
#    Dữ liệu nhỏ -> ưu tiên mô hình đơn giản: lá ít, min_samples_leaf lớn, l2 mạnh
# ----------------------------------------------------------------------------
param_dist = {
    "model__learning_rate": loguniform(0.01, 0.3),
    "model__max_iter": [100, 200, 300, 500, 800],
    "model__max_leaf_nodes": [4, 6, 8, 12, 16, 24, 31],
    "model__max_depth": [None, 3, 4, 6, 8],
    "model__min_samples_leaf": [5, 10, 20, 30, 50],
    "model__l2_regularization": loguniform(1e-3, 10),
    "model__max_features": [0.5, 0.7, 0.85, 1.0],
}


def make_search(n_iter: int, cv, seed: int = RANDOM_STATE) -> RandomizedSearchCV:
    return RandomizedSearchCV(make_hgb_pipe(), param_dist, n_iter=n_iter, cv=cv,
                              scoring="r2", n_jobs=-1, random_state=seed, refit=True)


def _py(v):
    return v.item() if hasattr(v, "item") else v


print(f"\nĐang tìm siêu tham số: {N_ITER_SEARCH} tổ hợp x 10 fold (có thể mất vài phút)...")
inner_cv = RepeatedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)
search = make_search(N_ITER_SEARCH, inner_cv)
search.fit(X_tr, y_tr)

res = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
top = res[["mean_test_score", "std_test_score"] +
          [c for c in res.columns if c.startswith("param_")]].head(8)
top.columns = [c.replace("param_model__", "") for c in top.columns]

print("\n=============== [1] TOP 8 TỔ HỢP SIÊU THAM SỐ (CV, R2 log giá) ===============")
with pd.option_context("display.width", 250, "display.max_columns", None):
    print(top.round(4).to_string(index=False))

best_params = {k.replace("model__", ""): _py(v) for k, v in search.best_params_.items()}
best_est = search.best_estimator_
print(f"\n=> Tham số tối ưu: {best_params}")
print(f"   CV R2 = {search.best_score_:.4f} (số vòng boosting thực tế: "
      f"{best_est.named_steps['model'].n_iter_})")

nested = None
if RUN_NESTED_CV:
    print(f"\nĐang chạy nested CV 5 vòng ngoài ({N_ITER_NESTED} tổ hợp/vòng trong)...")
    outer = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    nested = cross_val_score(
        make_search(N_ITER_NESTED, KFold(5, shuffle=True, random_state=1)),
        X, y, cv=outer, scoring="r2")
    print(f"[Nested CV 5x5]  R2 = {nested.mean():.4f} ± {nested.std():.4f}  "
          f"(ước lượng không thiên lệch cho cả quy trình tối ưu)")

# ----------------------------------------------------------------------------
# 4. ĐÁNH GIÁ TRÊN TẬP HOLD-OUT TEST (20%)
# ----------------------------------------------------------------------------
def evaluate(est):
    p_log = est.predict(X_te)
    p, t = np.exp(p_log), np.exp(y_te.values)
    ape = np.abs(p - t) / t
    return dict(r2_log=float(r2_score(y_te, p_log)),
                rmse_log=float(np.sqrt(mean_squared_error(y_te, p_log))),
                r2_vnd=float(r2_score(t, p)), mae_vnd=float(mean_absolute_error(t, p)),
                rmse_vnd=float(np.sqrt(mean_squared_error(t, p))),
                mape=float(ape.mean()), median_ape=float(np.median(ape)),
                within20=float((ape <= 0.2).mean())), p_log


candidates = {
    "Ridge (tham chiếu tuyến tính)": clone(make_linear_baseline()).fit(X_tr, y_tr),
    "HGB mặc định": make_hgb_pipe().fit(X_tr, y_tr),
    "HGB tối ưu": best_est,
}
rows = {}
for name, est in candidates.items():
    m, p_log = evaluate(est)
    rows[name] = m
    if name == "HGB tối ưu":
        metrics, pred_log = m, p_log

print("\n=============== [2] HOLD-OUT TEST 20% ===============")
summary = pd.DataFrame(rows).T
summary["mape"] *= 100
summary["median_ape"] *= 100
summary["within20"] *= 100
summary = summary.rename(columns={"mape": "MAPE_%", "median_ape": "MedAPE_%", "within20": "<=20%_%"})
with pd.option_context("display.width", 250, "display.max_columns", None,
                       "display.float_format", "{:,.4f}".format):
    print(summary.to_string())

m = metrics
print(f"\n--- Chi tiết HGB tối ưu ---")
print(f"  R2 (log giá)      : {m['r2_log']:.4f}")
print(f"  RMSE (log giá)    : {m['rmse_log']:.4f}")
print(f"  R2 (VND)          : {m['r2_vnd']:.4f}")
print(f"  MAE (VND)         : {m['mae_vnd']:,.0f}")
print(f"  RMSE (VND)        : {m['rmse_vnd']:,.0f}")
print(f"  MAPE              : {m['mape']*100:.1f}%")
print(f"  Median APE        : {m['median_ape']*100:.1f}%")
print(f"  % dự đoán sai <=20%: {m['within20']*100:.1f}%")

# ----------------------------------------------------------------------------
# 5. MỨC ĐỘ QUAN TRỌNG CỦA TỪNG BIẾN (permutation importance trên tập test)
#    Xáo trộn từng cột THÔ -> tự gộp luôn các biến phân loại (không cần one-hot)
# ----------------------------------------------------------------------------
pi = permutation_importance(best_est, X_te, y_te, scoring="r2", n_repeats=20,
                            random_state=RANDOM_STATE, n_jobs=-1)
imp = pd.DataFrame({"R2_giam_khi_xao_tron": pi.importances_mean,
                    "std": pi.importances_std}, index=RAW_FEATURES
                   ).sort_values("R2_giam_khi_xao_tron", ascending=False)
print("\n=============== [3] ĐỘ QUAN TRỌNG BIẾN (permutation, tập test) ===============")
print(imp.round(4).to_string())

# ----------------------------------------------------------------------------
# 6. HUẤN LUYỆN LẠI TRÊN TOÀN BỘ DỮ LIỆU & XUẤT PKL
#    TransformedTargetRegressor: nhập dữ liệu thô -> trả GIÁ THUÊ (VND)
#    (clone để KHÔNG ghi đè search.best_estimator_)
# ----------------------------------------------------------------------------
final = TransformedTargetRegressor(
    regressor=clone(best_est), func=np.log, inverse_func=np.exp)
final.fit(df[RAW_FEATURES], df["price_vnd"])

chk = clone(best_est).fit(X, y)
assert np.allclose(final.predict(X), np.exp(chk.predict(X))), "Pipeline PKL lệch kết quả"

# ----------------------------------------------------------------------------
# 7. DỰ ĐOÁN MẪU (giá trị phổ biến nhất trong dữ liệu -> chắc chắn là nhóm đã biết)
# ----------------------------------------------------------------------------
sample = pd.DataFrame([{
    "size_m2": 60, "rooms": 2, "dist_to_cbd": 2.5, "dist_to_beach": 1.0,
    "dist_to_nearest_uni": 1.5, "has_balcony": 1, "has_elevator": 1, "has_window": 1,
    "furnishing_status": "Đầy đủ nội thất",
    "ward_name": df["ward_name"].mode()[0],
    "nearest_beach_code": df["nearest_beach_code"].mode()[0],
    "nearest_uni_code": df["nearest_uni_code"].mode()[0],
}])

enc = final.regressor_.named_steps["prep"].named_transformers_["cat"]
for col, cats in zip(CAT_COLS, enc.categories_):
    if sample.loc[0, col] not in cats:
        print(f"[CẢNH BÁO] {col}='{sample.loc[0, col]}' không có trong dữ liệu huấn luyện "
              f"-> bị coi là giá trị thiếu")

s = sample.iloc[0]
print("\n=============== [4] DỰ ĐOÁN MẪU ===============")
print(f"Căn hộ {s.size_m2:.0f}m2, {s.rooms:.0f} phòng, {s.ward_name}, {s.furnishing_status} "
      f"-> {final.predict(sample)[0]:,.0f} VND/tháng")

# ----------------------------------------------------------------------------
# 8. XUẤT PKL + METADATA (tắt bằng SAVE_MODEL=False)
# ----------------------------------------------------------------------------
if SAVE_MODEL:
    joblib.dump(final, MODEL_PATH)
    meta = {
        "model": "HistGradientBoostingRegressor",
        "category": "Căn hộ/Chung cư",
        "best_params": best_params,
        "cv_r2_log": float(search.best_score_),
        "nested_cv_r2_log": float(nested.mean()) if nested is not None else None,
        "holdout_test": metrics,
        "n_train_rows": int(len(df)),
        "input_columns": RAW_FEATURES,
        "rooms_default": ROOMS_DEFAULT,
        "furnishing_values": FURN_ORDER,
        "ward_values": sorted(df["ward_name"].unique().tolist()),
        "nearest_beach_code_values": sorted(df["nearest_beach_code"].unique().tolist()),
        "nearest_uni_code_values": sorted(df["nearest_uni_code"].unique().tolist()),
        "permutation_importance": {k: float(v) for k, v in imp["R2_giam_khi_xao_tron"].items()},
        "output": "Giá thuê dự đoán (VND/tháng)",
        "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__,
                     "numpy": np.__version__, "joblib": joblib.__version__},
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu model: {MODEL_PATH} ({os.path.getsize(MODEL_PATH) / 1024:.1f} KB) "
          f"| metadata: {META_PATH}")