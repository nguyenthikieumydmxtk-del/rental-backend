"""
Linear Regression dự đoán giá thuê - CHỈ PHÒNG TRỌ 2025-2026 (Đà Nẵng)
Tối ưu tham số (Ridge / Lasso / ElasticNet vs LinearRegression) + xuất file PKL.

Chạy:  python rental_price_lr_tro_final.py [đường_dẫn_csv]
Đầu ra: rental_price_model_tro.pkl, model_metadata.json,
        cv_results_tro.csv, lr_tro_final_results.png
"""
import json
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import warnings

# Nhóm hiếm/chưa gặp được gộp vào "infrequent" theo chủ đích -> không cần cảnh báo
warnings.filterwarnings("ignore", message="Found unknown categories")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, KFold, RepeatedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "rental_room_rows1.csv"
RANDOM_STATE = 42
MODEL_PATH = "rental_price_model_tro.pkl"

# ----------------------------------------------------------------------------
# 1. LỌC & LÀM SẠCH  (giữ nguyên logic của bạn)
# ----------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

df = df[df["category_name"] == "Phòng trọ"].copy()
n_tro = len(df)

# posted_date có 2 định dạng -> format="mixed" để không bị NaT
df["posted_date"] = pd.to_datetime(df["posted_date"], format="mixed")
df = df[df["posted_date"].dt.year.isin([2025,2026])].copy()
n_year = len(df)
df = df[df["ward_name"] != "Hải Vân"]

df = df.drop_duplicates(
    ["latitude", "longitude", "price_vnd", "size_m2", "furnishing_status"]
)
n_dedup = len(df)

df["furnishing_status"] = df["furnishing_status"].replace(
    "Không xác định", "Không nội thất"
)

df["log_price"] = np.log(df["price_vnd"])


def iqr_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return s.between(q1 - k * iqr, q3 + 1 * iqr)


df = df[iqr_mask(df["log_price"]) & iqr_mask(df["size_m2"])].reset_index(drop=True)
print(f"Phòng trọ: {n_tro} -> năm 2025-2026: {n_year} -> bỏ trùng {n_dedup} "
      f"-> bỏ outlier IQR {len(df)}")
print(f"Giá còn lại: {df.price_vnd.min():,.0f} - {df.price_vnd.max():,.0f} VND | "
      f"Diện tích: {df.size_m2.min():.0f} - {df.size_m2.max():.0f} m2")

# ----------------------------------------------------------------------------
# 2. PIPELINE TIỀN XỬ LÝ (nhận DỮ LIỆU THÔ -> file PKL dùng trực tiếp được)
#    - size_m2            -> log -> chuẩn hoá          (= log_size)
#    - furnishing_status  -> "Không xác định" => "Không nội thất" -> label encode 0-3
#    - ward_name, nearest_beach_code, nearest_uni_code -> one-hot (drop first)
#      (nhóm hiếm < MIN_FREQ mẫu được gộp vào "infrequent" để tránh quá khớp)
#    - dist_*, has_*      -> chuẩn hoá (cần thiết cho Ridge/Lasso/ElasticNet)
# ----------------------------------------------------------------------------
FURN_ORDER = ["Không nội thất", "Nội thất cơ bản", "Đầy đủ nội thất", "Nội thất cao cấp"]
LOG_COLS = ["size_m2"]
NUM_COLS = ["dist_to_cbd", "dist_to_beach", "dist_to_nearest_uni",
            "has_balcony", "has_elevator", "has_window"]
FURN_COL = ["furnishing_status"]
CAT_COLS = ["ward_name", "nearest_beach_code", "nearest_uni_code"]
# Các bãi biển Đà Nẵng và lân cận dùng để tính khoảng cách tới bãi gần nhất.
# BEACHES = {
#     # Quanh bán đảo Sơn Trà
#     "TienSa": (16.1264, 108.2435),
#     "BaiBac": (16.1221, 108.2825),
#     "BaiBut_BaiRang": (16.1018, 108.2762),
#     # Tuyến biển du lịch phía Đông
#     "PhamVanDong": (16.0746, 108.2482),
#     "MyKhe": (16.0652, 108.2471),
#     "BacMyAn": (16.0398, 108.2523),
#     "SonThuy": (16.01627344969784, 108.26275388058494),
#     "NonNuoc": (15.995852110443508, 108.27384535559464),
#     "TanTra": (15.98927595939733, 108.27665363751355),
#     # Tuyến biển vịnh Đà Nẵng
#     "ThanhBinh": (16.07340142742003, 108.20052576741917),
#     "XuanThieu": (16.1042, 108.1631),
#     "NamO": (16.1241, 108.1223),
#     "LangVan": (16.1648, 108.1252),
# }
#
# # Sân bay Quốc tế Đà Nẵng (Nhà ga hành khách)
# AIRPORT_COORDS = (16.0439, 108.1994)
#
# # Bounding box Đà Nẵng (min_lat, min_lng, max_lat, max_lng)
# DA_NANG_BBOX = (15.85, 107.85, 16.30, 108.35)
#
# # Các trường Đại học tại Đà Nẵng (Toạ độ cổng chính / khuôn viên thực tế)
# UNIVERSITIES = {
#     # --- Trường Đại học Đà Nẵng & Các trường thành viên ---
#     'DUE': (16.0478, 108.2400),     # ĐH Kinh tế - 71 Ngũ Hành Sơn
#     'DUT': (16.0738, 108.1499),     # ĐH Bách Khoa - 54 Nguyễn Lương Bằng, Hòa Khánh
#     'UTE': (16.0604, 108.2140),     # ĐH Sư Phạm Kỹ Thuật - 48 Cao Thắng, Hải Châu (mã chuẩn UTE, cũ ghi DHN)
#     'UFLS': (16.0319, 108.2198),    # ĐH Ngoại Ngữ - 131 Lương Nhữ Hộc, Cẩm Lệ (mã chuẩn UFLS)
#     'UED': (16.0612, 108.1583),     # ĐH Sư Phạm - 459 Tôn Đức Thắng, Liên Chiểu (mã chuẩn UED)
#     'VKU': (15.9753, 108.2524),     # ĐH CNTT & Truyền thông Việt - Hàn - Làng ĐH Hòa Quý (mã chuẩn VKU, cũ ghi DUS)
#     'VNUK': (16.0776, 108.2201),    # Viện Nghiên cứu & Đào tạo Việt - Anh - 158A Lê Lợi, Hải Châu
#     'SMP': (15.9752, 108.2523),     # Khoa Y Dược - ĐH Đà Nẵng - Làng ĐH Hòa Quý
#
#     # --- Các Trường Đại học Tư thục & Quốc tế ---
#     'FPTU': (15.9688, 108.2605),    # ĐH FPT Đà Nẵng - Khu đô thị FPT City, Ngũ Hành Sơn
#     'DTU_NVL': (16.0602, 108.2098), # ĐH Duy Tân - Cơ sở 254 Nguyễn Văn Linh
#     'DTU_HK': (16.0822, 108.1472),  # ĐH Duy Tân - Cơ sở Hòa Khánh (03 Nam Cao)
#     'DAU': (16.0327, 108.2206),     # ĐH Kiến Trúc Đà Nẵng - 566 Núi Thành, Hải Châu
#     'UDA': (16.0322, 108.2189),     # ĐH Đông Á - 339 Nguyễn Hữu Thọ, Hải Châu
#     'UPU': (15.9741, 108.2561),     # ĐH Phan Châu Trinh - KĐT Nam Đà Nẵng
#     'UPES3': (16.0682, 108.1831),   # ĐH Thể dục Thể thao Đà Nẵng - 44 Dũng Sĩ Thanh Khê
# }

MIN_FREQ = 10   # nhóm có < 10 mẫu (vd. FPTU, LangVan, NamO...) gộp thành "infrequent"
RAW_FEATURES = LOG_COLS + NUM_COLS + FURN_COL + CAT_COLS

# pd.DataFrame.replace là hàm của pandas nên pickle được (không cần module riêng)
fix_furn = FunctionTransformer(
    pd.DataFrame.replace,
    kw_args={"to_replace": {"Không xác định": "Không nội thất"}},
    feature_names_out="one-to-one",
)


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("log_size", Pipeline([
                ("log", FunctionTransformer(np.log, feature_names_out="one-to-one")),
                ("sc", StandardScaler())]), LOG_COLS),
            ("num", StandardScaler(), NUM_COLS),
            ("furn", Pipeline([
                ("fix", fix_furn),
                ("ord", OrdinalEncoder(categories=[FURN_ORDER])),
                ("sc", StandardScaler())]), FURN_COL),
            ("ward", OneHotEncoder(drop="first", handle_unknown="infrequent_if_exist",
                                 min_frequency=MIN_FREQ), CAT_COLS),
        ],
        verbose_feature_names_out=False,
    )


pipe = Pipeline([("prep", make_preprocessor()), ("model", LinearRegression())])

X, y = df[RAW_FEATURES], df["log_price"]   # tối ưu trên thang log (như các bản trước)

# ----------------------------------------------------------------------------
# 3. TỐI ƯU THAM SỐ
#    LinearRegression thuần KHÔNG có tham số -> thêm các biến thể có regularization.
# ----------------------------------------------------------------------------
param_grid = [
    {"model": [LinearRegression()]},
    {"model": [Ridge()], "model__alpha": np.logspace(-3, 3, 25)},
    {"model": [Lasso(max_iter=50_000)], "model__alpha": np.logspace(-5, -1, 20)},
    {"model": [ElasticNet(max_iter=50_000)],
     "model__alpha": np.logspace(-5, -1, 12),
     "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]},
]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

inner_cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
search = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="r2", n_jobs=-1, refit=True)
search.fit(X_tr, y_tr)

res = pd.DataFrame(search.cv_results_)
res["model_name"] = res["param_model"].apply(lambda m: type(m).__name__)
res = res.sort_values("rank_test_score")
res[["model_name", "param_model__alpha", "param_model__l1_ratio",
     "mean_test_score", "std_test_score", "rank_test_score"]].to_csv(
    "cv_results_tro.csv", index=False)

print("\n[TỐI ƯU THAM SỐ] Điểm CV tốt nhất theo từng họ mô hình (R2 log giá):")
best_by_family = res.groupby("model_name").first()[
    ["mean_test_score", "std_test_score", "param_model__alpha", "param_model__l1_ratio"]
].sort_values("mean_test_score", ascending=False)
print(best_by_family.round(4).to_string())

best_model = search.best_estimator_.named_steps["model"]
best_params = {k.replace("model__", ""): float(v) for k, v in search.best_params_.items()
               if k != "model"}
print(f"\n=> Bộ tối ưu: {type(best_model).__name__} {best_params}")
print(f"   CV R2 = {search.best_score_:.4f}")

# Nested CV: ước lượng không thiên lệch cho cả quy trình 'tìm tham số + huấn luyện'
outer = KFold(5, shuffle=True, random_state=RANDOM_STATE)
nested = cross_val_score(
    GridSearchCV(pipe, param_grid, cv=KFold(5, shuffle=True, random_state=1),
                 scoring="r2", n_jobs=-1),
    X, y, cv=outer, scoring="r2")
plain = cross_val_score(pipe, X, y, cv=outer, scoring="r2")
print(f"\n[Nested CV 5x5]   Có tối ưu tham số : R2 = {nested.mean():.4f} ± {nested.std():.4f}")
print(f"[CV 5-fold]       LinearRegression thuần: R2 = {plain.mean():.4f} ± {plain.std():.4f}")

# ----------------------------------------------------------------------------
# 4. ĐÁNH GIÁ TRÊN TẬP TEST (mô hình chỉ học từ 80% train)
# ----------------------------------------------------------------------------
def report(name, est):
    p_log = est.predict(X_te)
    p, t = np.exp(p_log), np.exp(y_te.values)
    ape = np.abs(p - t) / t
    print(f"\n[HOLD-OUT TEST 20%] {name}")
    print(f"  R2 (log giá)      : {r2_score(y_te, p_log):.4f}")
    print(f"  RMSE (log giá)    : {np.sqrt(mean_squared_error(y_te, p_log)):.4f}")
    print(f"  R2 (VND)          : {r2_score(t, p):.4f}")
    print(f"  MAE (VND)         : {mean_absolute_error(t, p):,.0f}")
    print(f"  RMSE (VND)        : {np.sqrt(mean_squared_error(t, p)):,.0f}")
    print(f"  MAPE              : {ape.mean()*100:.1f}%")
    print(f"  Median APE        : {np.median(ape)*100:.1f}%")
    print(f"  % dự đoán sai <=20%: {(ape <= 0.20).mean()*100:.1f}%")
    return dict(r2_log=r2_score(y_te, p_log), mae_vnd=mean_absolute_error(t, p),
                median_ape=float(np.median(ape)), mape=float(ape.mean()),
                within20=float((ape <= 0.2).mean())), p, t, p_log


plain_fit = clone(pipe).fit(X_tr, y_tr)
report("LinearRegression thuần", plain_fit)
metrics, pred, true, pred_log = report(f"Mô hình tối ưu ({type(best_model).__name__})",
                                       search.best_estimator_)

# ----------------------------------------------------------------------------
# 5. HUẤN LUYỆN LẠI TRÊN TOÀN BỘ DỮ LIỆU & XUẤT PKL
#    TransformedTargetRegressor: nhập dữ liệu thô -> trả GIÁ THUÊ (VND)
# ----------------------------------------------------------------------------
final = TransformedTargetRegressor(
    regressor=clone(search.best_estimator_),
    func=np.log, inverse_func=np.exp,
)
final.fit(df[RAW_FEATURES], df["price_vnd"])

# Kiểm tra: PKL trả đúng exp(dự đoán log)
chk = clone(search.best_estimator_).fit(X, y)
assert np.allclose(final.predict(X), np.exp(chk.predict(X))), "Pipeline PKL lệch kết quả"

joblib.dump(final, MODEL_PATH)
size_kb = __import__("os").path.getsize(MODEL_PATH) / 1024
print(f"\nĐã lưu model: {MODEL_PATH} ({size_kb:.1f} KB)")

# Hệ số (thang log, biến số đã chuẩn hoá)
reg = final.regressor_
names = reg.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(reg.named_steps["model"].coef_, index=names)
print("\n[Hệ số] (biến số đã chuẩn hoá: ảnh hưởng của +1 độ lệch chuẩn | ward_*: so với An Hải)")
print(pd.DataFrame({"coef": coefs, "%_gia_thay_doi": (np.exp(coefs) - 1) * 100})
      .sort_values("coef", key=abs, ascending=False).round(3).to_string())

meta = {
    "model": type(best_model).__name__,
    "best_params": best_params,
    "cv_r2_log": float(search.best_score_),
    "nested_cv_r2_log": float(nested.mean()),
    "holdout_test": metrics,
    "n_train_rows": int(len(df)),
    "input_columns": RAW_FEATURES,
    "furnishing_values": FURN_ORDER + ["Không xác định (tự đổi thành Không nội thất)"],
    "ward_values": sorted(df["ward_name"].unique().tolist()),
    "nearest_beach_code_values": sorted(df["nearest_beach_code"].unique().tolist()),
    "nearest_uni_code_values": sorted(df["nearest_uni_code"].unique().tolist()),
    "output": "Giá thuê dự đoán (VND/tháng)",
    "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__,
                 "numpy": np.__version__, "joblib": joblib.__version__},
}
with open("model_metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 6. KIỂM TRA TẢI LẠI PKL & DỰ ĐOÁN MẪU
# ----------------------------------------------------------------------------
loaded = joblib.load(MODEL_PATH)
sample = pd.DataFrame([{
    "size_m2": 25, "dist_to_cbd": 4.0, "dist_to_beach": 2.0, "dist_to_nearest_uni": 1.5,
    "has_balcony": 1, "has_elevator": 0, "has_window": 0,
    "furnishing_status": "Đầy đủ nội thất", "ward_name": "Hòa Khánh",
    "nearest_beach_code": "XuanThieu", "nearest_uni_code": "DUT",
}])
print(f"\n[Dự đoán mẫu] phòng 25m2, đầy đủ nội thất, Hòa Khánh, có ban công "
      f"-> {loaded.predict(sample)[0]:,.0f} VND/tháng")

# ----------------------------------------------------------------------------
# 7. BIỂU ĐỒ
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(true / 1e6, pred / 1e6, s=10, alpha=0.4)
lim = [0, max(true.max(), pred.max()) / 1e6]
ax[0].plot(lim, lim, "r--")
ax[0].set(xlabel="Giá thực (triệu VND)", ylabel="Giá dự đoán (triệu VND)",
          title="Thực tế vs Dự đoán (test) - Phòng trọ")
ax[1].scatter(pred_log, y_te.values - pred_log, s=10, alpha=0.4)
ax[1].axhline(0, color="r", ls="--")
ax[1].set(xlabel="log(giá) dự đoán", ylabel="Residual", title="Residual plot")
plt.tight_layout()
plt.savefig("lr_tro_final_results.png", dpi=120)
print("Đã lưu biểu đồ: lr_tro_final_results.png")