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
from collections import Counter

from scipy import stats
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import (GridSearchCV, KFold, RepeatedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "rental_room_rows.csv"
RANDOM_STATE = 42
MODEL_PATH = "rental_price_model_tro.pkl"

# --- Cấu hình nested CV ---
RUN_NESTED_CV = True     # nested CV tốn thời gian; đặt False để bỏ qua
N_OUTER_SPLITS = 5       # số fold vòng ngoài (đánh giá)
N_OUTER_REPEATS = 3      # lặp vòng ngoài với các cách chia khác nhau -> ước lượng ổn định hơn

# ----------------------------------------------------------------------------
# 1. LỌC & LÀM SẠCH  (giữ nguyên logic của bạn)
# ----------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

df = df[df["category_name"] == "Phòng trọ"].copy()
n_tro = len(df)
df = df[df["price_vnd"] <= 7000000]
# posted_date có 2 định dạng -> format="mixed" để không bị NaT
df["posted_date"] = pd.to_datetime(df["posted_date"], format="mixed")
df = df[df["posted_date"].dt.year.isin([2026])].copy()
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
    return s.between(q1 - k * iqr, q3 + k * iqr)


df = df[iqr_mask(df["log_price"]) & iqr_mask(df["size_m2"])].reset_index(drop=True)
print(f"Phòng trọ: {n_tro} -> năm 2025-2026: {n_year} -> bỏ trùng {n_dedup} "
      f"-> bỏ outlier IQR {len(df)}")
print(f"Giá còn lại: {df.price_vnd.min():,.0f} - {df.price_vnd.max():,.0f} VND | "
      f"Diện tích: {df.size_m2.min():.0f} - {df.size_m2.max():.0f} m2 | "
      f"Khoảng cách đến trường đại học: {df.dist_to_nearest_uni.min():.0f} - {df.dist_to_nearest_uni.max():.0f} | ")

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
NUM_COLS = ["dist_to_cbd", "dist_to_beach",
            "has_balcony", "has_elevator", "has_window", "dist_to_nearest_uni"]
FURN_COL = ["furnishing_status"]
CAT_COLS = ["ward_name", "nearest_beach_code", "nearest_uni_code"]

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

# ----------------------------------------------------------------------------
# 3b. NESTED CV - ĐÁNH GIÁ KHÁCH QUAN CẢ QUY TRÌNH "chọn mô hình + tìm tham số + huấn luyện"
#     Vòng trong (KFold 5)  : GridSearchCV chọn họ mô hình & alpha CHỈ trên phần train của fold ngoài
#     Vòng ngoài (Repeated) : chấm điểm trên phần test chưa từng tham gia chọn tham số
#     -> điểm vòng ngoài không bị lạc quan do "chọn tham số tốt nhất trên chính dữ liệu đánh giá"
#     Ba thiết lập chạy trên CÙNG các fold ngoài nên so sánh công bằng (paired).
# ----------------------------------------------------------------------------
def _vnd(fn):
    """Biến metric tính trên VND thành scorer (mô hình dự đoán log giá)."""
    return lambda yt, yp: fn(np.exp(np.asarray(yt)), np.exp(np.asarray(yp)))

_mape = lambda t, p: float(np.mean(np.abs(p - t) / t))
_medape = lambda t, p: float(np.median(np.abs(p - t) / t))
_within20 = lambda t, p: float(np.mean(np.abs(p - t) / t <= 0.20))

SCORING = {
    "r2_log":   "r2",
    "rmse_log": make_scorer(lambda a, b: np.sqrt(mean_squared_error(a, b)), greater_is_better=False),
    "r2_vnd":   make_scorer(_vnd(r2_score)),
    "mae_vnd":  make_scorer(_vnd(mean_absolute_error), greater_is_better=False),
    "mape":     make_scorer(_vnd(_mape), greater_is_better=False),
    "median_ape": make_scorer(_vnd(_medape), greater_is_better=False),
    "within20": make_scorer(_vnd(_within20)),
}
ERROR_METRICS = {"rmse_log", "mae_vnd", "mape", "median_ape"}   # scorer trả giá trị âm -> đổi dấu khi hiển thị

nested = None
nested_summary = None
if RUN_NESTED_CV:
    n_outer_total = N_OUTER_SPLITS * N_OUTER_REPEATS
    print(f"\n[NESTED CV] {N_OUTER_SPLITS}-fold x {N_OUTER_REPEATS} lần lặp = {n_outer_total} vòng ngoài "
          f"(mỗi vòng chạy lại toàn bộ GridSearchCV, có thể mất vài phút)...")
    outer = RepeatedKFold(n_splits=N_OUTER_SPLITS, n_repeats=N_OUTER_REPEATS, random_state=RANDOM_STATE)

    # n_jobs=-1 đặt ở GridSearchCV (vòng trong); vòng ngoài chạy tuần tự để tránh quá tải CPU
    nested_est = GridSearchCV(pipe, param_grid, cv=KFold(5, shuffle=True, random_state=1),
                              scoring="r2", n_jobs=-1)
    cv_nested = cross_validate(nested_est, X, y, cv=outer, scoring=SCORING,
                               return_estimator=True, n_jobs=1)
    cv_plain = cross_validate(clone(pipe), X, y, cv=outer, scoring=SCORING, n_jobs=1)
    cv_base = cross_validate(DummyRegressor(strategy="median"), X, y, cv=outer, scoring=SCORING, n_jobs=1)

    def to_frame(cv):
        d = pd.DataFrame({k: cv[f"test_{k}"] for k in SCORING})
        for k in ERROR_METRICS:
            d[k] = -d[k]
        return d

    f_nested, f_plain, f_base = to_frame(cv_nested), to_frame(cv_plain), to_frame(cv_base)

    # ---- (a) Bảng so sánh trung bình ± độ lệch chuẩn qua các fold ngoài
    def fmt(col, k):
        m, sd = col.mean(), col.std()
        if k in ("mae_vnd",):
            return f"{m:,.0f} ± {sd:,.0f}"
        if k in ("mape", "median_ape", "within20"):
            return f"{m*100:.1f}% ± {sd*100:.1f}%"
        return f"{m:.4f} ± {sd:.4f}"

    cmp_tbl = pd.DataFrame({
        "Median baseline (không học)": {k: fmt(f_base[k], k) for k in SCORING},
        "LinearRegression thuần": {k: fmt(f_plain[k], k) for k in SCORING},
        "Quy trình tối ưu (nested)": {k: fmt(f_nested[k], k) for k in SCORING},
    })
    print("\n[NESTED CV] Điểm vòng ngoài (trung bình ± độ lệch chuẩn giữa các fold):")
    print(cmp_tbl.to_string())

    # ---- (b) Độ lạc quan: điểm CV vòng trong (best_score_) so với điểm vòng ngoài
    inner_best = np.array([e.best_score_ for e in cv_nested["estimator"]])
    gap = inner_best - f_nested["r2_log"].values
    print(f"\n[Độ lạc quan] CV vòng trong (điểm của bộ tham số thắng) : R2 = {inner_best.mean():.4f}")
    print(f"              Điểm vòng ngoài (khách quan)                : R2 = {f_nested['r2_log'].mean():.4f}")
    print(f"              Chênh lệch trung bình                       : {gap.mean():+.4f} "
          f"(lớn = chọn tham số bị lạc quan)")

    # ---- (c) Độ ổn định lựa chọn: mỗi vòng ngoài chọn họ mô hình / alpha nào?
    chosen = []
    for e in cv_nested["estimator"]:
        bp = {k.replace("model__", ""): v for k, v in e.best_params_.items() if k != "model"}
        chosen.append(dict(model=type(e.best_estimator_.named_steps["model"]).__name__, **bp))
    chosen = pd.DataFrame(chosen)
    print("\n[Độ ổn định] Họ mô hình được chọn qua các vòng ngoài:")
    print(chosen["model"].value_counts().to_string())
    if "alpha" in chosen:
        print("Alpha được chọn (trung vị / min - max theo họ mô hình):")
        print(chosen.groupby("model")["alpha"].agg(["median", "min", "max"]).dropna().to_string())

    # ---- (d) Tối ưu tham số có thực sự hơn LinearRegression thuần? (paired, Nadeau-Bengio)
    #      Các fold của RepeatedKFold không độc lập -> hiệu chỉnh phương sai bằng (1/J + n_test/n_train)
    d = f_nested["r2_log"].values - f_plain["r2_log"].values
    n_te = len(X) / N_OUTER_SPLITS
    corr_var = (1 / len(d) + n_te / (len(X) - n_te)) * d.var(ddof=1)
    t_stat = d.mean() / np.sqrt(corr_var) if corr_var > 0 else np.nan
    p_val = 2 * stats.t.sf(abs(t_stat), df=len(d) - 1) if corr_var > 0 else np.nan
    print(f"\n[So sánh paired] R2(tối ưu) - R2(LinearRegression thuần) = {d.mean():+.4f} "
          f"(thắng {(d > 0).sum()}/{len(d)} vòng) | t hiệu chỉnh = {t_stat:.2f}, p = {p_val:.3f}")
    print("  -> p >= 0.05: chưa đủ bằng chứng rằng regularization cải thiện so với LinearRegression thuần.")

    # ---- (e) Lưu chi tiết từng vòng ngoài
    folds_out = f_nested.add_prefix("nested_").join(f_plain.add_prefix("plain_")).join(chosen)
    folds_out["inner_best_r2"] = inner_best
    folds_out.to_csv("nested_cv_folds_tro.csv", index_label="outer_fold")

    nested = f_nested["r2_log"].values
    nested_summary = {
        "n_outer_folds": int(n_outer_total),
        "mean": {k: float(f_nested[k].mean()) for k in SCORING},
        "std": {k: float(f_nested[k].std()) for k in SCORING},
        "plain_linear_mean": {k: float(f_plain[k].mean()) for k in SCORING},
        "median_baseline_mean": {k: float(f_base[k].mean()) for k in SCORING},
        "inner_vs_outer_r2_gap": float(gap.mean()),
        "model_selection_counts": chosen["model"].value_counts().to_dict(),
        "paired_corrected_t": {"diff_r2": float(d.mean()), "t": float(t_stat), "p": float(p_val)},
    }

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
    "nested_cv_r2_log": float(nested.mean()) if nested is not None else None,
    "nested_cv": nested_summary,
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