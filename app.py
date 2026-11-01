import streamlit as st
import requests

# 1. Cấu hình giao diện tổng thể
st.set_page_config(
    page_title="Hệ thống Định giá Bất động sản Đà Nẵng",
    page_icon="🏠",
    layout="centered"
)

# 2. Sidebar chọn loại hình định giá
st.sidebar.title("🎛️ Bảng Điều Khiển")
app_mode = st.sidebar.radio(
    "Chọn loại hình bất động sản:",
    ["🏠 Phòng trọ", "🏢 Căn hộ / Chung cư"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Lưu ý cấu hình Backend:**\n"
    "- **Căn hộ**: Chạy ở cổng `8000`\n"
    "- **Phòng trọ**: Chạy ở cổng `8001` (hoặc cùng cổng nếu gộp router)"
)

# Định nghĩa địa chỉ API tương ứng với từng loại hình# Thay vì chia 8000 và 8001, trỏ cả 2 mode về cổng 8000
API_URLS = {
    "🏠 Phòng trọ": "http://localhost:8000",
    "🏢 Căn hộ / Chung cư": "http://localhost:8000"
}
CURRENT_API_URL = API_URLS[app_mode]


# Hàm lấy metadata từ Backend
@st.cache_data
def fetch_meta(api_url):
    try:
        response = requests.get(f"{api_url}/meta", timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    # Giá trị dự phòng nếu backend chưa bật
    return {
        "ward_values": ["Hòa Khánh", "Hải Châu", "Ngũ Hành Sơn", "Sơn Trà", "Phước Mỹ"],
        "furnishing_values": ["Không nội thất", "Nội thất cơ bản", "Đầy đủ nội thất", "Nội thất cao cấp", "Không xác định"],
        "nearest_beach_code_values": ["MyKhe", "PhamVanDong", "XuanThieu", "NonNuoc"],
        "nearest_uni_code_values": ["DUE", "DUT", "VKU", "SP"]
    }


meta_data = fetch_meta(CURRENT_API_URL)


# ============================================================================
# MODE 1: PHÒNG TRỌ
# ============================================================================
if app_mode == "🏠 Phòng trọ":
    st.title("🏠 Định Giá Phòng Trọ Đà Nẵng")
    st.write("🤖 Hệ thống Học máy dự đoán giá thuê phòng trọ kết hợp giải thích SHAP")
    st.markdown("---")

    valid_wards = meta_data.get("ward_values", ["Hòa Khánh", "Hải Châu", "Ngũ Hành Sơn"])
    furnishing_values = meta_data.get("furnishing_values", ["Không nội thất", "Nội thất cơ bản", "Đầy đủ nội thất"])
    valid_beaches = meta_data.get("nearest_beach_code_values", ["MyKhe", "PhamVanDong"])
    valid_unis = meta_data.get("nearest_uni_code_values", ["DUE", "DUT"])

    col1, col2 = st.columns(2)

    with col1:
        size_m2 = st.number_input("Diện tích phòng (m²):", min_value=5.0, max_value=100.0, value=25.0, step=1.0, key="tro_size")
        ward_name = st.selectbox("Chọn Phường/Xã tại Đà Nẵng:", valid_wards, key="tro_ward")
        furnishing_status = st.selectbox("Tình trạng nội thất:", furnishing_values, key="tro_furn")

        st.markdown("**Tiện ích đi kèm:**")
        has_balcony = 1 if st.checkbox("Có ban công", key="tro_bal") else 0
        has_elevator = 1 if st.checkbox("Có thang máy", key="tro_elev") else 0
        has_window = 1 if st.checkbox("Có cửa sổ", key="tro_win") else 0

    with col2:
        dist_to_cbd = st.slider("Khoảng cách đến Trung tâm (km):", 0.0, 20.0, 3.5, 0.1, key="tro_cbd")
        dist_to_beach = st.slider("Khoảng cách đến Biển (km):", 0.0, 15.0, 2.0, 0.1, key="tro_beach")
        dist_to_nearest_uni = st.slider("Khoảng cách đến ĐH gần nhất (km):", 0.0, 10.0, 1.5, 0.1, key="tro_uni")
        nearest_beach_code = st.selectbox("Mã bãi biển gần nhất:", valid_beaches, key="tro_bcode")
        nearest_uni_code = st.selectbox("Mã trường ĐH gần nhất:", valid_unis, key="tro_ucode")

    st.markdown("---")

    if st.button("💰 DỰ ĐOÁN GIÁ THUÊ PHÒNG TRỌ", type="primary"):
        payload = {
            "size_m2": size_m2,
            "dist_to_cbd": dist_to_cbd,
            "dist_to_beach": dist_to_beach,
            "dist_to_nearest_uni": dist_to_nearest_uni,
            "has_balcony": has_balcony,
            "has_elevator": has_elevator,
            "has_window": has_window,
            "furnishing_status": furnishing_status,
            "ward_name": ward_name,
            "nearest_beach_code": nearest_beach_code,
            "nearest_uni_code": nearest_uni_code
        }

        with st.spinner("🤖 AI đang phân tích và định giá phòng trọ..."):
            try:
                response = requests.post(f"{CURRENT_API_URL}/predict", json=payload, timeout=5)
                if response.status_code == 200:
                    res_data = response.json()
                    price_vnd = res_data["prediction"]["price_vnd"]
                    price_mil = res_data["prediction"]["price_million_vnd"]

                    st.success("✨ Dự đoán thành công!")
                    st.metric(label="💵 Giá thuê gợi ý", value=f"{price_vnd:,.0f} VND/tháng", delta=f"{price_mil} Triệu VND")

                    rng = res_data["prediction"].get("range_estimate")
                    if rng:
                        st.info(f"📊 Khoảng tham khảo: Từ **{rng['low']:,.0f} VND** đến **{rng['high']:,.0f} VND**")

                    st.markdown("### 🔍 Phân tích các yếu tố ảnh hưởng (SHAP):")
                    for c in res_data["explanation"].get("contributions", [])[:5]:
                        icon = "📈" if c["direction"] == "tăng giá" else "📉"
                        st.write(f"- **{c['label']}** ({c['value']}): làm **{c['direction']}** khoảng **{c['pct_effect']:.1f}%** {icon}")
                else:
                    st.error(f"⚠️ Lỗi từ Server: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"❌ Không thể kết nối tới API ({CURRENT_API_URL}): {e}")


# ============================================================================
# MODE 2: CĂN HỘ / CHUNG CƯ
# ============================================================================
elif app_mode == "🏢 Căn hộ / Chung cư":
    st.title("🏢 Định Giá Căn Hộ & Chung Cư Đà Nẵng")
    st.write("🤖 Hệ thống Học máy dự đoán giá thuê căn hộ kết hợp giải thích SHAP")
    st.markdown("---")

    valid_wards = meta_data.get("ward_values", ["Phước Mỹ", "Hải Châu", "Sơn Trà", "Ngũ Hành Sơn"])
    furnishing_values = meta_data.get("furnishing_values", [
        "Không nội thất", "Nội thất cơ bản", "Đầy đủ nội thất", "Nội thất cao cấp", "Không xác định"
    ])
    valid_beaches = meta_data.get("nearest_beach_code_values", ["MyKhe", "PhamVanDong"])
    valid_unis = meta_data.get("nearest_uni_code_values", ["DUE", "DUT"])

    col1, col2 = st.columns(2)

    with col1:
        size_m2 = st.number_input("Diện tích căn hộ (m²):", min_value=10.0, max_value=200.0, value=60.0, step=1.0, key="canho_size")
        rooms = st.number_input("Số phòng ngủ / Không gian:", min_value=1, max_value=6, value=2, step=1, key="canho_rooms")
        ward_name = st.selectbox("Chọn Phường/Xã tại Đà Nẵng:", valid_wards, key="canho_ward")
        furnishing_status = st.selectbox("Tình trạng nội thất:", furnishing_values, key="canho_furn")

        st.markdown("**Tiện ích đi kèm:**")
        has_balcony = 1 if st.checkbox("Có ban công", value=True, key="canho_bal") else 0
        has_elevator = 1 if st.checkbox("Có thang máy", value=True, key="canho_elev") else 0
        has_window = 1 if st.checkbox("Có cửa sổ thoáng", value=True, key="canho_win") else 0

    with col2:
        dist_to_cbd = st.slider("Khoảng cách đến Trung tâm (km):", 0.0, 20.0, 2.5, 0.1, key="canho_cbd")
        dist_to_beach = st.slider("Khoảng cách đến Biển (km):", 0.0, 15.0, 1.0, 0.1, key="canho_beach")
        dist_to_nearest_uni = st.slider("Khoảng cách đến ĐH gần nhất (km):", 0.0, 10.0, 1.5, 0.1, key="canho_uni")
        nearest_beach_code = st.selectbox("Mã bãi biển gần nhất:", valid_beaches, key="canho_bcode")
        nearest_uni_code = st.selectbox("Mã trường ĐH gần nhất:", valid_unis, key="canho_ucode")

    st.markdown("---")

    if st.button("💰 DỰ ĐOÁN GIÁ THUÊ CĂN HỘ", type="primary"):
        payload = {
            "size_m2": size_m2,
            "rooms": int(rooms),
            "dist_to_cbd": dist_to_cbd,
            "dist_to_beach": dist_to_beach,
            "dist_to_nearest_uni": dist_to_nearest_uni,
            "has_balcony": has_balcony,
            "has_elevator": has_elevator,
            "has_window": has_window,
            "furnishing_status": furnishing_status,
            "ward_name": ward_name,
            "nearest_beach_code": nearest_beach_code,
            "nearest_uni_code": nearest_uni_code
        }

        with st.spinner("🤖 AI đang phân tích và định giá căn hộ..."):
            try:
                response = requests.post(f"{CURRENT_API_URL}/predict", json=payload, timeout=5)
                if response.status_code == 200:
                    res_data = response.json()
                    price_vnd = res_data["prediction"]["price_vnd"]
                    price_mil = res_data["prediction"]["price_million_vnd"]

                    st.success("✨ Dự đoán thành công!")
                    st.metric(label="💵 Giá thuê gợi ý", value=f"{price_vnd:,.0f} VND/tháng", delta=f"{price_mil} Triệu VND")

                    rng = res_data["prediction"].get("range_estimate")
                    if rng:
                        st.info(f"📊 Khoảng tham khảo: Từ **{rng['low']:,.0f} VND** đến **{rng['high']:,.0f} VND**")

                    st.markdown("### 🔍 Phân tích các yếu tố ảnh hưởng (SHAP):")
                    for c in res_data["explanation"].get("contributions", [])[:5]:
                        icon = "📈" if c["direction"] == "tăng giá" else "📉"
                        st.write(f"- **{c['label']}** ({c['value']}): làm **{c['direction']}** khoảng **{c['pct_effect']:.1f}%** {icon}")
                else:
                    st.error(f"⚠️ Lỗi từ Server: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"❌ Không thể kết nối tới API ({CURRENT_API_URL}): {e}")