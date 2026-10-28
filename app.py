import streamlit as st
import requests

# 1. Cấu hình giao diện trang web
st.set_page_config(
    page_title="Định giá Phòng trọ Đà Nẵng",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Ứng Dụng Định Giá Phòng Trọ Đà Nẵng")
st.write("🤖 Hệ thống Học máy dự đoán giá thuê trọ kết hợp giải thích bằng SHAP (API FastAPI)")
st.markdown("---")

# Địa chỉ FastAPI backend
FASTAPI_URL = "http://localhost:8000"


# 2. Lấy dữ liệu metadata từ Backend (/meta) để hiển thị danh sách Phường động
@st.cache_data
def fetch_meta():
    try:
        response = requests.get(f"{FASTAPI_URL}/meta")
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None
    return None


meta_data = fetch_meta()

if not meta_data:
    st.error(
        "❌ Không thể kết nối đến FastAPI Backend! Hãy đảm bảo bạn đã chạy `uvicorn main:app --reload` ở cổng 8000.")
    st.stop()

valid_wards = meta_data.get("ward_values", ["Hòa Khánh", "Hải Châu", "Ngũ Hành Sơn", "Liên Chiểu"])
furnishing_values = meta_data.get("furnishing_values", ["Không nội thất", "Nội thất cơ bản", "Đầy đủ nội thất"])

# 3. Tạo Form nhập liệu cho người dùng
st.subheader("📋 Nhập thông tin phòng trọ:")

col1, col2 = st.columns(2)

with col1:
    size_m2 = st.number_input("Diện tích phòng (m²):", min_value=5.0, max_value=200.0, value=25.0, step=1.0)
    ward_name = st.selectbox("Chọn Phường/Xã tại Đà Nẵng:", valid_wards)
    furnishing_status = st.selectbox("Tình trạng nội thất:", furnishing_values)

    st.markdown("**Tiện ích đi kèm:**")
    has_balcony = 1 if st.checkbox("Có ban công") else 0
    has_elevator = 1 if st.checkbox("Có thang máy") else 0
    has_window = 1 if st.checkbox("Có cửa sổ") else 0

with col2:
    dist_to_cbd = st.slider("Khoảng cách đến Trung tâm (km):", 0.0, 20.0, 3.5, 0.1)
    dist_to_beach = st.slider("Khoảng cách đến Biển (km):", 0.0, 15.0, 2.0, 0.1)
    dist_to_nearest_uni = st.slider("Khoảng cách đến ĐH gần nhất (km):", 0.0, 10.0, 1.5, 0.1)

st.markdown("---")

# 4. Gửi Request lên FastAPI khi bấm nút dự đoán
if st.button("💰 GỬI DỰ ĐOÁN GIÁ THUÊ", type="primary"):
    payload = {
        "size_m2": size_m2,
        "dist_to_cbd": dist_to_cbd,
        "dist_to_beach": dist_to_beach,
        "dist_to_nearest_uni": dist_to_nearest_uni,
        "has_balcony": has_balcony,
        "has_elevator": has_elevator,
        "has_window": has_window,
        "furnishing_status": furnishing_status,
        "ward_name": ward_name
    }

    with st.spinner("🤖 AI đang phân tích và tính toán giá..."):
        try:
            response = requests.post(f"{FASTAPI_URL}/predict", json=payload)
            if response.status_code == 200:
                res_data = response.json()

                # Hiển thị kết quả
                price_vnd = res_data["prediction"]["price_vnd"]
                price_mil = res_data["prediction"]["price_million_vnd"]

                st.success("✨ Dự đoán thành công!")
                st.metric(label="💵 Giá thuê gợi ý chuẩn thị trường", value=f"{price_vnd:,.0f} VND/tháng",
                          delta=f"{price_mil} Triệu VND")

                # Hiển thị khoảng ước lượng nếu có
                rng = res_data["prediction"].get("range_estimate")
                if rng:
                    st.info(
                        f"📊 Khoảng tham khảo (theo sai số test): Từ **{rng['low']:,.0f} VND** đến **{rng['high']:,.0f} VND**")

                # Hiển thị giải thích yếu tố tác động (SHAP)
                st.markdown("### 🔍 Phân tích các yếu tố ảnh hưởng (SHAP):")
                contribs = res_data["explanation"].get("contributions", [])

                for c in contribs[:5]:  # Hiển thị top 5 yếu tố tác động mạnh nhất
                    direction_icon = "📈" if c["direction"] == "tăng giá" else "📉"
                    st.write(
                        f"- **{c['label']}** ({c['value']}): làm **{c['direction']}** khoảng **{c['pct_effect']:.1f}%** {direction_icon}")

            else:
                st.error(f"⚠️ Lỗi từ Server: {response.json().get('detail')}")
        except Exception as e:
            st.error(f"❌ Không thể kết nối tới API: {e}")