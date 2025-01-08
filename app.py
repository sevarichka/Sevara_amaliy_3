import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Modelni yuklash
model = joblib.load("loan_model.pkl")

# Dizayn sozlamalari
st.set_page_config(
    page_title="Bank Kredit Tahlili",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sarlavha
st.title("🏦 Bank Kredit Tahlil Ilovasi")
st.markdown("Siz uchun qulay xizmatlar! Bankimizda mijozlarga ishonch va yuksak sifatni taqdim etamiz.")

# Sidebar menyu
st.sidebar.header("📊 Ilova Bo'limlari")
menu = st.sidebar.radio(
    "Menyu:",
    ["Bosh Sahifa", "Kredit Tasdiqlash", "Statistik Tahlil"]
)

# Bosh Sahifa
if menu == "Bosh Sahifa":
    st.header("Bizning xizmatlarimiz")
    st.markdown("""
    - 💰 **Kredit xizmatlari**: Turli xil kredit turlari mavjud.
    - 📈 **Moliyaviy tahlil**: Mijozlarning moliyaviy ahvolini chuqur tahlil qilish.
    - 🛡️ **Xavfsiz tranzaksiyalar**: Sizning mablag'laringiz xavfsiz qo'llarimizda.
    """)

# Kredit Tasdiqlash
elif menu == "Kredit Tasdiqlash":
    st.header("📋 Kredit Tasdiqlash")

    # Foydalanuvchi ma'lumotlarini kiritish
    st.markdown("Mijoz ma'lumotlarini kiriting:")
    age = st.slider("Yosh:", 18, 100, 25)
    income = st.slider("Yillik daromad (so'm):", 10000, 1000000, 50000)
    loan_amount = st.slider("Kredit miqdori (so'm):", 1000, 500000, 20000)
    credit_score = st.slider("Kredit reytingi:", 300, 850, 700)

    # Animatsion natija
    if st.button("🔮 Aniqlash"):
        features = np.array([[age, income, loan_amount, credit_score]])
        prediction = model.predict(features)
        if prediction[0] == 1:
            st.success("✅ Kredit tasdiqlandi!")
            st.balloons()
        else:
            st.error("❌ Kredit rad etildi.")
            st.snow()

# Statistik Tahlil
elif menu == "Statistik Tahlil":
    st.header("📈 Bank Statistikasi")

    # Dummy ma'lumotlar
    data = pd.DataFrame({
        'Yosh': np.random.randint(18, 70, 100),
        'Daromad': np.random.randint(30000, 150000, 100),
        'Kredit miqdori': np.random.randint(10000, 50000, 100),
        'Kredit reytingi': np.random.randint(400, 800, 100)
    })

    # 3D grafikalar
    st.markdown("### 3D Kredit Miqdori Tahlili")
    fig = px.scatter_3d(
        data,
        x='Daromad',
        y='Kredit miqdori',
        z='Kredit reytingi',
        color='Yosh',
        title="Daromad, Kredit miqdori va Reyting bo'yicha 3D grafik"
    )
    st.plotly_chart(fig)

    st.markdown("### Yosh va Daromad Jadvallari")
    table_data = data.groupby('Yosh').mean().reset_index()
    st.write(table_data)

    # 2D Grafik
    st.markdown("### Daromad va Kredit miqdori bo'yicha bog'liqlik")
    fig2 = px.line(data, x='Daromad', y='Kredit miqdori', title="Daromad va Kredit miqdori")
    st.plotly_chart(fig2)
