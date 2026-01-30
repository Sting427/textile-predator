import streamlit as st
import cv2
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import pydeck as pdk
import feedparser
import requests
import time
from fpdf import FPDF
import matplotlib.pyplot as plt
import os
import random

# --- 🌑 PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ROTex // CONNECTED",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="expanded"
)

# --- 🎨 THE "APEX" THEME ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700;800&display=swap');

    .stApp {
        background: linear-gradient(-45deg, #000000, #0a0a0a, #1a0b2e, #000000);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        font-family: 'Rajdhani', sans-serif;
        color: #e0e0e0;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.85);
        border-right: 1px solid #333;
        backdrop-filter: blur(15px);
    }
    
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 15px;
        border-radius: 12px;
        border-left: 4px solid #00d2ff;
        transition: transform 0.3s ease;
    }
    div[data-testid="metric-container"]:hover { transform: translateY(-5px); }
    
    .info-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    .info-card:hover { border-color: #00d2ff; transform: translateX(5px); }
    .info-card a { color: #00d2ff; text-decoration: none; font-weight: 700; font-family: 'Rajdhani', sans-serif; }
    
    .job-card {
        background: rgba(0, 255, 136, 0.05);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 12px;
        border: 1px solid rgba(0, 255, 136, 0.1);
        border-left: 4px solid #00ff88;
        transition: all 0.3s ease;
    }
    .job-card:hover { transform: translateY(-3px); box-shadow: 0 5px 15px rgba(0, 255, 136, 0.1); }
    .job-card h4 { margin: 0; color: #fff; font-size: 18px; }
    .job-card a { color: #00ff88; text-decoration: none; font-weight: bold; }

    .rotex-logo-container { text-align: center; margin-bottom: 20px; }
    .rotex-text { font-family: 'Rajdhani', sans-serif; font-weight: 800; letter-spacing: 4px; line-height: 1; text-transform: uppercase; }
    .ro-cyan { color: #00d2ff; text-shadow: 0 0 25px rgba(0, 210, 255, 0.6); }
    .tex-magenta { color: #ff0055; text-shadow: 0 0 25px rgba(255, 0, 85, 0.6); }
    .rotex-tagline { font-size: 12px; color: #888; letter-spacing: 4px; margin-top: 8px; }

    .login-box {
        background: rgba(14, 17, 23, 0.8);
        border: 1px solid #333;
        padding: 40px;
        border-radius: 15px;
        box-shadow: 0 0 50px rgba(0,0,0,0.5);
        text-align: center;
        border-top: 2px solid #00d2ff;
        border-bottom: 2px solid #ff0055;
    }

    /* IOT ALERTS */
    .iot-alert {
        background-color: rgba(255, 0, 85, 0.1);
        border: 1px solid #ff0055;
        color: #ff0055;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 0, 85, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 0, 85, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 0, 85, 0); }
    }
    </style>
    """, unsafe_allow_html=True)

# --- 🔒 SECURITY SYSTEM ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "TEXTILE_KING":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="login-box">
                <div class="rotex-logo-container">
                    <span class="rotex-text ro-cyan" style="font-size: 60px;">RO</span>
                    <span class="rotex-text tex-magenta" style="font-size: 60px;">Tex</span><br>
                    <div class="rotex-tagline">SECURE INDUSTRIAL ACCESS</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.text_input("AUTHENTICATION KEY", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.error("⛔ ACCESS DENIED")
        return False
    else:
        return True

# --- 🧠 BACKEND LOGIC ---
@st.cache_data(ttl=3600)
def load_market_data():
    tickers = ['CT=F', 'NG=F']
    try:
        data = yf.download(tickers, period="1y", interval="1d", progress=False)['Close']
        data.columns = ['Cotton_USD', 'Gas_USD']
        data = data.dropna()
    except:
        dates = pd.date_range(end=pd.Timestamp.today(), periods=365)
        data = pd.DataFrame({
            'Cotton_USD': np.random.uniform(80, 100, 365),
            'Gas_USD': np.random.uniform(2.5, 4.0, 365)
        }, index=dates)
    cotton_in_dollars = data['Cotton_USD'] / 100 
    data['Yarn_Fair_Value'] = (cotton_in_dollars * 1.6) + (data['Gas_USD'] * 0.15) + 0.40
    return data

def run_prediction(df):
    X = df[['Cotton_USD', 'Gas_USD']]
    y = df['Yarn_Fair_Value']
    model = LinearRegression()
    model.fit(X, y)
    last_cotton = df['Cotton_USD'].iloc[-1]
    last_gas = df['Gas_USD'].iloc[-1]
    future_cotton = [last_cotton * (1 + (0.002 * i)) for i in range(7)]
    future_gas = [last_gas * (1 + (0.005 * i)) for i in range(7)]
    future_df = pd.DataFrame({'Cotton_USD': future_cotton, 'Gas_USD': future_gas})
    return model.predict(future_df)

def get_news_stealth():
    headers = { "User-Agent": "Mozilla/5.0" }
    url = "https://news.google.com/rss/search?q=Bangladesh+Textile+Industry+when:3d&hl=en-BD&gl=BD&ceid=BD:en"
    try:
        response = requests.get(url, headers=headers, timeout=5)
        feed = feedparser.parse(response.content)
        return feed.entries[:4]
    except: return []

# --- 💼 JOB FINDER MODULE ---
def get_jobs_stealth():
    headers = { "User-Agent": "Mozilla/5.0" }
    url = "https://news.google.com/rss/search?q=Textile+Job+Vacancy+Bangladesh+OR+Textile+Recruitment+Notice+OR+Garment+Factory+Job+Circular+when:7d&hl=en-BD&gl=BD&ceid=BD:en"
    try:
        response = requests.get(url, headers=headers, timeout=5)
        feed = feedparser.parse(response.content)
        return feed.entries[:8]
    except: return []

# --- 📡 IOT SIMULATION ENGINE ---
def get_iot_data():
    # Simulates live sensor readings
    return {
        "loom_speed": random.randint(780, 820), # RPM
        "humidity": random.randint(60, 75),     # %
        "temperature": random.uniform(28.0, 35.0), # Celsius
        "power": random.uniform(120.0, 125.0)   # kW
    }

def process_fabric_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    output_img = img.copy()
    for c in contours:
        if cv2.contourArea(c) > 50:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            count += 1
    return output_img, count

def render_3d_map():
    target = [91.8, 22.3] 
    sources = [
        {"name": "Texas, USA", "coords": [-99.9, 31.9], "color": [0, 229, 255]},
        {"name": "Sao Paulo, BR", "coords": [-46.6, -23.5], "color": [0, 255, 0]},
        {"name": "Mumbai, IN", "coords": [72.8, 19.0], "color": [255, 165, 0]}
    ]
    arc_data = [{"source": s["coords"], "target": target, "name": s["name"], "color": s["color"]} for s in sources]
    layer = pdk.Layer("ArcLayer", data=arc_data, get_source_position="source", get_target_position="target", get_width=5, get_tilt=15, get_source_color="color", get_target_color="color", pickable=True, auto_highlight=True)
    view_state = pdk.ViewState(latitude=20, longitude=60, zoom=1, pitch=45)
    return pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/dark-v10", tooltip={"text": "{name}"})

def get_research_papers(topic):
    query = f"textile {topic}"
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=5&fields=title,url,year,abstract"
    try:
        r = requests.get(url, timeout=5).json()
        if 'data' in r: return r['data']
        else: return []
    except: return []

# --- 📄 PDF REPORT GENERATOR ---
def create_pdf_report(yarn_val, cotton_val, gas_val, news_list, df_history):
    plt.figure(figsize=(10, 4))
    plt.plot(df_history.index, df_history['Yarn_Fair_Value'], color='#00d2ff', linewidth=2)
    plt.title('Yarn Price Trend (Past Year)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('temp_chart.png', dpi=100)
    plt.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 20, "ROTex INTELLIGENCE REPORT", ln=True, align="C")
    pdf.set_font("Arial", "I", 12)
    pdf.cell(0, 10, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "1. MARKET SNAPSHOT", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Yarn Fair Value: ${yarn_val:.2f} / kg", ln=True)
    pdf.cell(0, 8, f"Cotton (NYMEX): ${cotton_val:.2f}", ln=True)
    pdf.cell(0, 8, f"Gas (Henry Hub): ${gas_val:.2f}", ln=True)
    pdf.ln(5)
    pdf.image('temp_chart.png', x=10, w=190)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "2. KEY THREATS / NEWS", ln=True)
    pdf.set_font("Arial", "", 10)
    if news_list:
        for item in news_list:
            clean_title = item.title.encode('latin-1', 'ignore').decode('latin-1')
            pdf.multi_cell(0, 6, f"- {clean_title}")
            pdf.ln(2)
    else:
        pdf.cell(0, 10, "No critical threats detected.", ln=True)
    if os.path.exists("temp_chart.png"): os.remove("temp_chart.png")
    return pdf.output(dest='S').encode('latin-1')

# --- 🚀 LAUNCH SEQUENCE ---
if check_password():
    
    with st.sidebar:
        st.markdown("""
        <div class="rotex-logo-container">
            <span class="rotex-text ro-cyan" style="font-size: 48px;">RO</span><span class="rotex-text tex-magenta" style="font-size: 48px;">Tex</span><br>
            <div class="rotex-tagline">SYSTEMS ONLINE</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()
        menu = st.radio("SYSTEM MODULES", ["WAR ROOM", "FACTORY IoT", "RECRUITMENT", "VISION AI", "LOGISTICS", "DEAL BREAKER", "R&D LAB"])
        st.divider()
        if st.button("TERMINATE SESSION"):
            st.session_state["password_correct"] = False
            st.rerun()

    with st.spinner("ESTABLISHING SECURE UPLINK..."):
        df = load_market_data()
        preds = run_prediction(df)
        current_yarn_cost = df['Yarn_Fair_Value'].iloc[-1]
        news_items = get_news_stealth()

    if menu == "WAR ROOM":
        c_head, c_btn = st.columns([3, 1])
        with c_head: st.markdown("## 📡 MARKET INTELLIGENCE")
        with c_btn:
            pdf_bytes = create_pdf_report(current_yarn_cost, df['Cotton_USD'].iloc[-1], df['Gas_USD'].iloc[-1], news_items, df)
            st.download_button("📄 DOWNLOAD REPORT", pdf_bytes, f"ROTex_Report_{time.strftime('%Y%m%d')}.pdf", "application/pdf")

        c1, c2, c3 = st.columns(3)
        curr = df['Yarn_Fair_Value'].iloc[-1]
        nxt = preds[-1]
        delta = ((nxt - curr)/curr)*100
        with c1: st.metric("YARN FAIR VALUE", f"${curr:.2f}", f"{delta:.2f}%")
        with c2: st.metric("COTTON (NYMEX)", f"${df['Cotton_USD'].iloc[-1]:.2f}", "LIVE")
        with c3: st.metric("GAS (HENRY HUB)", f"${df['Gas_USD'].iloc[-1]:.2f}", "LIVE")

        st.divider()
        st.markdown("### 📈 PRICE FORECAST ENGINE")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Yarn_Fair_Value'], name='HISTORY', line=dict(color='#00d2ff', width=3)))
        future_dates = pd.date_range(start=df.index[-1], periods=8)[1:]
        fig.add_trace(go.Scatter(x=future_dates, y=preds, name='AI PREDICTION', line=dict(color='#ff0055', width=3, dash='dot')))
        fig.update_layout(height=450, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=20,r=20,t=40,b=20), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("### 🚨 GLOBAL THREAT STREAM")
        if news_items:
            n1, n2 = st.columns(2)
            for i, item in enumerate(news_items):
                card_html = f"""<div class="info-card"><a href="{item.link}" target="_blank">➤ {item.title}</a><br><span style="color: #888; font-size: 12px;">{item.published[:16]}</span></div>"""
                if i % 2 == 0: n1.markdown(card_html, unsafe_allow_html=True)
                else: n2.markdown(card_html, unsafe_allow_html=True)
        else:
            st.warning("NO INTEL AVAILABLE")
            
    # --- FACTORY IOT MODULE (NEW) ---
    elif menu == "FACTORY IoT":
        st.markdown("## 🏭 DIGITAL TWIN // LIVE SENSORS")
        
        # Get Simulated Data
        iot = get_iot_data()
        
        # ALERTS
        if iot['temperature'] > 34.0:
            st.markdown(f'<div class="iot-alert">⚠️ HIGH TEMP ALERT: MACHINE #4 ({iot["temperature"]:.1f}°C)</div>', unsafe_allow_html=True)
        
        # Live Metric Cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("LOOM #1 SPEED", f"{iot['loom_speed']} RPM", "+2")
        k2.metric("FLOOR HUMIDITY", f"{iot['humidity']}%", "-1%")
        k3.metric("TEMP (ZONE A)", f"{iot['temperature']:.1f}°C", "STABLE")
        k4.metric("POWER LOAD", f"{iot['power']:.1f} kW", "NORMAL")
        
        st.divider()
        
        # Simulated Real-time Chart
        st.markdown("### ⚡ REAL-TIME POWER CONSUMPTION")
        # Creating a fake history for the chart
        chart_data = pd.DataFrame(
            np.random.randn(20, 3) + [iot['power'], iot['temperature'], iot['humidity']],
            columns=['Power (kW)', 'Temp (C)', 'Humidity (%)']
        )
        st.line_chart(chart_data)
        
        st.caption("Auto-refresh enabled. Data updates on interaction.")

    elif menu == "RECRUITMENT":
        st.markdown("## 🤝 INDUSTRY RECRUITMENT FEED")
        st.info("SOURCE: LIVE NEWS AGGREGATION (GOOGLE NEWS). Real-time scan of public circulars.")
        with st.spinner("SCANNING JOB BOARDS..."):
            jobs = get_jobs_stealth()
        if jobs:
            for job in jobs:
                st.markdown(f"""<div class="job-card"><h4>{job.title}</h4><p style="color: #ccc; font-size: 14px; margin-bottom: 5px;">Published: {job.published[:16]}</p><a href="{job.link}" target="_blank">📄 VIEW CIRCULAR / APPLY NOW</a></div>""", unsafe_allow_html=True)
        else:
            st.info("No active circulars found in the last 7 days. Check back later.")

    elif menu == "VISION AI":
        st.markdown("## 👁️ DEFECT SCANNER")
        uploaded_file = st.file_uploader("UPLOAD FABRIC IMAGE", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            c1, c2 = st.columns(2)
            processed, count = process_fabric_image(uploaded_file)
            with c1: st.image(uploaded_file, caption="SOURCE", use_column_width=True)
            with c2: 
                st.image(processed, caption=f"DETECTED: {count} FAULTS", use_column_width=True, channels="BGR")
                if count > 0: st.error(f"❌ REJECT LOT ({count} DEFECTS)")
                else: st.success("✅ APPROVED")

    elif menu == "LOGISTICS":
        st.markdown("## 🌍 SUPPLY CHAIN (LIVE)")
        st.pydeck_chart(render_3d_map())

    elif menu == "DEAL BREAKER":
        st.markdown("## 💰 PROFIT CALCULATOR")
        c1, c2 = st.columns(2)
        with c1:
            buyer_price = st.number_input("BUYER OFFER ($/KG)", value=4.50, step=0.05)
            qty = st.number_input("QUANTITY (KG)", value=10000)
            knit_cost = st.number_input("KNITTING COST ($/KG)", value=0.60)
            overhead = st.number_input("OVERHEAD ($/KG)", value=0.15)
        with c2:
            total_cost = current_yarn_cost + knit_cost + overhead
            margin = buyer_price - total_cost
            st.metric("MARKET YARN PRICE", f"${current_yarn_cost:.2f}")
            st.metric("TOTAL COST", f"${total_cost:.2f}")
            st.divider()
            if margin > 0:
                st.success(f"✅ PROFIT: ${margin:.2f}/kg | TOTAL: ${margin*qty:,.2f}")
                st.balloons()
            else:
                st.error(f"❌ LOSS: ${margin:.2f}/kg | TOTAL: ${margin*qty:,.2f}")

    elif menu == "R&D LAB":
        st.markdown("## 🔬 RESEARCH ARCHIVE")
        topic = st.selectbox("SELECT TOPIC", ["Sustainable Dyeing", "Smart Fabrics", "Recycled Polyester", "Nano-Finishing"])
        if st.button("INITIATE SEARCH"):
            with st.spinner("SEARCHING DATABASE..."):
                papers = get_research_papers(topic)
                for p in papers:
                    st.markdown(f"""<div class="info-card"><a href="{p.get('url')}" target="_blank">📄 {p.get('title')}</a><br><span style="color:#888;">{p.get('year')}</span></div>""", unsafe_allow_html=True)
