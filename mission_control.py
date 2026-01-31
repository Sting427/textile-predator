import streamlit as st
import cv2
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import pydeck as pdk
import feedparser
import requests
import time
from fpdf import FPDF
import matplotlib.pyplot as plt
import os
import random
import sqlite3
from datetime import datetime, timedelta
import qrcode
from io import BytesIO
from scipy import signal

# --- 🌑 PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ROTex // SINGULARITY",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="collapsed"
)

# --- 🎨 THE "SINGULARITY" THEME (Glass + Neon + Motion) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Rajdhani:wght@500;700;800&display=swap');
    
    /* 1. DEEP SPACE BACKGROUND */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #1a1a2e 0%, #000000 90%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* 2. HYPER-GLASS CARDS */
    div[data-testid="metric-container"], .info-card, .job-card, .skunk-card, .target-card, .target-safe {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-testid="metric-container"]:hover, .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(0, 210, 255, 0.15);
        border-color: rgba(0, 210, 255, 0.3);
    }

    /* 3. NEON TEXT & LOGO */
    .rotex-text {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 800;
        font-size: 50px;
        background: linear-gradient(90deg, #00d2ff, #ffffff, #00d2ff);
        background-size: 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 5s linear infinite;
    }
    @keyframes shine { 0% { background-position: 200%; } 100% { background-position: 0%; } }
    
    .rotex-tagline { font-family: 'Rajdhani'; letter-spacing: 4px; color: #666; font-size: 12px; text-transform: uppercase; }

    /* 4. UI ELEMENTS */
    .target-card { border-left: 4px solid #ff4b4b !important; background: linear-gradient(90deg, rgba(255,0,0,0.1), transparent); }
    .target-safe { border-left: 4px solid #00ff88 !important; background: linear-gradient(90deg, rgba(0,255,136,0.1), transparent); }
    
    /* Login Box */
    .login-box {
        background: rgba(0,0,0,0.8);
        border: 1px solid #333;
        padding: 50px;
        border-radius: 20px;
        text-align: center;
        max-width: 450px;
        margin: auto;
        box-shadow: 0 0 100px rgba(0, 210, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 🗄️ DATABASE ---
DB_FILE = "rotex_core.db"
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS deals (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, buyer TEXT, qty REAL, price REAL, cost REAL, margin REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS scans (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, defects INTEGER, status TEXT)''')
    conn.commit(); conn.close()
def db_log_deal(buyer, qty, price, cost, margin):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT INTO deals (timestamp, buyer, qty, price, cost, margin) VALUES (?, ?, ?, ?, ?, ?)", (datetime.now().strftime("%Y-%m-%d %H:%M"), buyer, qty, price, cost, margin))
    conn.commit(); conn.close()
def db_log_scan(defects, status):
    conn = sqlite3.connect(DB_FILE); c = conn.cursor()
    c.execute("INSERT INTO scans (timestamp, defects, status) VALUES (?, ?, ?)", (datetime.now().strftime("%Y-%m-%d %H:%M"), defects, status))
    conn.commit(); conn.close()
def db_fetch_table(table_name):
    conn = sqlite3.connect(DB_FILE); df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY id DESC", conn); conn.close()
    return df
init_db()

# --- 🔒 SECURITY ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "TEXTILE_KING":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else: st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown('<div class="login-box"><div class="rotex-logo-container"><div class="rotex-text">ROTex</div><div class="rotex-tagline">Singularity v25.0</div></div></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2: st.text_input("IDENTITY VERIFICATION", type="password", on_change=password_entered, key="password", label_visibility="collapsed", placeholder="Enter Key...")
        return False
    return st.session_state["password_correct"]

# --- 🧠 LOGIC & UTILS ---
@st.cache_data(ttl=3600)
def load_market_data():
    try:
        data = yf.download(['CT=F', 'NG=F'], period="1y", interval="1d", progress=False)['Close']
        data.columns = ['Cotton_USD', 'Gas_USD']
        data['Yarn_Fair_Value'] = ((data['Cotton_USD']/100) * 1.6) + (data['Gas_USD'] * 0.15) + 0.40
        return data.dropna()
    except:
        dates = pd.date_range(end=pd.Timestamp.today(), periods=100)
        return pd.DataFrame({'Cotton_USD': 85.0, 'Gas_USD': 3.0, 'Yarn_Fair_Value': 4.10}, index=dates)

def get_news_stealth():
    try: return feedparser.parse(requests.get("https://news.google.com/rss/search?q=Bangladesh+Textile+Industry+when:3d&hl=en-BD&gl=BD&ceid=BD:en").content).entries[:4]
    except: return []

def get_jobs_stealth():
    try: return feedparser.parse(requests.get("https://news.google.com/rss/search?q=Textile+Job+Vacancy+Bangladesh+when:7d&hl=en-BD&gl=BD&ceid=BD:en").content).entries[:8]
    except: return []

def process_fabric_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_img = img.copy()
    count = 0
    for c in contours:
        if cv2.contourArea(c) > 50:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            count += 1
    return output_img, count

def create_pdf_report(yarn, cotton, gas, news, df_hist):
    plt.figure(figsize=(10, 4)); plt.plot(df_hist.index, df_hist['Yarn_Fair_Value'], color='#00d2ff'); plt.savefig('temp.png'); plt.close()
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", "B", 24); pdf.cell(0, 20, "ROTex REPORT", ln=True, align="C")
    pdf.set_font("Arial", "", 12); pdf.cell(0, 10, f"Yarn: ${yarn:.2f} | Cotton: ${cotton:.2f} | Gas: ${gas:.2f}", ln=True)
    pdf.image('temp.png', x=10, w=190)
    return pdf.output(dest='S').encode('latin-1')

def generate_qr(data):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img

def generate_noise_pattern(freq, chaos):
    w, h = 300, 300
    x = np.linspace(0, freq, w)
    y = np.linspace(0, freq, h)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X + random.random()*chaos) * np.cos(Y + random.random()*chaos)
    Z_norm = cv2.normalize(Z, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    Z_color = cv2.applyColorMap(Z_norm, cv2.COLORMAP_JET)
    return Z_color

# --- 🚀 LAUNCH ---
if check_password():
    with st.sidebar:
        st.markdown('<div class="rotex-logo-container"><div class="rotex-text">ROTex</div><div class="rotex-tagline">System Online</div></div>', unsafe_allow_html=True)
        menu = st.radio("COMMAND", ["GLOBAL COMMAND", "WAR GAMES", "ALIEN TECH", "DIGITAL TWIN LAB", "HOLOGRAPHIC FLOOR", "NEURAL SCANNER", "ORBITAL LOGISTICS", "DEAL BREAKER", "LEDGER"])
        st.divider()
        if st.button("LOGOUT"): st.session_state["password_correct"] = False; st.rerun()

    df = load_market_data()
    yarn_cost = df['Yarn_Fair_Value'].iloc[-1]

    # 1. GLOBAL COMMAND CENTER (Upgraded War Room)
    if menu == "GLOBAL COMMAND":
        st.markdown("## 📡 GLOBAL COMMAND CENTER")
        
        # Live Ticker
        st.markdown(f"<div style='background:rgba(0,0,0,0.5); padding:10px; border-radius:5px; white-space:nowrap; overflow:hidden; color:#00ff88; font-family:monospace;'>LIVE FEED: COTTON: ${df['Cotton_USD'].iloc[-1]:.2f} ▲ | GAS: ${df['Gas_USD'].iloc[-1]:.2f} ▼ | YARN FAIR VALUE: ${yarn_cost:.2f} ▲ | SHANGHAI FUTURES: UP 2.1% | CHITTAGONG PORT: CONGESTION LOW</div>", unsafe_allow_html=True)
        st.write("")

        c1, c2, c3 = st.columns(3)
        c1.metric("Yarn Index", f"${yarn_cost:.2f}", "+1.2%")
        c2.metric("Cotton Futures", f"${df['Cotton_USD'].iloc[-1]:.2f}", "-0.5%")
        c3.metric("Energy Index", f"${df['Gas_USD'].iloc[-1]:.2f}", "+0.1%")
        
        col_main, col_intel = st.columns([2, 1])
        with col_main:
            # Geopolitical Heatmap (Simulated)
            st.markdown("### 🗺️ Geopolitical Threat Map")
            map_data = pd.DataFrame({'lat': [23.8, 31.2, 21.0, 39.9], 'lon': [90.4, 121.4, 105.8, 116.4], 'risk': [10, 50, 30, 80]})
            layer = pdk.Layer("HeatmapLayer", data=map_data, get_position='[lon, lat]', get_weight="risk", radiusPixels=60)
            view_state = pdk.ViewState(latitude=25, longitude=100, zoom=2, pitch=45)
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/dark-v10"))
        
        with col_intel:
            st.markdown("### 🧠 AI Sentiment Analysis")
            news = get_news_stealth()
            sentiment_score = random.randint(40, 80)
            st.progress(sentiment_score)
            st.caption(f"Market Sentiment: {sentiment_score}% BULLISH")
            for item in news: 
                st.markdown(f'<div class="info-card" style="font-size:12px; padding:10px;"><a href="{item.link}" target="_blank" style="color:#00d2ff; text-decoration:none;">➤ {item.title[:60]}...</a></div>', unsafe_allow_html=True)

    # 2. WAR GAMES (Upgraded Target Lock)
    elif menu == "WAR GAMES":
        st.markdown("## ⚔️ WAR GAMES SIMULATOR")
        st.write("Run 'What-If' scenarios to predict deal outcomes.")
        
        col_ctrl, col_sim = st.columns([1, 2])
        with col_ctrl:
            st.markdown("### 🎛️ Simulation Controls")
            fabric = st.selectbox("Fabric Class", ["Cotton Single Jersey", "CVC Fleece", "Poly Mesh"])
            my_quote = st.number_input("Your Quote ($/kg)", 4.50)
            shock = st.slider("Global Price Shock (%)", -20, 20, 0)
            
        with col_sim:
            # Dynamic Calc
            base = yarn_cost * (1 + shock/100)
            china = base * 0.94; india = base * 0.96; vietnam = base * 0.98
            
            # Gauge Chart for Probability
            diff = my_quote - min(china, india, vietnam)
            prob = max(0, min(100, 100 - (diff * 200))) # Fake probability algo
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = prob,
                title = {'text': "Win Probability"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#00d2ff"},
                         'steps': [{'range': [0, 50], 'color': "#333"}, {'range': [50, 100], 'color': "#111"}]}
            ))
            fig.update_layout(height=250, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
            st.plotly_chart(fig, use_container_width=True)
            
            # Rivals
            c1, c2, c3 = st.columns(3)
            c1.metric("🇨🇳 China", f"${china:.2f}")
            c2.metric("🇮🇳 India", f"${india:.2f}")
            c3.metric("🇻🇳 Vietnam", f"${vietnam:.2f}")

    # 3. ALIEN TECH (Upgraded Skunkworks)
    elif menu == "ALIEN TECH":
        st.markdown("## 👽 ALIEN TECHNOLOGY DIVISION")
        tab1, tab2, tab3 = st.tabs(["🔊 Loom Whisperer 2.0", "🧬 Algo-Weaver 2.0", "⛓️ Digital Passport"])
        
        with tab1:
            st.markdown('<div class="skunk-card"><div class="skunk-title">3D ACOUSTIC TOPOLOGY</div></div>', unsafe_allow_html=True)
            if st.button("SCAN FREQUENCIES"):
                # 3D Surface Plot of Sound
                x = np.linspace(-5, 5, 100); y = np.linspace(-5, 5, 100)
                X, Y = np.meshgrid(x, y)
                R = np.sqrt(X**2 + Y**2); Z = np.sin(R)
                fig = go.Figure(data=[go.Surface(z=Z, colorscale='Viridis')])
                fig.update_layout(title='Motor Harmonic Surface', autosize=False, width=500, height=500, margin=dict(l=65, r=50, b=65, t=90), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown('<div class="skunk-card"><div class="skunk-title">INTERACTIVE GENERATIVE DESIGN</div></div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            freq = c1.slider("Pattern Frequency", 1, 20, 10)
            chaos = c2.slider("Chaos Factor", 1, 10, 5)
            if st.button("GENERATE ORGANIC PATTERN", use_container_width=True):
                pat = generate_noise_pattern(freq, chaos)
                st.image(pat, use_column_width=True, channels="BGR")
                
        with tab3:
            st.write("Blockchain Passport System (Standard)")
            st.info("System Operational. Minting active.")

    # 4. DIGITAL TWIN LAB (Upgraded Laboratory)
    elif menu == "DIGITAL TWIN LAB":
        st.markdown("## 🧪 DIGITAL TWIN LAB")
        test = st.selectbox("Select Protocol", ["GSM Calc", "Shrinkage Sim", "AQL Inspector"])
        if test == "GSM Calc":
            w = st.number_input("Weight (g)", 2.5)
            st.metric("GSM", f"{w*100:.1f} g/m²")
        elif test == "AQL Inspector":
            qty = st.number_input("Lot Qty", 5000)
            st.success(f"Protocol: Inspect 200 pcs. Reject if > 10 defects.")
            st.progress(10)

    # 5. HOLOGRAPHIC FLOOR (Upgraded IoT)
    elif menu == "HOLOGRAPHIC FLOOR":
        st.markdown("## 🏭 HOLOGRAPHIC FLOOR")
        c1, c2, c3 = st.columns(3)
        
        # Simulated Radial Gauges
        fig_speed = go.Figure(go.Indicator(mode="gauge+number", value=random.randint(750, 850), title={'text': "Loom RPM"}, gauge={'axis': {'range': [0, 1000]}, 'bar': {'color': "#00ff88"}}))
        fig_speed.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        c1.plotly_chart(fig_speed, use_container_width=True)
        
        fig_temp = go.Figure(go.Indicator(mode="gauge+number", value=random.randint(28, 40), title={'text': "Temp (°C)"}, gauge={'axis': {'range': [0, 50]}, 'bar': {'color': "#ff0055"}}))
        fig_temp.update_layout(height=200, paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
        c2.plotly_chart(fig_temp, use_container_width=True)
        
        c3.markdown("### 🔮 Predictive AI")
        c3.info("Loom #4: Bearing Failure predicted in 48 hours.")
        c3.success("Loom #1-3: Optimal.")

    # 6. NEURAL SCANNER (Upgraded Vision)
    elif menu == "NEURAL SCANNER":
        st.markdown("## 👁️ NEURAL DEFECT SCANNER")
        up = st.file_uploader("Upload Fabric Feed")
        if up:
            img, cnt = process_fabric_image(up)
            st.image(img, caption=f"Neural Net Detected: {cnt} Anomalies", use_column_width=True)
            if cnt > 0: st.error("⚠️ QUALITY THRESHOLD BREACHED")
            else: st.success("✅ GRADE A CERTIFIED")

    # 7. ORBITAL LOGISTICS
    elif menu == "ORBITAL LOGISTICS":
        st.markdown("## 🌍 ORBITAL TRACKING")
        # Animated Arc Layer
        data = [{"source": [90.4, 23.8], "target": [-74.0, 40.7], "color": [0, 255, 136]}] # Dhaka to NYC
        layer = pdk.Layer("ArcLayer", data=data, get_width=5, get_source_position="source", get_target_position="target", get_source_color="color", get_target_color="color")
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1, pitch=40), map_style="mapbox://styles/mapbox/dark-v10"))
        
        st.markdown("### 🚢 Live Manifest")
        st.dataframe(pd.DataFrame({"Vessel": ["Ever Given", "Maersk Alabama"], "Dest": ["NYC", "Hamburg"], "ETA": ["4 Days", "12 Days"], "Status": ["On Time", "Delayed"]}), use_container_width=True)

    # 8. UTILS
    elif menu == "DEAL BREAKER":
        st.markdown("## 💰 MARGIN CALCULATOR")
        p = st.number_input("Price", 4.50)
        st.metric("Margin", f"${p - (yarn_cost+0.75):.2f}")
        if st.button("Save"): db_log_deal("Test", 0, p, 0, 0); st.success("Saved")

    elif menu == "LEDGER":
        st.dataframe(db_fetch_table("deals"), use_container_width=True)
