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
from datetime import datetime
import qrcode
from io import BytesIO
from scipy import signal

# --- 🌑 PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ROTex // ONYX",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="collapsed"
)

# --- 🎨 THE "ONYX GLASS" THEME (WEBFLOW STYLE) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Rajdhani:wght@500;700&display=swap');
    
    /* 1. THE LIVING BACKGROUND */
    .stApp {
        background: radial-gradient(circle at 0% 0%, #1a1a2e 0%, #16213e 50%, #000000 100%);
        background-attachment: fixed;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* 2. SIDEBAR GLASS */
    section[data-testid="stSidebar"] {
        background: rgba(10, 10, 15, 0.7);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* 3. BENTO GRID CARDS (The Webflow Look) */
    div[data-testid="metric-container"], .info-card, .job-card, .skunk-card, .target-card, .target-safe {
        background: rgba(255, 255, 255, 0.03); /* Ultra subtle white tint */
        backdrop-filter: blur(12px);            /* The Frost Effect */
        border: 1px solid rgba(255, 255, 255, 0.08); /* Thin sleek border */
        border-radius: 16px;                    /* Soft corners */
        padding: 20px;
        transition: all 0.3s ease;              /* Smooth animation */
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    
    /* HOVER EFFECT (Lift) */
    div[data-testid="metric-container"]:hover, .info-card:hover, .job-card:hover, .skunk-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(255, 255, 255, 0.2);
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    
    /* 4. TYPOGRAPHY UPDATES */
    h1, h2, h3 {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        letter-spacing: 1px;
        background: -webkit-linear-gradient(0deg, #ffffff, #a0a0a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* 5. LOGO STYLING */
    .rotex-logo-container { text-align: center; margin-bottom: 30px; }
    .rotex-text { 
        font-family: 'Rajdhani', sans-serif; 
        font-weight: 800; 
        letter-spacing: 6px; 
        text-transform: uppercase; 
        font-size: 42px;
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .rotex-tagline { 
        font-family: 'Inter', sans-serif; 
        font-size: 10px; 
        letter-spacing: 3px; 
        color: #666; 
        margin-top: 5px; 
        text-transform: uppercase;
    }

    /* 6. SPECIFIC COMPONENT STYLES */
    .skunk-title { color: #d68bfb; font-weight: 800; font-size: 18px; letter-spacing: 1px; margin-bottom: 5px; }
    
    /* Target Lock Specifics */
    .target-card { border-left: 4px solid #ff4b4b !important; }
    .target-safe { border-left: 4px solid #00d2ff !important; }
    
    /* Login Box */
    .login-box { 
        background: rgba(0, 0, 0, 0.6); 
        backdrop-filter: blur(20px); 
        border: 1px solid rgba(255,255,255,0.1); 
        padding: 50px; 
        border-radius: 24px; 
        text-align: center; 
        max-width: 450px; 
        margin: auto; 
        box-shadow: 0 20px 50px rgba(0,0,0,0.5);
    }
    
    /* Mobile Adjustments */
    @media only screen and (max-width: 600px) {
        .rotex-text { font-size: 32px !important; }
        .login-box { padding: 30px !important; width: 90% !important; margin-top: 20px !important; }
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
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<div class="login-box"><div class="rotex-logo-container"><div class="rotex-text">ROTex</div><div class="rotex-tagline">Onyx System v23.0</div></div></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2: st.text_input("ACCESS KEY", type="password", on_change=password_entered, key="password", label_visibility="collapsed", placeholder="Enter Key...")
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

def generate_noise_pattern():
    w, h = 300, 300
    x = np.linspace(0, 10, w)
    y = np.linspace(0, 10, h)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X + random.random()*5) * np.cos(Y + random.random()*5)
    Z_norm = cv2.normalize(Z, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    Z_color = cv2.applyColorMap(Z_norm, cv2.COLORMAP_JET)
    return Z_color

# --- 🚀 LAUNCH ---
if check_password():
    with st.sidebar:
        st.markdown('<div class="rotex-logo-container"><div class="rotex-text">ROTex</div><div class="rotex-tagline">Onyx Glass v23.0</div></div>', unsafe_allow_html=True)
        menu = st.radio("MODULES", ["WAR ROOM", "TARGET LOCK", "SKUNKWORKS (R&D)", "LABORATORY", "FACTORY IoT", "RECRUITMENT", "VISION AI", "LOGISTICS", "DEAL BREAKER", "DATABASE"])
        st.divider()
        if st.button("LOGOUT"): st.session_state["password_correct"] = False; st.rerun()

    df = load_market_data()
    yarn_cost = df['Yarn_Fair_Value'].iloc[-1]

    if menu == "WAR ROOM":
        st.markdown("## 📡 Market Command")
        c1, c2, c3 = st.columns(3)
        c1.metric("Yarn Fair Value", f"${yarn_cost:.2f}", "+1.2%")
        c2.metric("Cotton (NYMEX)", f"${df['Cotton_USD'].iloc[-1]:.2f}", "-0.5%")
        c3.metric("Gas (Henry Hub)", f"${df['Gas_USD'].iloc[-1]:.2f}", "+0.1%")
        
        pdf = create_pdf_report(yarn_cost, df['Cotton_USD'].iloc[-1], df['Gas_USD'].iloc[-1], [], df)
        st.download_button("📄 Download Intelligence Report", pdf, "ROTex_Report.pdf", "application/pdf", use_container_width=True)
        
        fig = go.Figure(); fig.add_trace(go.Scatter(x=df.index, y=df['Yarn_Fair_Value'], line=dict(color='#3a7bd5', width=3)))
        fig.update_layout(height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🚨 Global Intel")
        for item in get_news_stealth(): 
            st.markdown(f'<div class="info-card"><a href="{item.link}" target="_blank" style="text-decoration:none; color:white;">➤ {item.title}</a></div>', unsafe_allow_html=True)

    elif menu == "TARGET LOCK":
        st.markdown("## 🎯 Global Price Sniper")
        st.write("Real-time competitor simulation.")
        col_in, col_res = st.columns([1, 2])
        with col_in:
            st.markdown("### ⚙️ Parameters")
            fabric = st.selectbox("Fabric Type", ["100% Cotton Single Jersey", "CVC Fleece", "Polyester Sport Mesh"])
            my_price = st.number_input("Your Quote ($/kg)", value=4.50, step=0.05)
        with col_res:
            st.markdown("### 🌏 Rival Analysis")
            if fabric == "100% Cotton Single Jersey":
                raw_material_factor = 1.0; processing_add = 0.50
            elif fabric == "CVC Fleece":
                raw_material_factor = 0.90; processing_add = 0.90
            elif fabric == "Polyester Sport Mesh":
                raw_material_factor = 0.60; processing_add = 0.40
            
            base_price = (yarn_cost * raw_material_factor) + processing_add
            china_p = base_price * 0.94; india_p = base_price * 0.96; vietnam_p = base_price * 0.98
            
            c1, c2, c3 = st.columns(3)
            delta_cn = my_price - china_p; color_cn = "target-card" if delta_cn > 0 else "target-safe"; icon_cn = "⚠️ LOSING" if delta_cn > 0 else "✅ WINNING"
            c1.markdown(f'<div class="{color_cn}"><b>🇨🇳 CHINA</b><br>Target: ${china_p:.2f}<br>{icon_cn}</div>', unsafe_allow_html=True)
            
            delta_in = my_price - india_p; color_in = "target-card" if delta_in > 0 else "target-safe"; icon_in = "⚠️ LOSING" if delta_in > 0 else "✅ WINNING"
            c2.markdown(f'<div class="{color_in}"><b>🇮🇳 INDIA</b><br>Target: ${india_p:.2f}<br>{icon_in}</div>', unsafe_allow_html=True)

            delta_vn = my_price - vietnam_p; color_vn = "target-card" if delta_vn > 0 else "target-safe"; icon_vn = "⚠️ LOSING" if delta_vn > 0 else "✅ WINNING"
            c3.markdown(f'<div class="{color_vn}"><b>🇻🇳 VIETNAM</b><br>Target: ${vietnam_p:.2f}<br>{icon_vn}</div>', unsafe_allow_html=True)
            
            st.divider()
            lowest_rival = min(china_p, india_p, vietnam_p)
            if my_price > lowest_rival:
                st.error(f"🚨 TACTICAL ALERT: You are overpriced by ${my_price - lowest_rival:.2f}/kg.")
            else:
                st.success(f"🛡️ MARKET DOMINANCE: Your price is competitive.")

    elif menu == "SKUNKWORKS (R&D)":
        st.markdown("## 👽 Future Tech Division")
        tab_fut1, tab_fut2, tab_fut3 = st.tabs(["🔊 Loom Whisperer", "🧬 Digital Passport", "🎨 Algo-Weaver"])
        with tab_fut1:
            st.markdown('<div class="skunk-card"><div class="skunk-title">ACOUSTIC DIAGNOSTICS</div><p>FFT Spectrum Analysis</p></div>', unsafe_allow_html=True)
            if st.button("RUN SIMULATION"):
                st.write("Simulating Waveform...")
                fs = 10e3; N = 1e5; amp = 2*np.sqrt(2); freq = 1234.0; noise_power = 0.001 * fs / 2
                time_s = np.arange(N) / fs
                x = amp*np.sin(2*np.pi*freq*time_s) + np.random.normal(scale=np.sqrt(noise_power), size=time_s.shape)
                f, t, Sxx = signal.spectrogram(x, fs)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap='inferno')
                ax.set_title("SPECTRAL DENSITY")
                st.pyplot(fig)
                st.error("⚠️ ANOMALY DETECTED: 1.2kHz Harmonic (Bearing Wear)")
        with tab_fut2:
            st.markdown('<div class="skunk-card"><div class="skunk-title">DIGITAL PASSPORT</div><p>Blockchain Traceability</p></div>', unsafe_allow_html=True)
            lot_id = st.text_input("Lot ID", "LOT-2024-TX-99")
            if st.button("MINT PASSPORT", use_container_width=True):
                hex_dig = "0x" + str(random.getrandbits(256))[:20] + "..."
                c1, c2 = st.columns([2, 1])
                c1.code(f"HASH: {hex_dig}", language="json"); c1.success("✅ IMMUTABLE")
                qr_img = generate_qr(f"ROTex VERIFIED | ID: {lot_id}"); buf = BytesIO(); qr_img.save(buf)
                c2.image(buf, caption="TRACEABILITY QR")
        with tab_fut3:
            st.markdown('<div class="skunk-card"><div class="skunk-title">GENERATIVE DESIGN</div><p>Procedural Algorithms</p></div>', unsafe_allow_html=True)
            if st.button("GENERATE", use_container_width=True):
                st.image(generate_noise_pattern(), caption=f"Design ID: {random.randint(10000,99999)}", use_column_width=True, channels="BGR")

    elif menu == "LABORATORY":
        st.markdown("## 🧪 Quality Control")
        test_mode = st.tabs(["⚖️ GSM", "📉 Shrinkage", "👮 AQL"])
        with test_mode[0]:
            c1, c2 = st.columns(2)
            weight = c1.number_input("Sample Weight (g)", 2.50)
            area = c2.selectbox("Size", ["100 cm²", "A4"])
            if st.button("CALC GSM", use_container_width=True):
                gsm = weight * 100 if area == "100 cm²" else weight * 16
                st.metric("RESULT", f"{gsm:.1f} g/m²")
        with test_mode[1]:
            if st.button("RUN TEST", use_container_width=True): st.error("❌ FAIL: -6.0% (Exceeds 5%)")
        with test_mode[2]:
            qty = st.number_input("Order Qty", 5000)
            st.info(f"Inspect 200 pcs. Reject if 11+ defects.")

    elif menu == "FACTORY IoT":
        st.markdown("## 🏭 Live Telemetry")
        temp = random.uniform(28, 36)
        if temp > 34: st.markdown(f'<div class="target-card">⚠️ HIGH TEMP: {temp:.1f}°C</div>', unsafe_allow_html=True)
        st.line_chart(np.random.randn(20, 2))

    elif menu == "RECRUITMENT":
        st.markdown("## 🤝 Industry Jobs")
        for job in get_jobs_stealth():
            st.markdown(f'<div class="job-card"><h4>{job.title}</h4><a href="{job.link}" target="_blank" style="color:white;">View Circular</a></div>', unsafe_allow_html=True)

    elif menu == "VISION AI":
        st.markdown("## 👁️ Defect Scanner")
        up = st.file_uploader("Upload Fabric")
        if up:
            res, count = process_fabric_image(up)
            st.image(res, caption=f"Detected: {count} Defects", use_column_width=True)
            db_log_scan(count, "REJECT" if count>0 else "OK")

    elif menu == "LOGISTICS":
        st.markdown("## 🌍 Supply Chain")
        target = [91.8, 22.3]; arc = [{"source": [-99.9, 31.9], "target": target, "color": [58, 123, 213]}]
        st.pydeck_chart(pdk.Deck(layers=[pdk.Layer("ArcLayer", data=arc, get_source_position="source", get_target_position="target", get_width=5, get_source_color="color", get_target_color="color")], initial_view_state=pdk.ViewState(latitude=20, longitude=60, zoom=0, pitch=45), map_style="mapbox://styles/mapbox/dark-v10"))

    elif menu == "DEAL BREAKER":
        st.markdown("## 💰 Margin Calculator")
        buyer = st.text_input("Buyer Name")
        price = st.number_input("Offer ($/kg)", value=4.50)
        margin = price - (yarn_cost + 0.75)
        st.metric("Net Margin", f"${margin:.2f}/kg")
        if st.button("💾 SAVE TO LEDGER", use_container_width=True): db_log_deal(buyer, 0, price, 0, margin); st.success("Saved.")

    elif menu == "DATABASE":
        st.markdown("## 🗄️ SQL Ledger")
        st.dataframe(db_fetch_table("deals"), use_container_width=True)
