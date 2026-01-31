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
import sqlite3
from datetime import datetime

# --- 🌑 PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ROTex // SQL SYSTEM",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="expanded"
)

# --- 🎨 THE "APEX" THEME ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700;800&display=swap');
    .stApp { background: linear-gradient(-45deg, #000000, #0a0a0a, #1a0b2e, #000000); background-size: 400% 400%; animation: gradient 15s ease infinite; font-family: 'Rajdhani', sans-serif; color: #e0e0e0; }
    @keyframes gradient { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
    section[data-testid="stSidebar"] { background-color: rgba(0, 0, 0, 0.85); border-right: 1px solid #333; backdrop-filter: blur(15px); }
    div[data-testid="metric-container"] { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); padding: 15px; border-radius: 12px; border-left: 4px solid #00d2ff; }
    .info-card { background: rgba(255, 255, 255, 0.03); padding: 15px; border-radius: 10px; margin-bottom: 12px; border: 1px solid rgba(255, 255, 255, 0.05); }
    .job-card { background: rgba(0, 255, 136, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 12px; border: 1px solid rgba(0, 255, 136, 0.1); border-left: 4px solid #00ff88; }
    .rotex-logo-container { text-align: center; margin-bottom: 20px; }
    .rotex-text { font-family: 'Rajdhani', sans-serif; font-weight: 800; letter-spacing: 4px; text-transform: uppercase; }
    .ro-cyan { color: #00d2ff; text-shadow: 0 0 25px rgba(0, 210, 255, 0.6); }
    .tex-magenta { color: #ff0055; text-shadow: 0 0 25px rgba(255, 0, 85, 0.6); }
    .login-box { background: rgba(14, 17, 23, 0.8); border: 1px solid #333; padding: 40px; border-radius: 15px; text-align: center; border-top: 2px solid #00d2ff; border-bottom: 2px solid #ff0055; }
    .iot-alert { background-color: rgba(255, 0, 85, 0.1); border: 1px solid #ff0055; color: #ff0055; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center; margin-bottom: 10px; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(255, 0, 85, 0.4); } 70% { box-shadow: 0 0 0 10px rgba(255, 0, 85, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 0, 85, 0); } }
    </style>
    """, unsafe_allow_html=True)

# --- 🗄️ REAL DATABASE ENGINE (SQLite) ---
DB_FILE = "rotex_core.db"

def init_db():
    """Initializes the SQL database tables if they don't exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Table 1: Financial Transactions
    c.execute('''CREATE TABLE IF NOT EXISTS deals 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp TEXT, 
                  buyer TEXT, 
                  qty REAL, 
                  price REAL, 
                  cost REAL, 
                  margin REAL)''')
    # Table 2: AI Scan Logs
    c.execute('''CREATE TABLE IF NOT EXISTS scans 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp TEXT, 
                  defects INTEGER, 
                  status TEXT)''')
    conn.commit()
    conn.close()

def db_log_deal(buyer, qty, price, cost, margin):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO deals (timestamp, buyer, qty, price, cost, margin) VALUES (?, ?, ?, ?, ?, ?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M"), buyer, qty, price, cost, margin))
    conn.commit()
    conn.close()

def db_log_scan(defects, status):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO scans (timestamp, defects, status) VALUES (?, ?, ?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M"), defects, status))
    conn.commit()
    conn.close()

def db_fetch_table(table_name):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY id DESC", conn)
    conn.close()
    return df

# Initialize Database on Startup
init_db()

# --- 🔒 SECURITY ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "TEXTILE_KING":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else: st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown('<div class="login-box"><div class="rotex-logo-container"><span class="rotex-text ro-cyan" style="font-size: 60px;">RO</span><span class="rotex-text tex-magenta" style="font-size: 60px;">Tex</span><br><div class="rotex-tagline">SECURE SQL GATEWAY</div></div></div>', unsafe_allow_html=True)
            st.text_input("AUTHENTICATION KEY", type="password", on_change=password_entered, key="password")
        return False
    return st.session_state["password_correct"]

# --- 🧠 CORE LOGIC ---
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

def get_research_papers(topic):
    try: return requests.get(f"https://api.semanticscholar.org/graph/v1/paper/search?query=textile+{topic}&limit=5&fields=title,url,year").json().get('data', [])
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

# --- 🚀 LAUNCH ---
if check_password():
    with st.sidebar:
        st.markdown('<div class="rotex-logo-container"><span class="rotex-text ro-cyan" style="font-size: 48px;">RO</span><span class="rotex-text tex-magenta" style="font-size: 48px;">Tex</span><br><div class="rotex-tagline">SYSTEMS ONLINE</div></div>', unsafe_allow_html=True)
        menu = st.radio("SENSORS", ["WAR ROOM", "FACTORY IoT", "RECRUITMENT", "VISION AI", "LOGISTICS", "DEAL BREAKER", "DATABASE", "R&D LAB"])
        if st.button("TERMINATE"):
            st.session_state["password_correct"] = False
            st.rerun()

    df = load_market_data()
    yarn_cost = df['Yarn_Fair_Value'].iloc[-1]

    if menu == "WAR ROOM":
        c_head, c_btn = st.columns([3, 1])
        with c_head: st.markdown("## 📡 MARKET COMMAND")
        with c_btn:
            pdf = create_pdf_report(yarn_cost, df['Cotton_USD'].iloc[-1], df['Gas_USD'].iloc[-1], [], df)
            st.download_button("📄 DOWNLOAD REPORT", pdf, "ROTex_Report.pdf", "application/pdf")
        m1, m2, m3 = st.columns(3)
        m1.metric("YARN FAIR VALUE", f"${yarn_cost:.2f}")
        m2.metric("COTTON", f"${df['Cotton_USD'].iloc[-1]:.2f}")
        m3.metric("GAS", f"${df['Gas_USD'].iloc[-1]:.2f}")
        fig = go.Figure(); fig.add_trace(go.Scatter(x=df.index, y=df['Yarn_Fair_Value'], line=dict(color='#00d2ff', width=3)))
        fig.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### 🚨 INTEL FEED")
        for item in get_news_stealth(): st.markdown(f'<div class="info-card"><a href="{item.link}" target="_blank">➤ {item.title}</a></div>', unsafe_allow_html=True)

    elif menu == "FACTORY IoT":
        st.markdown("## 🏭 LIVE SENSOR TELEMETRY")
        temp = random.uniform(28, 36)
        if temp > 34: st.markdown(f'<div class="iot-alert">⚠️ HIGH TEMP ALERT: {temp:.1f}°C</div>', unsafe_allow_html=True)
        k1, k2, k3 = st.columns(3)
        k1.metric("LOOM SPEED", f"{random.randint(790, 810)} RPM")
        k2.metric("HUMIDITY", f"{random.randint(62, 68)}%")
        k3.metric("TEMP", f"{temp:.1f}°C")
        st.line_chart(np.random.randn(20, 2))

    elif menu == "RECRUITMENT":
        st.markdown("## 🤝 INDUSTRY RECRUITMENT")
        for job in get_jobs_stealth():
            st.markdown(f'<div class="job-card"><h4>{job.title}</h4><a href="{job.link}" target="_blank">VIEW CIRCULAR</a></div>', unsafe_allow_html=True)

    elif menu == "VISION AI":
        st.markdown("## 👁️ DEFECT SCANNER")
        up = st.file_uploader("Upload Fabric")
        if up:
            res, count = process_fabric_image(up)
            status = "REJECT" if count > 0 else "APPROVED"
            st.image(res, caption=f"Detected: {count} Defects")
            # LOG TO DATABASE
            db_log_scan(count, status)
            if count > 0: st.error("❌ LOGGED TO DATABASE: REJECTED")
            else: st.success("✅ LOGGED TO DATABASE: APPROVED")

    elif menu == "LOGISTICS":
        st.markdown("## 🌍 SUPPLY CHAIN")
        target = [91.8, 22.3]; arc = [{"source": [-99.9, 31.9], "target": target, "color": [0, 229, 255]}]
        st.pydeck_chart(pdk.Deck(layers=[pdk.Layer("ArcLayer", data=arc, get_source_position="source", get_target_position="target", get_width=5, get_source_color="color", get_target_color="color")], initial_view_state=pdk.ViewState(latitude=20, longitude=60, zoom=1, pitch=45), map_style="mapbox://styles/mapbox/dark-v10"))

    elif menu == "DEAL BREAKER":
        st.markdown("## 💰 MARGIN CALCULATOR")
        buyer = st.text_input("BUYER NAME")
        price = st.number_input("OFFER ($/kg)", value=4.50)
        qty = st.number_input("QTY (kg)", value=5000)
        margin = price - (yarn_cost + 0.75)
        st.metric("NET MARGIN", f"${margin:.2f}/kg", delta=f"${margin*qty:,.0f} Total")
        if st.button("💾 SAVE TO DATABASE"):
            db_log_deal(buyer, qty, price, (yarn_cost+0.75), margin)
            st.success("TRANSACTION COMMITTED TO SQL.")

    elif menu == "DATABASE":
        st.markdown("## 🗄️ SQL DATABASE VIEWER")
        tab1, tab2 = st.tabs(["💰 DEAL LEDGER", "👁️ QC SCANS"])
        with tab1:
            df_deals = db_fetch_table("deals")
            st.dataframe(df_deals, use_container_width=True)
            st.download_button("📂 EXPORT DEALS", df_deals.to_csv().encode('utf-8'), "rotex_deals.csv")
        with tab2:
            df_scans = db_fetch_table("scans")
            st.dataframe(df_scans, use_container_width=True)
            st.download_button("📂 EXPORT SCANS", df_scans.to_csv().encode('utf-8'), "rotex_scans.csv")

    elif menu == "R&D LAB":
        st.markdown("## 🔬 RESEARCH ARCHIVE")
        topic = st.selectbox("Topic", ["Sustainable Dyeing", "Smart Fabrics", "Recycled Polyester"])
        if st.button("Search"):
            for p in get_research_papers(topic): st.markdown(f'<div class="info-card"><a href="{p["url"]}">{p["title"]}</a></div>', unsafe_allow_html=True)
