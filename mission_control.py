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
from datetime import datetime

# --- 🌑 PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ROTex // ARCHIVE",
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
    .rotex-logo-container { text-align: center; margin-bottom: 20px; }
    .rotex-text { font-family: 'Rajdhani', sans-serif; font-weight: 800; letter-spacing: 4px; text-transform: uppercase; }
    .ro-cyan { color: #00d2ff; text-shadow: 0 0 25px rgba(0, 210, 255, 0.6); }
    .tex-magenta { color: #ff0055; text-shadow: 0 0 25px rgba(255, 0, 85, 0.6); }
    .login-box { background: rgba(14, 17, 23, 0.8); border: 1px solid #333; padding: 40px; border-radius: 15px; text-align: center; border-top: 2px solid #00d2ff; border-bottom: 2px solid #ff0055; }
    </style>
    """, unsafe_allow_html=True)

# --- 💾 DATA PERSISTENCE (THE BLACK BOX) ---
DB_FILE = "rotex_ledger.csv"

def log_transaction(buyer, qty, price, cost, margin):
    new_data = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M"), buyer, qty, price, cost, margin]], 
                            columns=["Timestamp", "Buyer", "Qty (kg)", "Price ($)", "Cost ($)", "Margin ($)"])
    if not os.path.isfile(DB_FILE):
        new_data.to_csv(DB_FILE, index=False)
    else:
        new_data.to_csv(DB_FILE, mode='a', header=False, index=False)

# --- 🔒 SECURITY SYSTEM ---
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
            st.markdown('<div class="login-box"><div class="rotex-logo-container"><span class="rotex-text ro-cyan" style="font-size: 60px;">RO</span><span class="rotex-text tex-magenta" style="font-size: 60px;">Tex</span><br></div></div>', unsafe_allow_html=True)
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
        return pd.DataFrame({'Cotton_USD': 85, 'Gas_USD': 3, 'Yarn_Fair_Value': 4.1}, index=dates)

def get_jobs_stealth():
    try: return feedparser.parse(requests.get("https://news.google.com/rss/search?q=Textile+Job+Vacancy+Bangladesh&hl=en-BD&gl=BD&ceid=BD:en").content).entries[:5]
    except: return []

# --- 🚀 LAUNCH SEQUENCE ---
if check_password():
    with st.sidebar:
        st.markdown('<div class="rotex-logo-container"><span class="rotex-text ro-cyan" style="font-size: 48px;">RO</span><span class="rotex-text tex-magenta" style="font-size: 48px;">Tex</span></div>', unsafe_allow_html=True)
        menu = st.radio("SENSORS", ["WAR ROOM", "FACTORY IoT", "DEAL BREAKER", "RECRUITMENT", "LEDGER", "VISION AI"])
        if st.button("TERMINATE"):
            st.session_state["password_correct"] = False
            st.rerun()

    df = load_market_data()
    current_yarn_cost = df['Yarn_Fair_Value'].iloc[-1]

    if menu == "WAR ROOM":
        st.markdown("## 📡 GLOBAL MARKET COMMAND")
        c1, c2, c3 = st.columns(3)
        c1.metric("YARN FAIR VALUE", f"${current_yarn_cost:.2f}")
        c2.metric("COTTON", f"${df['Cotton_USD'].iloc[-1]:.2f}")
        c3.metric("GAS", f"${df['Gas_USD'].iloc[-1]:.2f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Yarn_Fair_Value'], line=dict(color='#00d2ff', width=3)))
        fig.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    elif menu == "DEAL BREAKER":
        st.markdown("## 💰 MARGIN CALCULATOR")
        colA, colB = st.columns(2)
        with colA:
            buyer = st.text_input("BUYER NAME", "Global Brands Inc.")
            price = st.number_input("OFFER ($/kg)", value=4.50)
            qty = st.number_input("ORDER QTY (kg)", value=5000)
        with colB:
            total_cost = current_yarn_cost + 0.75 # Simple fixed prod cost
            margin = price - total_cost
            st.metric("NET MARGIN", f"${margin:.2f}/kg", delta=f"{margin*qty:,.0f} Total")
            if st.button("💾 SAVE TO LEDGER"):
                log_transaction(buyer, qty, price, total_cost, margin)
                st.success("TRANSACTION SECURED IN BLACK BOX.")

    elif menu == "LEDGER":
        st.markdown("## 📖 AUDIT LOG // HISTORY")
        if os.path.isfile(DB_FILE):
            history = pd.read_csv(DB_FILE)
            st.dataframe(history, use_container_width=True)
            st.download_button("📂 EXCEL EXPORT", history.to_csv().encode('utf-8'), "rotex_ledger.csv", "text/csv")
        else:
            st.info("Ledger is empty. Process your first deal in the Deal Breaker.")

    elif menu == "FACTORY IoT":
        st.markdown("## 🏭 LIVE SENSOR TELEMETRY")
        k1, k2 = st.columns(2)
        k1.metric("LOOM SPEED", f"{random.randint(790, 810)} RPM")
        k2.metric("HUMIDITY", f"{random.randint(62, 68)}%")
        st.line_chart(np.random.randn(20, 2))

    elif menu == "RECRUITMENT":
        st.markdown("## 🤝 LIVE RECRUITMENT")
        for job in get_jobs_stealth():
            st.markdown(f'<div style="border-left: 3px solid #00ff88; padding-left:10px;"><b>{job.title}</b><br><a href="{job.link}">Apply</a></div>', unsafe_allow_html=True)
            st.divider()

    elif menu == "VISION AI":
        st.markdown("## 👁️ DEFECT DETECTION")
        up = st.file_uploader("Upload Fabric")
        if up: st.image(up)
