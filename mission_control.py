import streamlit as st
import cv2
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import pydeck as pdk
import feedparser
import time

# --- 💀 PAGE CONFIGURATION ---
st.set_page_config(page_title="TEXTILE PREDATOR // COMMAND", layout="wide", page_icon="💀")

# --- 🎨 DARK MODE VISUALS ---
st.markdown("""
    <style>
    .big-font { font-size:30px !important; font-weight: bold; color: #FF4B4B; }
    </style>
    """, unsafe_allow_html=True)

# --- 🧠 MODULE 1: THE ORACLE ---
def load_market_data():
    tickers = ['CT=F', 'NG=F']
    try:
        data = yf.download(tickers, period="1y", interval="1d", progress=False)['Close']
        data.columns = ['Cotton_USD', 'Gas_USD']
        data = data.dropna()
    except:
        # Fallback if connection fails
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

# --- 👁️ MODULE 2: FABRIC EYE ---
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

# --- 🌍 MODULE 3: THE 3D GLOBE (NEW!) ---
def render_3d_map():
    # Target: Chittagong Port, Bangladesh
    target = [91.8, 22.3] 
    
    # Sources: The Global Cotton Belt
    sources = [
        {"name": "Texas, USA", "coords": [-99.9, 31.9], "color": [0, 255, 0]},   # Green
        {"name": "Sao Paulo, Brazil", "coords": [-46.6, -23.5], "color": [0, 128, 255]}, # Blue
        {"name": "Mumbai, India", "coords": [72.8, 19.0], "color": [255, 165, 0]}, # Orange
        {"name": "Queensland, Australia", "coords": [142.7, -20.9], "color": [255, 0, 255]} # Purple
    ]
    
    arc_data = []
    for s in sources:
        arc_data.append({
            "source": s["coords"],
            "target": target,
            "name": s["name"],
            "color": s["color"]
        })
    
    # The Flight Paths (Arcs)
    layer = pdk.Layer(
        "ArcLayer",
        data=arc_data,
        get_source_position="source",
        get_target_position="target",
        get_width=4,
        get_tilt=15,
        get_source_color="color",
        get_target_color="color",
        pickable=True,
        auto_highlight=True,
    )

    # The Map View
    view_state = pdk.ViewState(latitude=20, longitude=60, zoom=1, pitch=45)
    
    return pdk.Deck(
        layers=[layer], 
        initial_view_state=view_state, 
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={"text": "Shipment Origin: {name}"}
    )

# --- 🖥️ DASHBOARD UI ---
st.title("💀 TEXTILE PREDATOR // COMMAND v3.0")

# THE TABS
tab1, tab2, tab3 = st.tabs(["📈 WAR ROOM", "👁️ FABRIC EYE", "🌍 GLOBAL LOGISTICS (3D)"])

with tab1:
    st.markdown("### 📡 MARKET INTELLIGENCE")
    with st.spinner("Decrypting Market Data..."):
        df = load_market_data()
        preds = run_prediction(df)
    
    # Metrics
    curr = df['Yarn_Fair_Value'].iloc[-1]
    nxt = preds[-1]
    delta = ((nxt - curr)/curr)*100
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Yarn Fair Value", f"${curr:.2f}", f"{delta:.2f}% (7-Day)")
    c2.metric("Cotton (NYMEX)", f"${df['Cotton_USD'].iloc[-1]:.2f}", "Live")
    c3.metric("Gas (Henry Hub)", f"${df['Gas_USD'].iloc[-1]:.2f}", "Live")
    
    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Yarn_Fair_Value'], name='History', line=dict(color='#3498db')))
    future_dates = pd.date_range(start=df.index[-1], periods=8)[1:]
    fig.add_trace(go.Scatter(x=future_dates, y=preds, name='AI FORECAST', line=dict(color='#e74c3c', width=4, dash='dot')))
    fig.update_layout(height=350, template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    st.markdown("##### 🚨 LIVE THREATS")
    feeds = ["https://news.google.com/rss/search?q=Bangladesh+Textile+Industry+when:3d&hl=en-BD&gl=BD&ceid=BD:en"]
    for url in feeds:
        try:
            f = feedparser.parse(url)
            for entry in f.entries[:3]:
                st.info(f"**{entry.title}**")
        except: st.error("News Feed Offline")

with tab2:
    st.markdown("### 👁️ DEFECT SCANNER")
    uploaded_file = st.file_uploader("Upload Fabric Sample...", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        c1, c2 = st.columns(2)
        processed, count = process_fabric_image(uploaded_file)
        c1.image(uploaded_file, caption="Raw Input", use_column_width=True)
        c2.image(processed, caption=f"AI DETECTED: {count} FAULTS", use_column_width=True, channels="BGR")
        if count > 0: st.error(f"❌ REJECT LOT ({count} Defects)")
        else: st.success("✅ APPROVED")

with tab3:
    st.markdown("### 🌍 GLOBAL SUPPLY CHAIN")
    st.write("Visualizing Live Cotton Shipments to Chittagong Port.")
    
    # RENDER THE 3D GLOBE
    st.pydeck_chart(render_3d_map())
    
    st.caption("🟢 USA | 🔵 Brazil | 🟠 India | 🟣 Australia")
    
# --- 🔄 AUTO-REFRESH ---
st.divider()
if st.toggle("🔴 ACTIVATE WAR ROOM MODE (Auto-Update)", value=False):
    time.sleep(60)
    st.rerun()
