import streamlit as st
import cv2
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import feedparser
import datetime

# --- 💀 PAGE CONFIGURATION (Wide Mode) ---
st.set_page_config(page_title="TEXTILE PREDATOR // COMMAND", layout="wide", page_icon="💀")

# --- 🎨 CUSTOM CSS (The "Iron Man" Look) ---
st.markdown("""
    <style>
    .big-font { font-size:30px !important; font-weight: bold; color: #FF4B4B; }
    .metric-card { background-color: #0E1117; border: 1px solid #303030; padding: 20px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 🧠 MODULE 1: THE ORACLE (Financials) ---
def load_market_data():
    tickers = ['CT=F', 'NG=F']
    data = yf.download(tickers, period="1y", interval="1d", progress=False)['Close']
    data.columns = ['Cotton_USD', 'Gas_USD']
    data = data.dropna()
    
    # Calculate Yarn Fair Value
    cotton_in_dollars = data['Cotton_USD'] / 100 
    data['Yarn_Fair_Value'] = (cotton_in_dollars * 1.6) + (data['Gas_USD'] * 0.15) + 0.40
    return data

def run_prediction(df):
    X = df[['Cotton_USD', 'Gas_USD']]
    y = df['Yarn_Fair_Value']
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next 7 days
    last_cotton = df['Cotton_USD'].iloc[-1]
    last_gas = df['Gas_USD'].iloc[-1]
    
    future_cotton = [last_cotton * (1 + (0.002 * i)) for i in range(7)]
    future_gas = [last_gas * (1 + (0.005 * i)) for i in range(7)]
    
    future_df = pd.DataFrame({'Cotton_USD': future_cotton, 'Gas_USD': future_gas})
    predictions = model.predict(future_df)
    
    return predictions

# --- 👁️ MODULE 2: THE FABRIC EYE (Computer Vision) ---
def process_fabric_image(image_file):
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Blur to remove noise (dust/lint)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Thresholding (Find dark spots on light fabric)
    # This automatically finds the contrast difference
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Find Contours (The Defects)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Draw Boxes around Defects
    defect_count = 0
    output_img = img.copy()
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > 50: # Ignore tiny specks (< 50 pixels)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(output_img, "DEFECT", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            defect_count += 1
            
    return output_img, defect_count

# --- 🖥️ THE DASHBOARD INTERFACE ---
st.title("💀 TEXTILE PREDATOR // MISSION CONTROL")

# TABS
tab1, tab2 = st.tabs(["📈 WAR ROOM (Market Intel)", "👁️ FABRIC EYE (Defect Scanner)"])

with tab1:
    st.markdown("### 🌍 GLOBAL MARKET STATUS")
    
    # Load Data
    with st.spinner("Contacting Wall Street..."):
        df = load_market_data()
        preds = run_prediction(df)
        
    # Metrics Row
    curr_price = df['Yarn_Fair_Value'].iloc[-1]
    next_price = preds[-1]
    delta = next_price - curr_price
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Yarn Fair Value (Today)", f"${curr_price:.2f}", f"{delta:.2f} (7-Day Forecast)")
    c2.metric("Cotton Futures (NYMEX)", f"${df['Cotton_USD'].iloc[-1]:.2f}", "Live")
    c3.metric("Gas Futures (Henry Hub)", f"${df['Gas_USD'].iloc[-1]:.2f}", "Live")
    
    # Interactive Chart
    st.subheader("🔮 STRATEGIC FORECAST MODEL")
    
    # Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Yarn_Fair_Value'], name='Historical Cost', line=dict(color='gray')))
    
    # Future Dates
    future_dates = pd.date_range(start=df.index[-1], periods=8)[1:]
    fig.add_trace(go.Scatter(x=future_dates, y=preds, name='AI PREDICTION', line=dict(color='red', width=4, dash='dot')))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # News Feed
    st.subheader("🚨 LIVE THREAT INTELLIGENCE")
    feeds = ["https://news.google.com/rss/search?q=Bangladesh+Textile+Industry+when:3d&hl=en-BD&gl=BD&ceid=BD:en"]
    for url in feeds:
        f = feedparser.parse(url)
        for entry in f.entries[:5]:
            st.warning(f"**{entry.title}** ([Read Source]({entry.link}))")

with tab2:
    st.markdown("### 👁️ AUTOMATED QUALITY CONTROL")
    st.write("Upload a photo of fabric. The AI will scan for holes, oil stains, and knitting faults.")
    
    uploaded_file = st.file_uploader("Drop Fabric Image Here...", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        c1, c2 = st.columns(2)
        
        # Process
        processed_img, count = process_fabric_image(uploaded_file)
        
        # Display Results
        c1.image(uploaded_file, caption="Original Sample", use_column_width=True)
        c2.image(processed_img, caption=f"AI SCANNED: {count} DEFECTS FOUND", use_column_width=True, channels="BGR")
        
        if count > 0:
            st.error(f"❌ REJECT: {count} Defects Detected.")
        else:
            st.success("✅ PASS: No Visible Defects.")
            # --- 🔄 THE HEARTBEAT (AUTO-REFRESH) ---
import time

st.divider() # Draw a line to separate the footer

# The Toggle Switch
if st.toggle("🔴 ACTIVATE LIVE MODE (Auto-Refresh every 60s)", value=False):
    st.toast("⏳ Refreshing in 60 seconds...")
    time.sleep(60)
    st.rerun()
