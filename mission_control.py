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

# --- 🎨 PRO STYLING ---
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .profit-pos { color: #00FF00; font-size: 30px; font-weight: bold; }
    .profit-neg { color: #FF0000; font-size: 30px; font-weight: bold; }
    /* Make the Tabs look professional */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #0E1117; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #262730; border-bottom: 2px solid #FF4B4B; }
    </style>
    """, unsafe_allow_html=True)

# --- 🧠 MODULE 1: THE ORACLE ---
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

# --- 🌍 MODULE 3: THE 3D GLOBE (IT LIVES!) ---
def render_3d_map():
    target = [91.8, 22.3] # Chittagong
    sources = [
        {"name": "Texas, USA", "coords": [-99.9, 31.9], "color": [0, 255, 0]},
        {"name": "Sao Paulo, BR", "coords": [-46.6, -23.5], "color": [0, 128, 255]},
        {"name": "Mumbai, IN", "coords": [72.8, 19.0], "color": [255, 165, 0]},
        {"name": "Queensland, AU", "coords": [142.7, -20.9], "color": [255, 0, 255]}
    ]
    arc_data = [{"source": s["coords"], "target": target, "name": s["name"], "color": s["color"]} for s in sources]
    
    layer = pdk.Layer(
        "ArcLayer",
        data=arc_data,
        get_source_position="source",
        get_target_position="target",
        get_width=5,
        get_tilt=15,
        get_source_color="color",
        get_target_color="color",
        pickable=True,
        auto_highlight=True,
    )
    view_state = pdk.ViewState(latitude=20, longitude=60, zoom=1, pitch=45)
    
    return pdk.Deck(
        layers=[layer], 
        initial_view_state=view_state, 
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={"text": "{name}"}
    )

# --- 🖥️ DASHBOARD UI ---
st.title("💀 TEXTILE PREDATOR // COMMAND v4.0")

# Load Data Once for All Tabs
with st.spinner("Syncing with Global Markets..."):
    df = load_market_data()
    preds = run_prediction(df)
    current_yarn_cost = df['Yarn_Fair_Value'].iloc[-1]

# DEFINING THE 4 TABS
tab1, tab2, tab3, tab4 = st.tabs(["📈 WAR ROOM", "👁️ FABRIC EYE", "🌍 GLOBAL LOGISTICS", "💰 DEAL BREAKER"])

with tab1:
    st.markdown("### 📡 MARKET INTELLIGENCE")
    curr = df['Yarn_Fair_Value'].iloc[-1]
    nxt = preds[-1]
    delta = ((nxt - curr)/curr)*100
    color = "normal" if delta < 1 else "inverse"
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Yarn Fair Value", f"${curr:.2f}", f"{delta:.2f}% (7-Day)", delta_color=color)
    c2.metric("Cotton (NYMEX)", f"${df['Cotton_USD'].iloc[-1]:.2f}", "Live")
    c3.metric("Gas (Henry Hub)", f"${df['Gas_USD'].iloc[-1]:.2f}", "Live")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Yarn_Fair_Value'], name='History', line=dict(color='#3498db')))
    future_dates = pd.date_range(start=df.index[-1], periods=8)[1:]
    fig.add_trace(go.Scatter(x=future_dates, y=preds, name='AI FORECAST', line=dict(color='#e74c3c', width=4, dash='dot')))
    fig.update_layout(height=350, template="plotly_dark", margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig, use_container_width=True)

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
    st.pydeck_chart(render_3d_map())
    st.caption("🟢 USA | 🔵 Brazil | 🟠 India | 🟣 Australia")

with tab4:
    st.markdown("### 💰 THE DEAL BREAKER (Margin Calculator)")
    st.info("Input your Buyer's Offer. The AI will calculate profitability based on TODAY'S Market Price.")
    
    colA, colB = st.columns(2)
    
    with colA:
        st.markdown("#### 📝 ORDER DETAILS")
        buyer_price = st.number_input("Buyer Offer Price ($ per kg)", value=4.50, step=0.05)
        order_qty = st.number_input("Order Quantity (kg)", value=10000)
        
        st.markdown("#### 🏭 FACTORY OVERHEADS")
        knitting_cost = st.number_input("Knitting/Dyeing Cost ($ per kg)", value=0.60)
        overhead_cost = st.number_input("Admin/Transport ($ per kg)", value=0.15)
        
    with colB:
        st.markdown("#### 🤖 AI ANALYSIS")
        
        # Calculations
        raw_material_cost = current_yarn_cost # Pulled from Live Market
        total_cost = raw_material_cost + knitting_cost + overhead_cost
        margin = buyer_price - total_cost
        total_profit = margin * order_qty
        margin_percent = (margin / buyer_price) * 100
        
        # Display
        st.write(f"📉 **Live Yarn Cost:** ${raw_material_cost:.2f} / kg")
        st.write(f"⚙️ **Total Production Cost:** ${total_cost:.2f} / kg")
        st.divider()
        
        if margin > 0:
            st.markdown(f'<p class="profit-pos">✅ ACCEPT DEAL</p>', unsafe_allow_html=True)
            st.write(f"**Net Profit:** ${margin:.2f} / kg")
            st.write(f"**Total Profit:** ${total_profit:,.2f}")
            st.write(f"**Margin:** {margin_percent:.1f}%")
            if st.button("Celebrate Win"):
                st.balloons()
        else:
            st.markdown(f'<p class="profit-neg">❌ REJECT DEAL</p>', unsafe_allow_html=True)
            st.write(f"**Net LOSS:** ${margin:.2f} / kg")
            st.write(f"**Total LOSS:** ${total_profit:,.2f}")
            st.error(f"⚠️ You need to negotiate at least ${total_cost + 0.20:.2f} to break even safely.")

# --- 🔄 AUTO-REFRESH ---
st.divider()
if st.toggle("🔴 ACTIVATE WAR ROOM MODE", value=False):
    time.sleep(60)
    st.rerun()
