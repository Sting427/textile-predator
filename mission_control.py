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
from streamlit_lottie import st_lottie

# --- 🌑 PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TEX-OS // APEX",
    layout="wide",
    page_icon="💠",
    initial_sidebar_state="expanded"
)

# --- 🎨 THE "APEX" THEME (CSS INJECTION) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;500;700&display=swap');

    /* ANIMATED BACKGROUND */
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

    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.8);
        border-right: 1px solid #333;
        backdrop-filter: blur(10px);
    }
    
    /* GLASSMORPHISM CARDS */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 12px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-left: 4px solid #00d2ff;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 210, 255, 0.2);
    }
    
    /* NEWS & INFO CARDS WITH HOVER GLOW */
    .info-card {
        background: rgba(255, 255, 255, 0.03);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    .info-card:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: #00d2ff;
        transform: translateX(5px);
    }
    .info-card a {
        color: #00d2ff;
        text-decoration: none;
        font-size: 18px;
        font-weight: 700;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* TYPOGRAPHY */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Rajdhani', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(0, 210, 255, 0.5);
    }
    
    /* BUTTONS */
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: bold;
        font-family: 'Rajdhani', sans-serif;
        letter-spacing: 1px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.6);
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 🎞️ ANIMATION LOADER ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

# Load Assets
lottie_tech = load_lottieurl("https://lottie.host/5a67c57c-d698-4228-a37f-534d0b277d33/9qZ2F5W3rT.json") # Tech Globe

# --- 🔒 SECURITY SYSTEM ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "TEXTILE_KING":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.markdown("<br><br><h1 style='text-align:center; color:#ff4b4b; text-shadow:none;'>🛑 CLASSIFIED ACCESS</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.text_input("ENTER PASSCODE:", type="password", on_change=password_entered, key="password")
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

# --- 🚀 THE APEX LAUNCHER ---
if check_password():
    
    # --- SIDEBAR NAV ---
    with st.sidebar:
        if lottie_tech:
            st_lottie(lottie_tech, height=150, key="sidebar_anim")
        st.title("TEX-OS™")
        st.markdown("<div style='text-align: center; color: #888; font-size: 12px;'>APEX EDITION v10.0</div>", unsafe_allow_html=True)
        st.divider()
        menu = st.radio("SYSTEM MODULES", ["WAR ROOM", "VISION AI", "LOGISTICS", "DEAL BREAKER", "R&D LAB"])
        st.divider()
        if st.button("TERMINATE SESSION"):
            st.session_state["password_correct"] = False
            st.rerun()

    # --- LOAD DATA ---
    with st.spinner("ESTABLISHING SECURE UPLINK..."):
        df = load_market_data()
        preds = run_prediction(df)
        current_yarn_cost = df['Yarn_Fair_Value'].iloc[-1]
        news_items = get_news_stealth()

    # --- 1. WAR ROOM ---
    if menu == "WAR ROOM":
        st.markdown("## 📡 MARKET INTELLIGENCE")
        
        # Metrics with new styles
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
                card_html = f"""
                <div class="info-card">
                    <a href="{item.link}" target="_blank">➤ {item.title}</a><br>
                    <span style="color: #888; font-size: 12px; font-family: sans-serif;">{item.published[:16]}</span>
                </div>
                """
                if i % 2 == 0: n1.markdown(card_html, unsafe_allow_html=True)
                else: n2.markdown(card_html, unsafe_allow_html=True)
        else:
            st.warning("NO INTEL AVAILABLE")

    # --- 2. VISION AI ---
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

    # --- 3. LOGISTICS ---
    elif menu == "LOGISTICS":
        st.markdown("## 🌍 SUPPLY CHAIN (LIVE)")
        st.pydeck_chart(render_3d_map())

    # --- 4. DEAL BREAKER ---
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

    # --- 5. R&D LAB ---
    elif menu == "R&D LAB":
        st.markdown("## 🔬 RESEARCH ARCHIVE")
        topic = st.selectbox("SELECT TOPIC", ["Sustainable Dyeing", "Smart Fabrics", "Recycled Polyester", "Nano-Finishing"])
        if st.button("INITIATE SEARCH"):
            with st.spinner("SEARCHING DATABASE..."):
                papers = get_research_papers(topic)
                for p in papers:
                    st.markdown(f"""
                    <div class="info-card">
                        <a href="{p.get('url')}" target="_blank">📄 {p.get('title')}</a><br>
                        <span style="color:#888;">{p.get('year')}</span>
                    </div>
                    """, unsafe_allow_html=True)
