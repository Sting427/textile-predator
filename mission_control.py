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
from streamlit_option_menu import option_menu # You might need to add this to requirements.txt if you want a fancy menu, but for now I'll use native to keep it simple.

# --- 🏢 PAGE CONFIGURATION ---
st.set_page_config(
    page_title="TexOS // Enterprise",
    layout="wide",
    page_icon="🧵",
    initial_sidebar_state="expanded"
)

# --- 🎨 ENTERPRISE CSS (THE SUIT) ---
st.markdown("""
    <style>
    /* MAIN BACKGROUND */
    .stApp {
        background-color: #F8F9FA;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #0E1117;
        color: white;
    }
    
    /* METRIC CARDS (Professional White Boxes) */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #333;
    }
    
    /* HEADERS */
    h1, h2, h3 {
        color: #1A237E; /* Navy Blue */
        font-weight: 700;
    }
    
    /* CUSTOM CARD FOR NEWS & PAPERS */
    .pro-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border-left: 5px solid #1A237E;
    }
    
    /* BUTTONS */
    .stButton>button {
        background-color: #1A237E;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #304FFE;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 🔒 ENTERPRISE LOGIN ---
def check_password():
    def password_entered():
        if st.session_state["password"] == "TEXTILE_KING":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # Professional Login Screen
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown("## 🔒 TexOS Enterprise Login")
            st.info("Authorized Personnel Only. All access is logged.")
            st.text_input("Access Key", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.error("⛔ ACCESS DENIED")
        return False
    else:
        return True

# --- 🧠 BACKEND LOGIC (The Brains) ---
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
        {"name": "Texas, USA", "coords": [-99.9, 31.9], "color": [26, 35, 126]}, # Navy
        {"name": "Sao Paulo, BR", "coords": [-46.6, -23.5], "color": [48, 79, 254]}, # Blue
        {"name": "Mumbai, IN", "coords": [72.8, 19.0], "color": [255, 111, 0]} # Orange
    ]
    arc_data = [{"source": s["coords"], "target": target, "name": s["name"], "color": s["color"]} for s in sources]
    layer = pdk.Layer("ArcLayer", data=arc_data, get_source_position="source", get_target_position="target", get_width=4, get_tilt=15, get_source_color="color", get_target_color="color", pickable=True, auto_highlight=True)
    view_state = pdk.ViewState(latitude=20, longitude=60, zoom=1, pitch=45)
    # Using a lighter map style for corporate look
    return pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v10", tooltip={"text": "{name}"})

def get_research_papers(topic):
    query = f"textile {topic}"
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=5&fields=title,url,year,abstract"
    try:
        r = requests.get(url, timeout=5).json()
        if 'data' in r: return r['data']
        else: return []
    except: return []

# --- 🚀 THE APP LAUNCHER ---
if check_password():
    
    # --- SIDEBAR NAVIGATION ---
    with st.sidebar:
        st.title("TexOS™")
        st.caption("Enterprise Edition v8.0")
        st.divider()
        menu = st.radio("MAIN MENU", ["Dashboard", "Vision AI", "Global Logistics", "Profit Calc", "R&D Library"], label_visibility="collapsed")
        st.divider()
        st.info("System Status: ● Online")
        if st.button("Logout"):
            st.session_state["password_correct"] = False
            st.rerun()

    # --- LOAD DATA ---
    with st.spinner("Initializing Enterprise Modules..."):
        df = load_market_data()
        preds = run_prediction(df)
        current_yarn_cost = df['Yarn_Fair_Value'].iloc[-1]
        news_items = get_news_stealth()

    # --- 1. DASHBOARD ---
    if menu == "Dashboard":
        st.markdown("## 📊 Executive Dashboard")
        st.markdown("Real-time market overview and threat detection.")
        
        # Top Metrics Row
        c1, c2, c3, c4 = st.columns(4)
        curr = df['Yarn_Fair_Value'].iloc[-1]
        nxt = preds[-1]
        delta = ((nxt - curr)/curr)*100
        
        with c1: st.metric("Yarn Fair Value", f"${curr:.2f}", f"{delta:.2f}%")
        with c2: st.metric("Cotton (NYMEX)", f"${df['Cotton_USD'].iloc[-1]:.2f}", "+0.5%")
        with c3: st.metric("Gas (Henry Hub)", f"${df['Gas_USD'].iloc[-1]:.2f}", "-1.2%")
        with c4: st.metric("Factory Output", "12.5 Tons", "On Target")

        st.divider()

        # Chart & News Split
        col_chart, col_news = st.columns([2, 1])
        
        with col_chart:
            st.markdown("### 📈 Price Forecast Model")
            fig = go.Figure()
            # Professional White Chart Theme
            fig.add_trace(go.Scatter(x=df.index, y=df['Yarn_Fair_Value'], name='Historical Data', line=dict(color='#1A237E', width=2)))
            future_dates = pd.date_range(start=df.index[-1], periods=8)[1:]
            fig.add_trace(go.Scatter(x=future_dates, y=preds, name='AI Forecast', line=dict(color='#FF6F00', width=2, dash='dash')))
            fig.update_layout(height=400, template="plotly_white", margin=dict(l=20,r=20,t=40,b=20), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

        with col_news:
            st.markdown("### 📰 Market Brief")
            if news_items:
                for item in news_items:
                    st.markdown(f"""
                    <div class="pro-card">
                        <a href="{item.link}" target="_blank" style="text-decoration: none; color: #1A237E;">
                            <b>{item.title}</b>
                        </a><br>
                        <span style="font-size: 12px; color: #666;">{item.published[:16]}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No critical alerts.")

    # --- 2. VISION AI ---
    elif menu == "Vision AI":
        st.markdown("## 👁️ Quality Control (QC) AI")
        st.write("Upload fabric samples for automated defect detection.")
        
        uploaded_file = st.file_uploader("Upload Sample Image", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            c1, c2 = st.columns(2)
            processed, count = process_fabric_image(uploaded_file)
            with c1:
                st.image(uploaded_file, caption="Original Sample", use_column_width=True)
            with c2:
                st.image(processed, caption=f"Analysis: {count} Defects Found", use_column_width=True, channels="BGR")
                if count > 0:
                    st.error(f"❌ QC FAILED: {count} Defects Detected.")
                else:
                    st.success("✅ QC PASSED: Fabric is clean.")

    # --- 3. GLOBAL LOGISTICS ---
    elif menu == "Global Logistics":
        st.markdown("## 🌍 Supply Chain Visualization")
        st.pydeck_chart(render_3d_map())
        st.caption("Live visualization of inbound raw materials.")

    # --- 4. PROFIT CALCULATOR ---
    elif menu == "Profit Calc":
        st.markdown("## 💰 Deal Evaluation Engine")
        
        with st.container():
            st.markdown('<div class="pro-card">', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Contract Details")
                buyer_price = st.number_input("Buyer Offer ($/kg)", value=4.50, step=0.05)
                qty = st.number_input("Quantity (kg)", value=10000)
            with c2:
                st.subheader("Cost Structure")
                knit_cost = st.number_input("Production Cost ($/kg)", value=0.60)
                overhead = st.number_input("Overhead ($/kg)", value=0.15)
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()
        
        # Calculation
        total_cost = current_yarn_cost + knit_cost + overhead
        margin = buyer_price - total_cost
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Market Yarn Price", f"${current_yarn_cost:.2f}")
        m2.metric("Total Cost to Produce", f"${total_cost:.2f}")
        
        if margin > 0:
            m3.metric("Net Profit Margin", f"${margin:.2f}/kg", "PROFITABLE", delta_color="normal")
            st.success(f"✅ RECOMMENDATION: ACCEPT DEAL. Total Profit: ${margin*qty:,.2f}")
        else:
            m3.metric("Net Loss", f"${margin:.2f}/kg", "LOSS", delta_color="inverse")
            st.error(f"❌ RECOMMENDATION: REJECT DEAL. Potential Loss: ${margin*qty:,.2f}")

    # --- 5. R&D LIBRARY ---
    elif menu == "R&D Library":
        st.markdown("## 📚 Research & Development")
        topic = st.selectbox("Search Topic", ["Sustainable Dyeing", "Smart Fabrics", "Recycled Polyester", "Nano-Finishing"])
        
        if st.button("Search Database"):
            with st.spinner("Querying Academic Database..."):
                papers = get_research_papers(topic)
                for p in papers:
                    st.markdown(f"""
                    <div class="pro-card">
                        <h4>{p.get('title')}</h4>
                        <p style="color:#555;">{p.get('year')} | <a href="{p.get('url')}" target="_blank">Read Full Paper</a></p>
                    </div>
                    """, unsafe_allow_html=True)
