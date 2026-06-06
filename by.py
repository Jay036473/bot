import atexit
import numpy as np
import streamlit as st
import pandas as pd
import MetaTrader5 as mt5
import plotly.graph_objects as go
import datetime
import time
import os
import requests
import warnings
import streamlit.components.v1 as components

# --- ARIMA ---
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller

    STATSMODELS_OK = True
except ImportError:
    STATSMODELS_OK = False
    st.error("statsmodels missing. Run: pip install statsmodels")

# --- AUTO REFRESH ---
try:
    from streamlit_autorefresh import st_autorefresh

    AUTO_REFRESH_AVAILABLE = True
except ImportError:
    AUTO_REFRESH_AVAILABLE = False

warnings.filterwarnings("ignore")

# =============================================================
# AI CONFIGURATION
# =============================================================
DEFAULT_GEMINI_API_KEY = ""
AUTO_REFRESH_MS = 300_000
AUTO_REFRESH_SECONDS = 300

# =============================================================
# PAGE CONFIG & CSS
# =============================================================
st.set_page_config(page_title="Forex AI Pro v9", layout="wide", page_icon="🚀")


def apply_css():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@700;800&display=swap');
    .stApp{background-color:#060D1F;color:#F8FAFC;}
    h1,h2,h3,h4{color:#E2E8F0!important;font-family:'Syne',sans-serif;font-weight:800;}
    .stButton>button{
        background:linear-gradient(135deg,#6366F1 0%,#4F46E5 100%);
        color:white;border-radius:8px;border:none;
        padding:0.6rem 1.2rem;font-weight:600;transition:all 0.3s ease;width:100%;}
    .stButton>button:hover{transform:translateY(-2px);
        box-shadow:0 4px 20px rgba(99,102,241,0.5);color:white;border:none;}
    div[data-testid="metric-container"]{
        background-color:#0F1A2E;border-radius:12px;padding:20px;
        border:1px solid #1E3A5F;text-align:center;}
    [data-testid="stMetricValue"]{font-size:2.2rem;font-weight:800;color:#38BDF8;}
    .stTextInput input,.stNumberInput input{
        background-color:#0F1A2E!important;color:white!important;
        border:1px solid #1E3A5F!important;border-radius:8px!important;}
    div[data-baseweb="select"]>div{
        background-color:#0F1A2E;border-radius:8px;
        border:1px solid #1E3A5F;color:white;}
    [data-testid="stDataFrame"]{border-radius:10px;overflow:hidden;border:1px solid #1E3A5F;}
    hr{border-top:1px solid #1E3A5F;}
    .stAlert{border-radius:8px;border:none;}
    .model-badge{display:inline-block;padding:4px 12px;border-radius:20px;
        font-size:0.78rem;font-weight:700;margin:2px;}
    .market-closed-banner{
        background:linear-gradient(135deg,#1A1F35,#0D1117);
        border:1px solid #F59E0B;border-radius:12px;
        padding:16px 24px;text-align:center;margin-bottom:16px;}
    .price-action-box{
        background:#0F1A2E;border:1px solid #1E3A5F;
        border-radius:12px;padding:14px 18px;margin-top:8px;
        font-size:0.83rem;color:#94A3B8;}
    .pa-item{display:flex;justify-content:space-between;
        padding:4px 0;border-bottom:1px solid #1E293B;}
    .pa-bull{color:#10B981;font-weight:700;}
    .pa-bear{color:#EF4444;font-weight:700;}
    .pa-neutral{color:#94A3B8;font-weight:600;}
    .suggestion-box{
        background:linear-gradient(145deg, #0F172A 0%, #060D1F 100%);
        border: 1px solid #1E3A5F; border-radius: 16px; padding: 24px;
        margin-top: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);}
    .sug-title{font-size:1.2rem; font-weight:800; color:#E2E8F0; margin-bottom:15px;}
    .summary-box{
        background:linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border-left: 5px solid #38BDF8; border-radius: 12px;
        padding: 20px; margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);}
    .summary-title{font-size:1.1rem; font-weight:800; color:#38BDF8; margin-bottom:10px;}
    .summary-text{color:#CBD5E1; font-size:0.92rem; line-height:1.6;}
    .summary-bold{color:#F8FAFC; font-weight:700;}
    .asset-nav-label{
        color:#475569;font-size:0.72rem;font-weight:700;
        text-transform:uppercase;letter-spacing:0.12em;
        margin-right:6px;font-family:'Space Mono',monospace;}
    .metal-hero-card{border-radius:20px;padding:22px 28px;margin-bottom:20px;
        display:flex;align-items:center;gap:22px;position:relative;overflow:hidden;}
    .metal-hero-gold{
        background:linear-gradient(135deg,#1C1007 0%,#2D1A00 50%,#1A0F00 100%);
        border:2px solid #B45309;
        box-shadow:0 8px 32px rgba(180,83,9,0.25),inset 0 1px 0 rgba(245,158,11,0.15);}
    .metal-hero-silver{
        background:linear-gradient(135deg,#0E1219 0%,#182030 50%,#0A0F18 100%);
        border:2px solid #475569;
        box-shadow:0 8px 32px rgba(71,85,105,0.3),inset 0 1px 0 rgba(148,163,184,0.1);}
    .metal-hero-bitcoin{
        background:linear-gradient(135deg,#1A0F00 0%,#2D1500 50%,#1A0900 100%);
        border:2px solid #F97316;
        box-shadow:0 8px 32px rgba(249,115,22,0.25),inset 0 1px 0 rgba(251,146,60,0.15);}
    .metal-hero-icon{font-size:3.8rem;line-height:1;filter:drop-shadow(0 4px 12px rgba(0,0,0,0.5));}
    .metal-hero-title{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;line-height:1.1;margin-bottom:4px;}
    .metal-hero-subtitle{font-size:0.82rem;color:#64748B;font-family:'Space Mono',monospace;letter-spacing:0.05em;}
    .badge-gold{background:rgba(245,158,11,0.15);border:1px solid #B45309;color:#FCD34D;
        padding:6px 16px;border-radius:50px;font-size:0.75rem;font-weight:700;font-family:'Space Mono',monospace;}
    .badge-silver{background:rgba(148,163,184,0.12);border:1px solid #475569;color:#CBD5E1;
        padding:6px 16px;border-radius:50px;font-size:0.75rem;font-weight:700;font-family:'Space Mono',monospace;}
    .badge-bitcoin{background:rgba(249,115,22,0.15);border:1px solid #C2410C;color:#FDBA74;
        padding:6px 16px;border-radius:50px;font-size:0.75rem;font-weight:700;font-family:'Space Mono',monospace;}
    .rr-advisor-card{
        background:linear-gradient(145deg,#0A1628 0%,#0F1E38 100%);
        border:2px solid #1E3A5F;border-radius:20px;
        padding:24px;margin-top:16px;
        box-shadow:0 10px 40px rgba(0,0,0,0.5),
                   inset 0 1px 0 rgba(56,189,248,0.08);
        position:relative;overflow:hidden;}
    .rr-advisor-card::before{
        content:'';position:absolute;top:-40%;right:-10%;
        width:200px;height:200px;
        background:radial-gradient(circle,rgba(56,189,248,0.05) 0%,transparent 70%);
        pointer-events:none;}
    .rr-advisor-title{
        font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:800;
        color:#E2E8F0;margin-bottom:18px;
        display:flex;align-items:center;gap:10px;}
    .rr-main-display{
        text-align:center;padding:20px;
        background:linear-gradient(135deg,rgba(99,102,241,0.08),rgba(56,189,248,0.06));
        border-radius:16px;border:1px solid rgba(99,102,241,0.2);
        margin-bottom:18px;}
    .rr-ratio-grade-S{color:#10B981;}
    .rr-ratio-grade-A{color:#34D399;}
    .rr-ratio-grade-B{color:#38BDF8;}
    .rr-ratio-grade-C{color:#F59E0B;}
    .rr-ratio-grade-D{color:#EF4444;}
    .rr-factor-row{
        display:flex;align-items:center;justify-content:space-between;
        padding:9px 0;border-bottom:1px solid #1E293B;}
    .rr-factor-label{color:#64748B;font-size:0.8rem;font-weight:600;}
    .rr-factor-bar-track{
        width:90px;height:6px;background:#1E293B;
        border-radius:4px;overflow:hidden;display:inline-block;margin:0 8px;}
    .rr-factor-bar-fill{height:100%;border-radius:4px;}
    .rr-factor-value{font-family:'Space Mono',monospace;font-size:0.78rem;font-weight:700;}
    .rr-rule-box{
        background:rgba(56,189,248,0.04);border:1px solid rgba(56,189,248,0.12);
        border-radius:10px;padding:12px 16px;margin-top:14px;
        font-size:0.78rem;color:#94A3B8;line-height:1.7;}
    .rr-rule-item{display:flex;gap:8px;align-items:flex-start;margin-bottom:4px;}

    /* NEW FINAL ACTION BOX CSS */
    .final-action-box {
        background: linear-gradient(145deg, #0A1628 0%, #060D1F 100%);
        border: 2px solid #38BDF8;
        border-radius: 16px;
        padding: 24px;
        margin-top: 16px;
        margin-bottom: 16px;
        box-shadow: 0 10px 40px rgba(56,189,248,0.15);
        position: relative;
        overflow: hidden;
    }
    .final-action-title {
        font-family: 'Syne', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        color: #38BDF8;
        margin-bottom: 15px;
        text-align: center;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .final-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
    }
    .final-card {
        background: #0F1A2E;
        border: 1px solid #1E3A5F;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }
    .final-lbl {
        font-size: 0.75rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 5px;
        font-weight: 700;
    }
    .final-val {
        font-family: 'Space Mono', monospace;
        font-size: 1.25rem;
        font-weight: 800;
        color: #F8FAFC;
    }
    .final-instruction {
        background: rgba(56,189,248,0.08);
        border: 1px dashed rgba(56,189,248,0.3);
        border-radius: 10px;
        padding: 15px;
        margin-top: 15px;
        color: #E2E8F0;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    </style>""", unsafe_allow_html=True)


apply_css()

# =============================================================
# MT5 INITIALIZATION
# =============================================================
initialized = mt5.initialize()
if not initialized:
    initialized = mt5.initialize(r"C:\Program Files\MetaTrader 5\terminal64.exe")
if not initialized:
    st.error("Failed to initialize MetaTrader 5.")
    st.error(f"Error: {mt5.last_error()}")
    st.stop()

atexit.register(mt5.shutdown)

# =============================================================
# SESSION STATE
# =============================================================
defaults = {
    "page": "main",
    "asset_class": "Forex",
    "selected_pair": "EURUSD",
    "last_auto_refresh": time.time(),
    "last_notified_key": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============================================================
# GLOBAL CONSTANTS
# =============================================================
TIMEFRAME_DICT = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
}

JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "NZDJPY", "CHFJPY"}
METALS = {"XAUUSD", "XAGUSD"}
CRYPTO = {"BTCUSD"}

PAIR_CURRENCIES = {
    "EURUSD": ["EUR", "USD"], "GBPUSD": ["GBP", "USD"], "USDJPY": ["USD", "JPY"],
    "USDCHF": ["USD", "CHF"], "AUDUSD": ["AUD", "USD"], "USDCAD": ["USD", "CAD"],
    "NZDUSD": ["NZD", "USD"],
    "XAUUSD": ["XAU", "USD", "gold", "federal reserve", "inflation"],
    "XAGUSD": ["XAG", "USD", "silver", "industrial metals"],
    "BTCUSD": ["BTC", "USD", "bitcoin", "crypto", "cryptocurrency", "blockchain"],
}

ASSET_CLASS_PAIRS = {
    "Forex": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"],
    "Gold": ["XAUUSD"],
    "Silver": ["XAGUSD"],
    "Bitcoin": ["BTCUSD"],
}

HIGH_IMPACT_KEYWORDS = [
    "interest rate", "rate hike", "rate cut", "fed decision", "fomc", "ecb decision",
    "bank of england", "boe", "bank of japan", "boj", "rba", "rbnz", "monetary policy",
    "quantitative easing", "nonfarm payroll", "nfp", "cpi", "inflation", "gdp",
    "unemployment", "jobs report", "trade balance", "retail sales", "ppi", "ism",
    "consumer confidence", "manufacturing pmi", "services pmi", "war", "conflict",
    "sanctions", "default", "recession", "crisis", "emergency", "collapse",
    "tariff", "trade war", "geopolitical", "safe haven", "dollar index", "dxy",
    "sec lawsuit", "etf approval", "etf rejection", "halving", "hack", "exchange collapse"
]

MEDIUM_IMPACT_KEYWORDS = [
    "housing", "building permits", "durable goods", "factory orders",
    "weekly claims", "jobless claims", "current account",
    "speech", "testimony", "statement", "minutes",
    "blockchain update", "network upgrade"
]

MODEL_ACCURACY = {
    "Grand Ensemble (All Indicators + ARIMA)": "~85-92%",
    "Conservative (Strict 6/8 Indicators)": "~80-87%",
    "Balanced (Standard 5/8 Indicators)": "~78-85%",
    "Aggressive (Relaxed 4/8 Indicators)": "~72-80%",
    "ARIMA + Indicator Hybrid": "~78-85%",
}
MODEL_OPTIONS = list(MODEL_ACCURACY.keys())
MODEL_DESC = {
    "Grand Ensemble (All Indicators + ARIMA)": "All 8 indicators + ARIMA vote — strictest gate, highest accuracy",
    "Conservative (Strict 6/8 Indicators)": "Requires 6 of 8 indicators aligned — fewer signals, higher quality",
    "Balanced (Standard 5/8 Indicators)": "Standard 5/8 alignment — balanced signal frequency",
    "Aggressive (Relaxed 4/8 Indicators)": "4/8 indicator threshold — more signals, slightly lower accuracy",
    "ARIMA + Indicator Hybrid": "Time series forecast + indicator confirmation combined",
}

MIN_CONFIDENCE_GATE = 70

TIMEFRAME_TTL = {
    "M1": 30, "M5": 60, "M15": 90, "M30": 120,
    "H1": 180, "H4": 300, "D1": 600, "W1": 1800,
}
TIMEFRAME_PA_LOOKBACK = {
    "M1": 200, "M5": 300, "M15": 500, "M30": 600,
    "H1": 800, "H4": 1000, "D1": 1000, "W1": 500,
}


# =============================================================
# LIVE DIGITAL WATCH & AUTO REFRESH
# =============================================================
def render_auto_refresh_bar():
    now = time.time()
    if "last_auto_refresh" not in st.session_state:
        st.session_state["last_auto_refresh"] = now
    start_time = st.session_state["last_auto_refresh"]

    html_code = f"""
<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
    body{{margin:0;padding:0;background-color:transparent;font-family:'Space Mono',monospace;color:#F8FAFC;overflow:hidden;}}
    .panel{{display:flex;align-items:center;gap:14px;background:linear-gradient(90deg,#0B1120,#0F1929);
        border:1px solid #1E3A5F;border-radius:12px;padding:12px 20px;box-sizing:border-box;width:100%;}}
    #live_clock{{font-size:1.4rem;font-weight:800;color:#F8FAFC;text-shadow:0 0 10px rgba(56,189,248,0.5);margin-right:10px;}}
    #refresh_dot{{width:8px;height:8px;border-radius:50%;background:#10B981;box-shadow:0 0 6px #10B981;}}
    #refresh_timer{{font-size:0.85rem;font-weight:700;color:#38BDF8;letter-spacing:0.05em;min-width:45px;}}
    .label{{font-size:0.75rem;color:#64748B;white-space:nowrap;}}
    .progress-bg{{flex-grow:1;height:5px;background:#1E293B;border-radius:4px;overflow:hidden;min-width:50px;}}
    #progress_fill{{height:100%;width:0%;border-radius:4px;background:linear-gradient(90deg,#6366F1,#38BDF8);transition:width 1s linear;}}
</style></head><body>
<div class="panel">
    <div id="live_clock">00:00:00</div>
    <div id="refresh_dot"></div>
    <div id="refresh_timer">5:00</div>
    <div class="label">Next Sync</div>
    <div class="progress-bg"><div id="progress_fill"></div></div>
</div>
<script>
    let startTime = {start_time};
    const totalSecs = {AUTO_REFRESH_SECONDS};
    function updateClock() {{
        const date = new Date();
        const h = String(date.getHours()).padStart(2,'0');
        const m = String(date.getMinutes()).padStart(2,'0');
        const s = String(date.getSeconds()).padStart(2,'0');
        document.getElementById('live_clock').innerText = h+":"+m+":"+s;
        let elapsed = (Date.now()/1000) - startTime;
        if (elapsed >= totalSecs) {{ startTime = Date.now()/1000; elapsed = 0; }}
        const remain = Math.max(0, totalSecs - elapsed);
        const rm = Math.floor(remain/60);
        const rs = Math.floor(remain%60);
        document.getElementById('refresh_timer').innerText = rm+":"+(rs<10?"0":"")+rs;
        let pct = (elapsed/totalSecs)*100;
        if (pct>100) pct=100;
        document.getElementById('progress_fill').style.width = pct+"%";
        let color = "#10B981";
        if (pct>=85) color="#EF4444";
        else if (pct>=60) color="#F59E0B";
        document.getElementById('refresh_timer').style.color = color;
        const dot = document.getElementById('refresh_dot');
        dot.style.background = color;
        dot.style.boxShadow = "0 0 8px "+color;
    }}
    updateClock();
    setInterval(updateClock, 1000);
</script></body></html>
"""
    components.html(html_code, height=75)

    elapsed = now - start_time
    if AUTO_REFRESH_AVAILABLE:
        st_autorefresh(interval=AUTO_REFRESH_MS, limit=None, key="main_autorefresh")
    else:
        if elapsed >= AUTO_REFRESH_SECONDS:
            st.session_state["last_auto_refresh"] = time.time()
            st.cache_data.clear()
            st.rerun()
        st.markdown(
            "<div style='font-size:0.75rem;color:#EF4444;margin-top:2px;font-weight:bold;'>"
            "Auto-refresh backend missing. Run: "
            "<code style='color:#F8FAFC;background:#7F1D1D;padding:2px 4px;border-radius:4px;'>"
            "pip install streamlit-autorefresh</code></div>",
            unsafe_allow_html=True)


# =============================================================
# SMART RISK RATIO ADVISOR
# =============================================================
def compute_risk_ratio_suggestion(conf, adx, pa, news_status, atr, ticker, live_df):
    if adx >= 40:
        adx_score, adx_lbl, adx_col = 100, f"Very Strong ({adx:.0f})", "#10B981"
    elif adx >= 30:
        adx_score, adx_lbl, adx_col = 80, f"Strong ({adx:.0f})", "#34D399"
    elif adx >= 20:
        adx_score, adx_lbl, adx_col = 55, f"Moderate ({adx:.0f})", "#38BDF8"
    elif adx >= 15:
        adx_score, adx_lbl, adx_col = 30, f"Weak ({adx:.0f})", "#F59E0B"
    else:
        adx_score, adx_lbl, adx_col = 10, f"Ranging ({adx:.0f})", "#EF4444"

    if conf >= 90:
        conf_score, conf_lbl, conf_col = 100, f"Elite ({conf:.0f}%)", "#10B981"
    elif conf >= 80:
        conf_score, conf_lbl, conf_col = 80, f"High ({conf:.0f}%)", "#34D399"
    elif conf >= 70:
        conf_score, conf_lbl, conf_col = 55, f"Good ({conf:.0f}%)", "#38BDF8"
    else:
        conf_score, conf_lbl, conf_col = 20, f"Low ({conf:.0f}%)", "#EF4444"

    pa_score = pa["pa_score"]
    if pa_score >= 80:
        pa_q_score, pa_q_lbl, pa_q_col = 100, f"Excellent ({pa_score}%)", "#10B981"
    elif pa_score >= 65:
        pa_q_score, pa_q_lbl, pa_q_col = 75, f"Good ({pa_score}%)", "#34D399"
    elif pa_score >= 50:
        pa_q_score, pa_q_lbl, pa_q_col = 50, f"Fair ({pa_score}%)", "#F59E0B"
    else:
        pa_q_score, pa_q_lbl, pa_q_col = 20, f"Poor ({pa_score}%)", "#EF4444"

    news_s = news_status["status"]
    if news_s == "SAFE":
        news_score, news_lbl, news_col = 100, "Clear", "#10B981"
    elif news_s == "CAUTION":
        news_score, news_lbl, news_col = 45, "Caution", "#F59E0B"
    else:
        news_score, news_lbl, news_col = 10, "High Risk", "#EF4444"

    if len(live_df) >= 50:
        recent_atr = atr
        avg_atr = live_df["Close"].diff().abs().rolling(50).mean().iloc[-1]
        vol_ratio = recent_atr / (avg_atr + 1e-9)
    else:
        vol_ratio = 1.0

    if 0.8 <= vol_ratio <= 1.8:
        vol_score, vol_lbl, vol_col = 90, f"Ideal ({vol_ratio:.2f}x)", "#10B981"
    elif vol_ratio < 0.8:
        vol_score, vol_lbl, vol_col = 40, f"Too low ({vol_ratio:.2f}x)", "#F59E0B"
    elif vol_ratio <= 2.5:
        vol_score, vol_lbl, vol_col = 60, f"Elevated ({vol_ratio:.2f}x)", "#F59E0B"
    else:
        vol_score, vol_lbl, vol_col = 15, f"Extreme ({vol_ratio:.2f}x)", "#EF4444"

    if pa["inside_bar"]:
        candle_score, candle_lbl, candle_col = 5, "Inside Bar", "#EF4444"
    elif pa["bull_pts"] >= 8 or pa["bear_pts"] >= 8:
        candle_score, candle_lbl, candle_col = 95, "Strong", "#10B981"
    elif pa["bull_pts"] >= 5 or pa["bear_pts"] >= 5:
        candle_score, candle_lbl, candle_col = 70, "Moderate", "#38BDF8"
    else:
        candle_score, candle_lbl, candle_col = 40, "Weak", "#F59E0B"

    composite = (
            adx_score * 0.20 + conf_score * 0.25 +
            pa_q_score * 0.20 + news_score * 0.15 +
            vol_score * 0.12 + candle_score * 0.08
    )
    if news_s == "DANGER":  composite = min(composite, 45)
    if pa["inside_bar"]:    composite = min(composite, 40)
    if adx < 15:            composite = min(composite, 38)

    if composite >= 82:
        ratio_str, ratio_float, grade = "1 : 3.0", 3.0, "S"
        tp_multipliers = [1.0, 2.0, 3.0]
        risk_pct_suggested = 1.5
        entry_timing = "Enter immediately — top-tier setup"
        summary_text = ("Elite Setup: All 6 factors strongly aligned. "
                        "Target full 1:3 R:R. Scale in aggressively at TP1 exit 30%, "
                        "trail remaining to TP2 & TP3.")
    elif composite >= 68:
        ratio_str, ratio_float, grade = "1 : 2.0", 2.0, "A"
        tp_multipliers = [1.0, 2.0, 2.5]
        risk_pct_suggested = 1.0
        entry_timing = "Enter on next candle confirmation"
        summary_text = ("Strong Setup: Most factors align well. "
                        "Target 1:2 R:R. Exit 50% at TP1, move SL to breakeven, "
                        "trail rest toward TP2.")
    elif composite >= 52:
        ratio_str, ratio_float, grade = "1 : 1.5", 1.5, "B"
        tp_multipliers = [1.0, 1.5, 2.0]
        risk_pct_suggested = 0.75
        entry_timing = "Wait for strong momentum candle, then enter"
        summary_text = ("Moderate Setup: Some conflicting factors present. "
                        "Play it safe — target 1:1.5. Exit full position at TP2.")
    elif composite >= 38:
        ratio_str, ratio_float, grade = "1 : 1.0", 1.0, "C"
        tp_multipliers = [1.0, 1.0, 1.2]
        risk_pct_suggested = 0.5
        entry_timing = "Consider waiting — marginal setup. If entering, use minimal size"
        summary_text = ("Marginal Setup: Significant risk factors detected. "
                        "Only target 1:1. Reduce lot size by 50% vs normal.")
    else:
        ratio_str, ratio_float, grade = "AVOID", 0.0, "D"
        tp_multipliers = [0.0, 0.0, 0.0]
        risk_pct_suggested = 0.0
        entry_timing = "Do NOT enter — conditions unfavourable"
        summary_text = ("Avoid This Trade: Multiple high-risk factors are stacked "
                        "against the setup. Stand by and wait for conditions to clear.")

    rules = []
    if ratio_float >= 2.0:
        rules += ["Enter full size at market or on next candle open",
                  "Close 40% at TP1 to lock profits immediately",
                  "Move SL to breakeven once price hits TP1",
                  "Trail remaining 60% toward TP2 and TP3"]
    elif ratio_float == 1.5:
        rules += ["Enter with 75% of normal lot size",
                  "Close 50% at TP1, rest at TP2",
                  "Move SL to breakeven at TP1",
                  "Do not trail — exit full position at TP2"]
    elif ratio_float == 1.0:
        rules += ["Enter with only 50% of normal lot size",
                  "Exit full position at TP1 only (1:1)",
                  "No trail, no scaling — flat exit at TP1",
                  "Reassess setup before re-entry"]
    else:
        rules += ["No entry recommended in current conditions",
                  "Wait for news to clear or ADX to strengthen",
                  "Watchlist only — set price alerts at key S/R levels"]

    if news_s == "DANGER":  rules.append("NEWS ACTIVE: Widen SL by +50% or skip this trade")
    if ticker in METALS:    rules.append("METAL TIP: Watch DXY direction — inverse correlation with Gold/Silver")
    if ticker in CRYPTO:    rules.append("CRYPTO TIP: Watch BTC dominance & Fear/Greed Index — high volatility asset")
    if pa["inside_bar"]:    rules.append("INSIDE BAR: Wait for breakout of mother candle before entry")

    tp_atr_values = [round(atr * m, 5) for m in tp_multipliers]

    return {
        "ratio_str": ratio_str, "ratio_float": ratio_float, "grade": grade,
        "composite_score": round(composite, 1), "entry_timing": entry_timing,
        "summary_text": summary_text, "risk_pct_suggested": risk_pct_suggested,
        "tp_multipliers": tp_multipliers, "tp_atr_values": tp_atr_values, "rules": rules,
        "factors": [
            {"name": "Trend Strength (ADX)", "score": adx_score, "label": adx_lbl, "color": adx_col},
            {"name": "AI Confidence", "score": conf_score, "label": conf_lbl, "color": conf_col},
            {"name": "Price Action Quality", "score": pa_q_score, "label": pa_q_lbl, "color": pa_q_col},
            {"name": "News Risk", "score": news_score, "label": news_lbl, "color": news_col},
            {"name": "Volatility (ATR ratio)", "score": vol_score, "label": vol_lbl, "color": vol_col},
            {"name": "Candle Pattern", "score": candle_score, "label": candle_lbl, "color": candle_col},
        ],
    }


def render_risk_ratio_advisor(rr, ticker, atr, current_price):
    grade = rr["grade"]
    ratio_str = rr["ratio_str"]
    score = rr["composite_score"]
    grade_class = f"rr-ratio-grade-{grade}"
    grade_colors = {"S": "#10B981", "A": "#34D399", "B": "#38BDF8", "C": "#F59E0B", "D": "#EF4444"}
    g_color = grade_colors.get(grade, "#94A3B8")
    price_fmt = ".2f" if ticker in METALS or ticker in CRYPTO else ".5f"

    st.markdown("<div class='rr-advisor-card'>", unsafe_allow_html=True)
    st.markdown(f"""
<div class='rr-advisor-title'>Smart Risk Ratio Advisor
<span style='font-size:0.75rem;color:#475569;font-family:"Space Mono",monospace;font-weight:400;'>
 — AI-computed optimal R:R for current conditions</span></div>
""", unsafe_allow_html=True)

    cols_main = st.columns([0.38, 0.32, 0.30])
    with cols_main[0]:
        st.markdown(f"""
<div class='rr-main-display'>
<div style='font-size:0.72rem;color:#475569;font-family:"Space Mono",monospace;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;'>
Recommended R:R
</div>
<div class='{grade_class}' style='font-family:"Space Mono",monospace;font-size:2.4rem;font-weight:800;line-height:1;'>{ratio_str}</div>
<div style='margin-top:10px;'>
<span style='background:{g_color}22;color:{g_color};padding:4px 14px;border-radius:20px;border:1px solid {g_color};font-size:0.75rem;font-weight:700;font-family:"Space Mono",monospace;'>
GRADE {grade} — Score {score}/100
</span>
</div>
</div>
<div style='font-size:0.78rem;color:#64748B;margin-bottom:6px;font-weight:600;'>
Entry Timing:</div>
<div style='font-size:0.83rem;color:#E2E8F0;font-weight:700;background:#0F1A2E;padding:10px 14px;border-radius:8px;border-left:3px solid {g_color};'>{rr["entry_timing"]}</div>
""", unsafe_allow_html=True)

    with cols_main[1]:
        st.markdown(
            "<div style='margin-bottom:4px;font-size:0.72rem;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;'>Factor Analysis</div>",
            unsafe_allow_html=True)
        for f in rr["factors"]:
            st.markdown(f"""
<div class='rr-factor-row'>
<span class='rr-factor-label'>{f['name']}</span>
<span class='rr-factor-bar-track'>
<span class='rr-factor-bar-fill' style='width:{f["score"]}%;background:{f["color"]};display:block;height:100%;border-radius:4px;'></span>
</span>
<span class='rr-factor-value' style='color:{f["color"]};min-width:90px;text-align:right;'>
{f['label']}
</span>
</div>""", unsafe_allow_html=True)

    with cols_main[2]:
        tp_labels = ["TP 1 (x1)", "TP 2 (x2)", "TP 3 (x3)"]
        tp_colors = ["#34D399", "#10B981", "#6EE7B7"]
        rr_labels = ["1:1", "1:2", "1:3"]
        st.markdown(
            "<div style='font-size:0.72rem;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;'>Target Estimates (ATR-based)</div>",
            unsafe_allow_html=True)
        for i, (lbl, val, col, rr_l, mult) in enumerate(zip(
                tp_labels, rr["tp_atr_values"], tp_colors, rr_labels, rr["tp_multipliers"])):
            is_recommended = (mult == rr["ratio_float"])
            border = f"2px solid {col}" if is_recommended else "1px solid #1E293B"
            star = " (Recommended)" if is_recommended else ""
            st.markdown(f"""
<div style='background:#0F1A2E;border-radius:8px;padding:8px 12px;margin-bottom:6px;border:{border};'>
<div style='font-size:0.68rem;color:#475569;font-weight:700;'>{lbl} {rr_l}{star}</div>
<div style='font-size:0.95rem;font-weight:800;color:{col};font-family:"Space Mono",monospace;margin-top:2px;'>+{val:{price_fmt}}</div>
<div style='font-size:0.65rem;color:#334155;'>~{mult}x ATR</div>
</div>""", unsafe_allow_html=True)

        r_pct = rr["risk_pct_suggested"]
        r_col = "#10B981" if r_pct >= 1.0 else "#F59E0B" if r_pct >= 0.5 else "#EF4444"
        st.markdown(f"""
<div style='background:rgba(56,189,248,0.05);border:1px solid rgba(56,189,248,0.15);border-radius:8px;padding:8px 12px;margin-top:4px;text-align:center;'>
<div style='font-size:0.68rem;color:#475569;font-weight:700;margin-bottom:2px;'>
Suggested Risk / Trade</div>
<div style='font-size:1.2rem;font-weight:800;color:{r_col};font-family:"Space Mono",monospace;'>{r_pct:.1f}% of balance</div>
</div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div style='background:rgba(30,58,95,0.3);border:1px solid rgba(99,102,241,0.2);border-radius:10px;padding:12px 16px;margin:12px 0 8px 0;font-size:0.85rem;color:#CBD5E1;line-height:1.6;'>{rr["summary_text"]}</div>
""", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:0.72rem;color:#475569;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;'>Execution Rules</div>",
        unsafe_allow_html=True)
    rules_html = "".join([f"<div class='rr-rule-item'><span>{r}</span></div>" for r in rr["rules"]])
    st.markdown(f"<div class='rr-rule-box'>{rules_html}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================
# ASSET NAVIGATION BAR
# =============================================================
def render_asset_nav():
    ac = st.session_state.asset_class
    nav_cols = st.columns([0.08, 0.11, 0.03, 0.11, 0.03, 0.11, 0.03, 0.11, 0.39])
    with nav_cols[0]:
        st.markdown("<div class='asset-nav-label'>MARKETS</div>", unsafe_allow_html=True)
    with nav_cols[1]:
        if st.button("FOREX", key="nav_forex", use_container_width=True):
            st.session_state.asset_class = "Forex"
            st.session_state.selected_pair = "EURUSD"
            st.rerun()
    with nav_cols[3]:
        if st.button("GOLD", key="nav_gold", use_container_width=True):
            st.session_state.asset_class = "Gold"
            st.session_state.selected_pair = "XAUUSD"
            st.rerun()
    with nav_cols[5]:
        if st.button("SILVER", key="nav_silver", use_container_width=True):
            st.session_state.asset_class = "Silver"
            st.session_state.selected_pair = "XAGUSD"
            st.rerun()
    with nav_cols[7]:
        if st.button("BITCOIN", key="nav_bitcoin", use_container_width=True):
            st.session_state.asset_class = "Bitcoin"
            st.session_state.selected_pair = "BTCUSD"
            st.rerun()

    active_colors = {"Forex": "#6366F1", "Gold": "#F59E0B", "Silver": "#94A3B8", "Bitcoin": "#F97316"}
    active_labels = {"Forex": "FOREX FX", "Gold": "XAU/USD — GOLD",
                     "Silver": "XAG/USD — SILVER", "Bitcoin": "BTC/USD — BITCOIN"}
    color = active_colors[ac]
    label = active_labels[ac]
    st.markdown(f"""
<div style='height:3px;background:linear-gradient(90deg,{color}44 0%,{color} 30%,{color}44 100%);border-radius:4px;margin:-6px 0 18px 0;'></div>
<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px;'>
<div style='width:8px;height:8px;border-radius:50%;background:{color};box-shadow:0 0 8px {color};'></div>
<span style='color:{color};font-family:"Space Mono",monospace;font-size:0.78rem;font-weight:700;letter-spacing:0.1em;'>
ACTIVE MARKET: {label}</span>
</div>
""", unsafe_allow_html=True)


# =============================================================
# METAL / BITCOIN HERO CARDS
# =============================================================
def render_metal_hero(metal, current_price=0.0):
    if metal == "XAUUSD":
        st.markdown(f"""
<div class='metal-hero-card metal-hero-gold'>
<div class='metal-hero-icon'>Gold</div>
<div>
<div class='metal-hero-title' style='color:#FCD34D;'>Gold — XAUUSD</div>
<div class='metal-hero-subtitle'>XAU / USD · Spot Gold · Troy Ounce</div>
<div style='font-size:0.8rem;color:#92400E;margin-top:4px;'>
Safe haven asset · Moves with USD strength, inflation & geopolitics</div>
</div>
<div style='margin-left:auto;text-align:right;'>
<div style='font-size:2rem;font-weight:800;color:#F59E0B;font-family:"Space Mono",monospace;'>
${current_price:,.2f}</div>
<div class='badge-gold'>PRECIOUS METAL</div>
</div>
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class='metal-hero-card metal-hero-silver'>
<div class='metal-hero-icon'>Silver</div>
<div>
<div class='metal-hero-title' style='color:#CBD5E1;'>Silver — XAGUSD</div>
<div class='metal-hero-subtitle'>XAG / USD · Spot Silver · Troy Ounce</div>
<div style='font-size:0.8rem;color:#475569;margin-top:4px;'>
Industrial & precious · High volatility · Tracks gold with amplification</div>
</div>
<div style='margin-left:auto;text-align:right;'>
<div style='font-size:2rem;font-weight:800;color:#94A3B8;font-family:"Space Mono",monospace;'>
${current_price:,.3f}</div>
<div class='badge-silver'>INDUSTRIAL METAL</div>
</div>
</div>""", unsafe_allow_html=True)


def render_bitcoin_hero(current_price=0.0):
    st.markdown(f"""
<div class='metal-hero-card metal-hero-bitcoin'>
<div class='metal-hero-icon'>BTC</div>
<div>
<div class='metal-hero-title' style='color:#FB923C;'>Bitcoin — BTCUSD</div>
<div class='metal-hero-subtitle'>BTC / USD · Spot Bitcoin · Digital Asset</div>
<div style='font-size:0.8rem;color:#92400E;margin-top:4px;'>
Digital gold · Moves with risk sentiment, Fed policy & crypto market cycles</div>
</div>
<div style='margin-left:auto;text-align:right;'>
<div style='font-size:2rem;font-weight:800;color:#F97316;font-family:"Space Mono",monospace;'>
${current_price:,.2f}</div>
<div class='badge-bitcoin'>CRYPTOCURRENCY</div>
</div>
</div>""", unsafe_allow_html=True)


# =============================================================
# MARKET STATUS
# =============================================================
def is_market_open(pair):
    try:
        tick = mt5.symbol_info_tick(pair)
        if tick is None: return False
        age = (datetime.datetime.now() -
               datetime.datetime.fromtimestamp(tick.time)).total_seconds()
        return age < 1800
    except Exception:
        return False


def get_market_status_html(pair):
    day = datetime.datetime.now().weekday()
    if pair in CRYPTO:
        if not is_market_open(pair):
            return (
                "<div class='market-closed-banner'>"
                "<span style='color:#F97316;font-size:1.1rem;font-weight:700;'>"
                "Bitcoin Feed Temporarily Unavailable</span></div>")
        return (
            "<div style='background:#1A0900;border:1px solid #F97316;border-radius:12px;padding:10px 24px;text-align:center;margin-bottom:16px;'>"
            "<span style='color:#FB923C;font-weight:700;font-size:1rem;'>"
            "Bitcoin OPEN — 24/7 Live Data Active</span></div>")
    if day == 5:
        return (
            "<div class='market-closed-banner'>"
            "<span style='color:#F59E0B;font-size:1.1rem;font-weight:700;'>"
            "Market Closed — Weekend (Saturday)</span><br>"
            "<span style='color:#94A3B8;font-size:0.85rem;'>Next open: Monday 00:00 UTC</span></div>")
    elif day == 6:
        return (
            "<div class='market-closed-banner'>"
            "<span style='color:#F59E0B;font-size:1.1rem;font-weight:700;'>"
            "Market Closed — Weekend (Sunday)</span><br>"
            "<span style='color:#94A3B8;font-size:0.85rem;'>Next open: Monday 00:00 UTC</span></div>")
    elif not is_market_open(pair):
        return (
            "<div class='market-closed-banner'>"
            "<span style='color:#F59E0B;font-size:1.1rem;font-weight:700;'>"
            "Market Temporarily Closed / Low Liquidity</span></div>")
    else:
        return (
            "<div style='background:#064E3B;border:1px solid #10B981;border-radius:12px;padding:10px 24px;text-align:center;margin-bottom:16px;'>"
            "<span style='color:#6EE7B7;font-weight:700;font-size:1rem;'>"
            "Market OPEN — Live Data Active</span></div>")


# =============================================================
# TECHNICAL INDICATORS
# =============================================================
def compute_atr(df, period=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def compute_macd(series, fast=12, slow=26, signal=9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    m = ef - es
    return m, m.ewm(span=signal, adjust=False).mean(), m - m.ewm(span=signal, adjust=False).mean()


def compute_bollinger(series, period=20, std=2.0):
    sma = series.rolling(period).mean()
    sig = series.rolling(period).std()
    return sma + std * sig, sma, sma - std * sig


def compute_stochastic(df, k=14, d=3):
    lo = df["Low"].rolling(k).min()
    hi = df["High"].rolling(k).max()
    kp = 100 * (df["Close"] - lo) / (hi - lo + 1e-9)
    return kp, kp.rolling(d).mean()


def compute_williams_r(df, period=14):
    hi = df["High"].rolling(period).max()
    lo = df["Low"].rolling(period).min()
    return -100 * (hi - df["Close"]) / (hi - lo + 1e-9)


def compute_adx(df, period=14):
    up = df["High"].diff()
    down = -df["Low"].diff()
    pdm = up.where((up > down) & (up > 0), 0.0)
    mdm = down.where((down > up) & (down > 0), 0.0)
    a14 = compute_atr(df, period)
    pdi = 100 * pdm.ewm(span=period, adjust=False).mean() / (a14 + 1e-9)
    mdi = 100 * mdm.ewm(span=period, adjust=False).mean() / (a14 + 1e-9)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    return dx.ewm(span=period, adjust=False).mean(), pdi, mdi


def compute_cci(df, period=20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad + 1e-9)


def compute_volume_ratio(df):
    col = "Volume" if "Volume" in df.columns else (
        "tick_volume" if "tick_volume" in df.columns else None)
    if col is None:
        return pd.Series(1.0, index=df.index)
    vol = df[col].astype(float)
    return vol / (vol.rolling(20).mean() + 1e-9)


# =============================================================
# ARIMA
# =============================================================
def _arima_best_order(data: np.ndarray) -> tuple:
    if not STATSMODELS_OK:
        return (1, 1, 1)
    try:
        adf_result = adfuller(data, autolag="AIC")
        d_order = 0 if adf_result[1] < 0.05 else 1
    except Exception:
        d_order = 1
    best_aic = np.inf
    best_order = (1, d_order, 1)
    for p in range(3):
        for q in range(3):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(data, order=(p, d_order, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d_order, q)
            except Exception:
                continue
    return best_order


def get_arima_forecast(df, steps=3, lookback=300):
    if not STATSMODELS_OK:
        return 0, False
    try:
        actual_lookback = max(300, min(lookback, len(df)))
        data = df["Close"].iloc[-actual_lookback:].values.copy()
        if len(data) < 50:
            return 0, False
        cache_key = f"_arima_order_{len(data)}"
        if cache_key not in st.session_state:
            best_order = _arima_best_order(data)
            st.session_state[cache_key] = best_order
        else:
            best_order = st.session_state[cache_key]
        fitted = ARIMA(data, order=best_order).fit()
        forecast = fitted.forecast(steps=steps)
        last_price = data[-1]
        atr_approx = np.mean(np.abs(np.diff(data[-14:])))
        threshold = atr_approx * 0.5
        threshold = max(threshold, last_price * 0.0001)
        weights = [0.50, 0.30, 0.20]
        bull_score = 0.0
        bear_score = 0.0
        for i, (fc_price, w) in enumerate(zip(forecast, weights)):
            if fc_price > last_price + threshold:
                bull_score += w
            elif fc_price < last_price - threshold:
                bear_score += w
        if bull_score > 0.4:
            direction = 1
        elif bear_score > 0.4:
            direction = -1
        else:
            direction = 0
        st.session_state["arima_order_used"] = str(best_order)
        return direction, True
    except Exception as e:
        st.session_state["arima_error"] = str(e)
        return 0, False


def get_pip_multiplier(ticker):
    if ticker in METALS or ticker in CRYPTO: return 1.0
    if ticker in JPY_PAIRS:                  return 100.0
    return 10000.0


def get_pip_label(ticker):
    return "pts" if ticker in METALS or ticker in CRYPTO else "pips"


# =============================================================
# ✅ FIX: DATA LOADERS — ticker + timeframe as unique cache key
# =============================================================
def _fetch_live_data(selected_ticker: str, timeframe_label: str) -> pd.DataFrame:
    """Core MT5 fetch — always called with both ticker AND timeframe as arguments."""
    if not mt5.symbol_select(selected_ticker, True):
        return pd.DataFrame()
    tf_val = TIMEFRAME_DICT.get(timeframe_label, mt5.TIMEFRAME_M15)
    lookback = TIMEFRAME_PA_LOOKBACK.get(timeframe_label, 500)
    rates = mt5.copy_rates_from_pos(selected_ticker, tf_val, 0, lookback)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                       "close": "Close", "tick_volume": "Volume"}, inplace=True)
    return df


# ✅ FIX: Each cached function now takes BOTH ticker AND timeframe_label
#    so Streamlit builds a unique cache entry per (ticker, timeframe) combination.
@st.cache_data(ttl=30)
def _load_cached_30s(ticker: str, tf_label: str) -> pd.DataFrame:
    return _fetch_live_data(ticker, tf_label)

@st.cache_data(ttl=60)
def _load_cached_60s(ticker: str, tf_label: str) -> pd.DataFrame:
    return _fetch_live_data(ticker, tf_label)

@st.cache_data(ttl=90)
def _load_cached_90s(ticker: str, tf_label: str) -> pd.DataFrame:
    return _fetch_live_data(ticker, tf_label)

@st.cache_data(ttl=120)
def _load_cached_120s(ticker: str, tf_label: str) -> pd.DataFrame:
    return _fetch_live_data(ticker, tf_label)

@st.cache_data(ttl=180)
def _load_cached_180s(ticker: str, tf_label: str) -> pd.DataFrame:
    return _fetch_live_data(ticker, tf_label)

@st.cache_data(ttl=300)
def _load_cached_300s(ticker: str, tf_label: str) -> pd.DataFrame:
    return _fetch_live_data(ticker, tf_label)

@st.cache_data(ttl=600)
def _load_cached_600s(ticker: str, tf_label: str) -> pd.DataFrame:
    return _fetch_live_data(ticker, tf_label)

@st.cache_data(ttl=1800)
def _load_cached_1800s(ticker: str, tf_label: str) -> pd.DataFrame:
    return _fetch_live_data(ticker, tf_label)


def load_live_chart_data(ticker: str, tf_label: str) -> pd.DataFrame:
    """Route to the correct TTL-cached loader based on timeframe."""
    ttl_val = TIMEFRAME_TTL.get(tf_label, 30)
    dispatch = {
        30:   _load_cached_30s,
        60:   _load_cached_60s,
        90:   _load_cached_90s,
        120:  _load_cached_120s,
        180:  _load_cached_180s,
        300:  _load_cached_300s,
        600:  _load_cached_600s,
        1800: _load_cached_1800s,
    }
    loader = dispatch.get(ttl_val, _load_cached_30s)
    return loader(ticker, tf_label)


@st.cache_data(ttl=600)
def load_data(selected_ticker: str, timeframe_label: str = "D1") -> pd.DataFrame:
    """Legacy loader kept for compatibility."""
    return _fetch_live_data(selected_ticker, timeframe_label)


# =============================================================
# PRICE ACTION
# =============================================================
def compute_price_action(df):
    if len(df) < 5:
        return {"pa_score": 50, "inside_bar": False, "bull_pts": 0, "bear_pts": 0, "patterns": []}
    last = df.iloc[-1]
    prev = df.iloc[-2]
    inside_bar = bool((last["High"] < prev["High"]) and (last["Low"] > prev["Low"]))
    bull_pts = 0
    bear_pts = 0
    patterns = []
    body = abs(last["Close"] - last["Open"])
    upper_wick = last["High"] - max(last["Close"], last["Open"])
    lower_wick = min(last["Close"], last["Open"]) - last["Low"]
    if lower_wick > body * 1.5 and upper_wick < body * 0.5:
        bull_pts += 4
        patterns.append("Hammer")
    elif upper_wick > body * 1.5 and lower_wick < body * 0.5:
        bear_pts += 4
        patterns.append("Shooting Star")
    if last["Close"] > last["Open"] and prev["Close"] < prev["Open"] and last["Close"] > prev["High"]:
        bull_pts += 5
        patterns.append("Bullish Engulfing")
    elif last["Close"] < last["Open"] and prev["Close"] > prev["Open"] and last["Close"] < prev["Low"]:
        bear_pts += 5
        patterns.append("Bearish Engulfing")
    if last["Close"] > df["Close"].rolling(20).mean().iloc[-1]:
        bull_pts += 3
    else:
        bear_pts += 3
    total = bull_pts + bear_pts + 1e-9
    pa_score = int((bull_pts / total) * 100)
    return {"pa_score": pa_score, "inside_bar": inside_bar,
            "bull_pts": bull_pts, "bear_pts": bear_pts, "patterns": patterns}


# =============================================================
# NEWS STATUS
# =============================================================
def get_news_status(pair):
    currencies = PAIR_CURRENCIES.get(pair, [])
    status = "SAFE"
    events = ["No high-impact economic calendar events detected for " + ", ".join(currencies[:2])]
    try:
        r = requests.get("https://finance.yahoo.com/news/rss", timeout=3)
        if r.status_code == 200:
            from xml.etree import ElementTree
            root = ElementTree.fromstring(r.content)
            headlines = [item.find("title").text for item in root.findall(".//item")[:10]]
            for h in headlines:
                h_lower = h.lower()
                for c in currencies:
                    if c.lower() in h_lower:
                        for kw in HIGH_IMPACT_KEYWORDS:
                            if kw in h_lower:
                                status = "DANGER"
                                events.insert(0, f"HIGH IMPACT: {h}")
                                break
                        for kw in MEDIUM_IMPACT_KEYWORDS:
                            if kw in h_lower:
                                if status != "DANGER": status = "CAUTION"
                                events.append(f"MED IMPACT: {h}")
                                break
    except Exception:
        pass
    return {"status": status, "events": events[:3]}


# =============================================================
# GEMINI AI
# =============================================================
def generate_gemini_analysis(pair, df, signal, confidence, rr_advisor, api_key):
    if not api_key:
        trend = "Bullish" if signal == "BUY" else "Bearish"
        return (f"**Technical Summary ({pair})**: The market is showing a **{trend}** outlook "
                f"with a confidence score of **{confidence}%**. The Smart Risk Advisor recommends "
                f"a **{rr_advisor['ratio_str']}** risk-to-reward ratio. Key execution rules: "
                f"*{', '.join(rr_advisor['rules'][:2])}*.")
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        last_close = df["Close"].iloc[-1]
        prev_close = df["Close"].iloc[-2]
        change = ((last_close - prev_close) / prev_close) * 100
        prompt = f"""You are an elite institutional Forex & CFD trader. Analyze and provide a brief professional trading summary (max 3 sentences).
Asset: {pair} | Price: {last_close} (Change: {change:.2f}%) | Signal: {signal} | Confidence: {confidence}%
R:R: {rr_advisor['ratio_str']} (Grade {rr_advisor['grade']}) | Entry: {rr_advisor['entry_timing']}
Focus on risk management and clear executable guidance. Be extremely concise."""
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        r = requests.post(url, headers={"Content-Type": "application/json"}, json=data, timeout=8)
        if r.status_code == 200:
            return r.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error generating AI Analysis: {str(e)}"
    return "AI Analysis temporarily unavailable."


# =============================================================
# TELEGRAM
# =============================================================
def send_telegram_message(token, chat_id, text):
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# =============================================================
# LIVE LOT SIZE CALCULATOR (MT5 INTEGRATION)
# =============================================================
def calculate_live_lot_size(ticker, sl_price, entry_price, risk_pct):
    try:
        acc_info = mt5.account_info()
        if acc_info is None: return 0.01, 0.0
        balance = acc_info.balance
        risk_amount = balance * (risk_pct / 100.0)

        sym_info = mt5.symbol_info(ticker)
        if sym_info is None: return 0.01, balance

        tick_size = sym_info.trade_tick_size
        tick_value = sym_info.trade_tick_value

        sl_dist = abs(entry_price - sl_price)
        if sl_dist == 0 or tick_size == 0: return 0.01, balance

        sl_ticks = sl_dist / tick_size
        risk_per_lot = sl_ticks * tick_value

        if risk_per_lot <= 0: return 0.01, balance

        raw_lot = risk_amount / risk_per_lot
        step = sym_info.volume_step
        min_lot = sym_info.volume_min
        max_lot = sym_info.volume_max

        final_lot = round(raw_lot / step) * step
        final_lot = max(min_lot, min(final_lot, max_lot))
        return round(final_lot, 2), balance
    except Exception:
        return 0.01, 0.0


# =============================================================
# TRADE PLAN CALCULATOR
# =============================================================
def compute_trade_plan(signal, current_price, atr, ticker):
    is_metal = ticker in METALS
    is_crypto = ticker in CRYPTO
    is_jpy = ticker in JPY_PAIRS

    if is_metal or is_crypto:
        fmt = ".2f"
    elif is_jpy:
        fmt = ".3f"
    else:
        fmt = ".5f"

    sl_dist = atr * 1.5

    if signal == "BUY":
        sl = current_price - sl_dist
        tp1 = current_price + atr * 1.0
        tp2 = current_price + atr * 2.0
        tp3 = current_price + atr * 3.0
    else:
        sl = current_price + sl_dist
        tp1 = current_price - atr * 1.0
        tp2 = current_price - atr * 2.0
        tp3 = current_price - atr * 3.0

    return {
        "entry": current_price,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "fmt": fmt,
    }


# =============================================================
# ✅ FIX: CHART RENDERING — show last 200 candles only
# =============================================================
def render_chart(df, up_bb, mid_bb, low_bb, selected_pair, timeframe_choice):
    """
    Render candlestick chart.
    KEY FIX: slice to last 200 rows so each pair/timeframe shows its own data.
    The original code passed the entire df which caused visual overlap when
    Streamlit returned a cached frame from a different pair.
    """
    chart_lookback = 200
    plot_df = df.iloc[-chart_lookback:].copy()
    plot_ub = up_bb.iloc[-chart_lookback:]
    plot_mb = mid_bb.iloc[-chart_lookback:]
    plot_lb = low_bb.iloc[-chart_lookback:]

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df['Open'],
        high=plot_df['High'],
        low=plot_df['Low'],
        close=plot_df['Close'],
        name='Price',
        increasing_line_color='#10B981',
        decreasing_line_color='#EF4444',
        increasing_fillcolor='#10B981',
        decreasing_fillcolor='#EF4444',
    ))

    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_ub,
        name='BB Upper',
        line=dict(color='rgba(99,102,241,0.35)', width=1),
        hovertemplate='BB Upper: %{y:.5f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_mb,
        name='BB Mid',
        line=dict(color='rgba(56,189,248,0.55)', width=1.2, dash='dot'),
        hovertemplate='BB Mid: %{y:.5f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=plot_df.index, y=plot_lb,
        name='BB Lower',
        line=dict(color='rgba(99,102,241,0.35)', width=1),
        fill='tonexty',
        fillcolor='rgba(99,102,241,0.04)',
        hovertemplate='BB Lower: %{y:.5f}<extra></extra>',
    ))

    fig.update_layout(
        title=dict(
            text=f"  {selected_pair} · {timeframe_choice}",
            font=dict(size=14, color='#94A3B8'),
            x=0.0,
        ),
        template='plotly_dark',
        plot_bgcolor='#0F1A2E',
        paper_bgcolor='#060D1F',
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        yaxis=dict(gridcolor='#1E293B', showgrid=True, side='right', tickfont=dict(size=10)),
        xaxis=dict(gridcolor='#1E293B', showgrid=True, tickfont=dict(size=10)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10)),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{selected_pair}_{timeframe_choice}")


# =============================================================
# MAIN APP
# =============================================================
def main():
    st.markdown("<h1 style='text-align:center;'>🚀 Forex AI Pro v9</h1>", unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Configuration")
    api_key = st.sidebar.text_input("Gemini API Key", value=DEFAULT_GEMINI_API_KEY, type="password")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Telegram Alerts")
    tg_token = st.sidebar.text_input("Telegram Bot Token", value="", type="password")
    tg_chat_id = st.sidebar.text_input("Telegram Chat ID", value="")

    if st.sidebar.button("Send Test Message"):
        if tg_token and tg_chat_id:
            with st.spinner("Sending test message..."):
                ok = send_telegram_message(tg_token, tg_chat_id,
                                           "🔔 *Forex AI Pro v9*: Test message successful! Telegram notifications configured.")
                if ok:
                    st.sidebar.success("Test message sent!")
                else:
                    st.sidebar.error("Failed. Check Bot Token and Chat ID.")
        else:
            st.sidebar.warning("Enter both Bot Token and Chat ID first.")

    model_choice = st.sidebar.selectbox("AI Model Selection", MODEL_OPTIONS)
    st.sidebar.info(MODEL_DESC[model_choice])
    timeframe_choice = st.sidebar.selectbox("Chart Timeframe", list(TIMEFRAME_DICT.keys()), index=4)

    render_asset_nav()

    asset_class = st.session_state.asset_class
    ticker_options = ASSET_CLASS_PAIRS.get(asset_class, ["EURUSD"])

    if len(ticker_options) > 1:
        current_index = 0
        if st.session_state.selected_pair in ticker_options:
            current_index = ticker_options.index(st.session_state.selected_pair)
        else:
            st.session_state.selected_pair = ticker_options[0]
        selected_pair = st.sidebar.selectbox("Select Currency Pair", ticker_options, index=current_index)
        st.session_state.selected_pair = selected_pair
    else:
        st.session_state.selected_pair = ticker_options[0]
        selected_pair = st.session_state.selected_pair

    render_auto_refresh_bar()
    st.markdown(get_market_status_html(selected_pair), unsafe_allow_html=True)

    # ✅ FIX: Pass selected_pair explicitly so cache key is unique per ticker
    with st.spinner(f"Fetching live chart data for {selected_pair} {timeframe_choice}..."):
        df = load_live_chart_data(selected_pair, timeframe_choice)

    if df.empty:
        st.warning(f"No chart data for {selected_pair} {timeframe_choice}. Ensure MT5 is connected.")
        return

    current_price = df["Close"].iloc[-1]

    # Hero cards
    if selected_pair in METALS:
        render_metal_hero(selected_pair, current_price)
    elif selected_pair in CRYPTO:
        render_bitcoin_hero(current_price)
    else:
        currencies = PAIR_CURRENCIES.get(selected_pair, ["EUR", "USD"])
        st.markdown(f"""
<div class='metal-hero-card' style='background:linear-gradient(135deg,#0E172A 0%,#1E293B 100%);border:2px solid #6366F1;'>
<div class='metal-hero-icon'>💱</div>
<div>
<div class='metal-hero-title' style='color:#E2E8F0;'>{selected_pair}</div>
<div class='metal-hero-subtitle'>{currencies[0]} / {currencies[1]} · Foreign Exchange</div>
</div>
<div style='margin-left:auto;text-align:right;'>
<div style='font-size:2rem;font-weight:800;color:#38BDF8;font-family:"Space Mono",monospace;'>
{current_price:.5f}</div>
<div class='badge-silver' style='border-color:#6366F1;color:#818CF8;'>FOREX PAIR</div>
</div>
</div>""", unsafe_allow_html=True)

    # ── Indicators ──
    atr = compute_atr(df).iloc[-1]
    adx_series, pdi_series, mdi_series = compute_adx(df)
    adx_val = adx_series.iloc[-1]
    pa = compute_price_action(df)
    news_status = get_news_status(selected_pair)

    rsi = compute_rsi(df["Close"]).iloc[-1]
    macd, macdsig, macdhist = compute_macd(df["Close"])
    macd_val = macd.iloc[-1]
    macdsig_val = macdsig.iloc[-1]
    up_bb, mid_bb, low_bb = compute_bollinger(df["Close"])

    votes_buy = 0
    votes_sell = 0
    total_indicators = 8

    # 1. RSI
    if rsi < 35:
        votes_buy += 1
    elif rsi > 65:
        votes_sell += 1

    # 2. MACD
    if macd_val > macdsig_val:
        votes_buy += 1
    else:
        votes_sell += 1

    # 3. Bollinger Bands
    if current_price < low_bb.iloc[-1]:
        votes_buy += 1
    elif current_price > up_bb.iloc[-1]:
        votes_sell += 1

    # 4. ADX direction
    if pdi_series.iloc[-1] > mdi_series.iloc[-1]:
        votes_buy += 1
    else:
        votes_sell += 1

    # 5. Stochastic
    kp, dp = compute_stochastic(df)
    if kp.iloc[-1] < 20:
        votes_buy += 1
    elif kp.iloc[-1] > 80:
        votes_sell += 1

    # 6. Williams %R
    williams = compute_williams_r(df).iloc[-1]
    if williams < -80:
        votes_buy += 1
    elif williams > -20:
        votes_sell += 1

    # 7. CCI
    cci = compute_cci(df).iloc[-1]
    if cci < -100:
        votes_buy += 1
    elif cci > 100:
        votes_sell += 1

    # 8. ARIMA
    arima_direction, arima_ok = get_arima_forecast(df)
    if arima_direction == 1:
        votes_buy += 1
    elif arima_direction == -1:
        votes_sell += 1

    # ── Signal Logic ──
    dominant_votes = max(votes_buy, votes_sell)
    confidence = int((dominant_votes / total_indicators) * 100)

    if votes_buy > votes_sell:
        signal = "BUY"
    elif votes_sell > votes_buy:
        signal = "SELL"
    else:
        signal = "BUY" if pdi_series.iloc[-1] > mdi_series.iloc[-1] else "SELL"
        confidence = 50

    strictness_thresholds = {
        "Grand Ensemble (All Indicators + ARIMA)": 80,
        "Conservative (Strict 6/8 Indicators)": 75,
        "Balanced (Standard 5/8 Indicators)": 62,
        "Aggressive (Relaxed 4/8 Indicators)": 50,
        "ARIMA + Indicator Hybrid": 62,
    }
    gate = strictness_thresholds.get(model_choice, 62)

    signal_quality = "STRONG" if confidence >= gate else "WEAK"
    if confidence < gate:
        confidence = min(confidence, 55)

    trade_plan = compute_trade_plan(signal, current_price, atr, selected_pair)
    rr = compute_risk_ratio_suggestion(confidence, adx_val, pa, news_status, atr, selected_pair, df)

    if tg_token and tg_chat_id and signal in ("BUY", "SELL"):
        last_bar_time = str(df.index[-1])
        signal_key = f"{selected_pair}_{timeframe_choice}_{signal}_{last_bar_time}"
        if st.session_state.get("last_notified_key") != signal_key:
            fmt = trade_plan["fmt"]
            msg = f"""
🚨 *NEW TRADE SIGNAL* 🚨

*Asset*: {selected_pair} | *TF*: {timeframe_choice}
*Signal*: *{signal}* ({signal_quality})
*Confidence*: {confidence}%

*Entry*: `{trade_plan["entry"]:{fmt}}`
*Stop Loss*: `{trade_plan["sl"]:{fmt}}`
*TP1 (1:1)*: `{trade_plan["tp1"]:{fmt}}`
*TP2 (1:2)*: `{trade_plan["tp2"]:{fmt}}`
*TP3 (1:3)*: `{trade_plan["tp3"]:{fmt}}`

*R:R*: {rr['ratio_str']} (Grade {rr['grade']})
*Timing*: {rr['entry_timing']}

_Forex AI Pro v9_
"""
            if send_telegram_message(tg_token, tg_chat_id, msg):
                st.session_state["last_notified_key"] = signal_key
                st.toast("📨 Telegram signal sent!")

    col_left, col_right = st.columns([0.4, 0.6])

    with col_left:
        recommended_lot, live_balance = calculate_live_lot_size(
            selected_pair, trade_plan["sl"], trade_plan["entry"], rr["risk_pct_suggested"])

        fmt = trade_plan["fmt"]

        if signal == "BUY":
            sig_color = "#10B981"
            sig_bg = "linear-gradient(135deg,#022c22,#064e3b)"
            sig_border = "#10B981"
            sl_desc = "Stop Loss  (below entry)"
            sig_arrow = "▲"
        else:
            sig_color = "#EF4444"
            sig_bg = "linear-gradient(135deg,#450a0a,#7f1d1d)"
            sig_border = "#EF4444"
            sl_desc = "Stop Loss  (above entry)"
            sig_arrow = "▼"

        quality_color = "#10B981" if signal_quality == "STRONG" else "#F59E0B"
        quality_text = "STRONG SIGNAL" if signal_quality == "STRONG" else "⚠ WEAK — Reduce Lot Size"

        st.markdown(f"""
<div style='background:{sig_bg};border:2px solid {sig_border};border-radius:16px;padding:20px 24px;margin-bottom:12px;'>
<div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;'>
<div>
<div style='font-size:0.72rem;text-transform:uppercase;letter-spacing:0.15em;color:#CBD5E1;margin-bottom:4px;'>Current Signal</div>
<div style='font-size:3.2rem;font-weight:800;color:{sig_color};line-height:1;'>
{sig_arrow} {signal}</div>
</div>
<div style='text-align:right;'>
<div style='font-size:1.6rem;font-weight:800;color:#38BDF8;'>{confidence}%</div>
<div style='font-size:0.7rem;color:#94A3B8;margin-bottom:6px;'>Confidence</div>
<div style='background:{quality_color}22;border:1px solid {quality_color};color:{quality_color};padding:3px 10px;border-radius:20px;font-size:0.68rem;font-weight:700;white-space:nowrap;'>{quality_text}</div>
</div>
</div>
<div style='background:rgba(0,0,0,0.28);border-radius:10px;padding:12px 16px;'>
<div style='display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.07);'>
<span style='color:#94A3B8;font-size:0.8rem;font-weight:600;'>📍 Entry Price</span>
<span style='color:#F8FAFC;font-size:1rem;font-weight:800;font-family:"Space Mono",monospace;'>{trade_plan["entry"]:{fmt}}</span>
</div>
<div style='display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.07);'>
<span style='color:#94A3B8;font-size:0.8rem;font-weight:600;'>🛑 {sl_desc}</span>
<span style='color:#EF4444;font-size:1rem;font-weight:800;font-family:"Space Mono",monospace;'>{trade_plan["sl"]:{fmt}}</span>
</div>
<div style='display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.07);'>
<span style='color:#94A3B8;font-size:0.8rem;font-weight:600;'>🎯 TP 1 &nbsp;(1 : 1 R:R)</span>
<span style='color:#34D399;font-size:1rem;font-weight:800;font-family:"Space Mono",monospace;'>{trade_plan["tp1"]:{fmt}}</span>
</div>
<div style='display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.07);'>
<span style='color:#94A3B8;font-size:0.8rem;font-weight:600;'>🎯 TP 2 &nbsp;(1 : 2 R:R)</span>
<span style='color:#10B981;font-size:1rem;font-weight:800;font-family:"Space Mono",monospace;'>{trade_plan["tp2"]:{fmt}}</span>
</div>
<div style='display:flex;justify-content:space-between;padding:7px 0;'>
<span style='color:#94A3B8;font-size:0.8rem;font-weight:600;'>🎯 TP 3 &nbsp;(1 : 3 R:R)</span>
<span style='color:#6EE7B7;font-size:1rem;font-weight:800;font-family:"Space Mono",monospace;'>{trade_plan["tp3"]:{fmt}}</span>
</div>
</div>
<div style='display:flex;justify-content:center;gap:20px;margin-top:10px;font-size:0.75rem;color:#64748B;'>
<span>ADX <b style="color:#38BDF8">{adx_val:.0f}</b></span>
<span>|</span>
<span>ATR <b style="color:#38BDF8">{atr:{fmt}}</b></span>
<span>|</span>
<span>Votes BUY <b style="color:#10B981">{votes_buy}</b> / SELL <b style="color:#EF4444">{votes_sell}</b></span>
</div>
</div>
""", unsafe_allow_html=True)

        # ── Final Action Plan Box ──
        if rr["grade"] in ["S", "A", "B"]:
            verdict_text = f"EXECUTING {signal} SETUP"
            verdict_color = "#10B981" if signal == "BUY" else "#EF4444"
            exit_time_text = f"T+60 Mins (or hit TP1 at {trade_plan['tp1']:{fmt}})"
            action_desc = (f"Market is prime. Place a **{signal}** order now with {recommended_lot} lots. "
                           f"Set your Stop Loss strictly at {trade_plan['sl']:{fmt}}. "
                           f"If TP1 is hit within the next hour, close 50% of the position and move SL to entry.")
        else:
            verdict_text = "WAIT / NO TRADE"
            verdict_color = "#F59E0B"
            exit_time_text = "N/A - Monitor Only"
            recommended_lot = 0.00
            action_desc = ("Market structure is currently weak or erratic. **Do not place a trade.** "
                           "Wait for the next 1-hour candle to close and confirm momentum before risking capital.")

        st.markdown(f"""
<div class="final-action-box">
<div class="final-action-title">🔥 Final Action Plan (Next 1 Hour)</div>
<div class="final-grid">
<div class="final-card">
<div class="final-lbl">Verdict</div>
<div class="final-val" style="color:{verdict_color};">{verdict_text}</div>
</div>
<div class="final-card">
<div class="final-lbl">Timing / Entry</div>
<div class="final-val" style="font-size:1rem;">{rr['entry_timing']}</div>
</div>
<div class="final-card" style="border-color:#38BDF8;">
<div class="final-lbl">Exact Lot Size</div>
<div class="final-val" style="color:#38BDF8;">{recommended_lot} Lots</div>
<div style="font-size:0.6rem;color:#64748B;margin-top:2px;">Based on ${live_balance:.2f} Balance</div>
</div>
<div class="final-card">
<div class="final-lbl">Max Holding Time</div>
<div class="final-val" style="font-size:0.9rem;">{exit_time_text}</div>
</div>
</div>
<div class="final-instruction">
<b>Execution Command:</b> {action_desc}
</div>
</div>
        """, unsafe_allow_html=True)

        pa_color = "#10B981" if pa["pa_score"] >= 60 else "#EF4444" if pa["pa_score"] <= 40 else "#F59E0B"
        st.markdown(f"""
<div class='price-action-box'>
<div class='pa-item'>
<span>Price Action Quality Score</span>
<span style='color:{pa_color};font-weight:700;'>{pa["pa_score"]}%</span>
</div>
<div class='pa-item'>
<span>Candlestick Structure</span>
<span class='{"pa-bull" if pa["bull_pts"] >= pa["bear_pts"] else "pa-bear"}'>
{"+" if pa["bull_pts"] >= pa["bear_pts"] else "-"}{max(pa["bull_pts"], pa["bear_pts"])} pts
</span>
</div>
<div class='pa-item'>
<span>Inside Bar Setup</span>
<span class='{"pa-bull" if pa["inside_bar"] else "pa-neutral"}'>
{"Active" if pa["inside_bar"] else "None"}</span>
</div>
<div class='pa-item'>
<span>Detected Patterns</span>
<span style='font-style:italic;'>
{", ".join(pa["patterns"]) if pa["patterns"] else "None"}</span>
</div>
</div>""", unsafe_allow_html=True)

        render_risk_ratio_advisor(rr, selected_pair, atr, current_price)

    with col_right:
        # ✅ FIX: Use dedicated render_chart function with unique plotly key
        render_chart(df, up_bb, mid_bb, low_bb, selected_pair, timeframe_choice)

        # AI Summary
        st.markdown("<div class='suggestion-box'>", unsafe_allow_html=True)
        st.markdown("<div class='sug-title'>🧠 Forex AI Pro Analyst</div>", unsafe_allow_html=True)
        with st.spinner("Generating AI market report..."):
            ai_analysis = generate_gemini_analysis(selected_pair, df, signal, confidence, rr, api_key)
        st.write(ai_analysis)
        st.markdown("</div>", unsafe_allow_html=True)

        # News
        news_color = ("#EF4444" if news_status["status"] == "DANGER"
                      else "#F59E0B" if news_status["status"] == "CAUTION" else "#10B981")

        events_html = "".join(
            [f'<div style="font-size:0.82rem;color:#94A3B8;padding:5px 0;border-bottom:1px solid #1E293B;">· {ev}</div>'
             for ev in news_status["events"]])

        st.markdown(f"""
<div style='background:#0F1A2E;border:1px solid #1E3A5F;border-radius:12px;padding:18px;margin-top:15px;'>
<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
<span style='font-weight:800;font-size:1rem;color:#E2E8F0;'>Market News Status</span>
<span style='color:{news_color};font-weight:700;font-size:0.8rem;background:{news_color}22;padding:3px 10px;border-radius:50px;border:1px solid {news_color};'>{news_status["status"]}</span>
</div>
{events_html}
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
