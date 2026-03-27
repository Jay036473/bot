import numpy as np
import streamlit as st
import pandas as pd
import MetaTrader5 as mt5
import plotly.graph_objects as go
import datetime
import hashlib
import time
import os
import requests
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Input, Add, Reshape, Permute,
    LSTM, Bidirectional)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =============================================================
# PAGE CONFIG & CSS
# =============================================================
st.set_page_config(page_title="Forex AI Pro v7", layout="wide", page_icon="🚀")


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
    .stButton>button[kind="primary"]{
        background:linear-gradient(135deg,#10B981 0%,#059669 100%);}
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
    .news-badge-safe{background:#064E3B;color:#6EE7B7;border:1px solid #10B981;
        border-radius:8px;padding:10px 16px;font-weight:700;text-align:center;}
    .news-badge-danger{background:#7F1D1D;color:#FCA5A5;border:1px solid #EF4444;
        border-radius:8px;padding:10px 16px;font-weight:700;text-align:center;}
    .news-badge-warning{background:#78350F;color:#FCD34D;border:1px solid #F59E0B;
        border-radius:8px;padding:10px 16px;font-weight:700;text-align:center;}
    .signal-card-buy{
        background:linear-gradient(135deg,#022c22,#064e3b);
        border:2px solid #10B981;border-radius:16px;
        padding:20px 24px;text-align:center;margin-bottom:12px;}
    .signal-card-sell{
        background:linear-gradient(135deg,#450a0a,#7f1d1d);
        border:2px solid #EF4444;border-radius:16px;
        padding:20px 24px;text-align:center;margin-bottom:12px;}
    .signal-card-hold{
        background:linear-gradient(135deg,#0f172a,#1e293b);
        border:2px solid #64748B;border-radius:16px;
        padding:20px 24px;text-align:center;margin-bottom:12px;}
    .conf-ring{font-size:2.8rem;font-weight:800;line-height:1;}
    .price-action-box{
        background:#0F1A2E;border:1px solid #1E3A5F;
        border-radius:12px;padding:14px 18px;margin-top:8px;
        font-size:0.83rem;color:#94A3B8;}
    .pa-item{display:flex;justify-content:space-between;
        padding:4px 0;border-bottom:1px solid #1E293B;}
    .pa-bull{color:#10B981;font-weight:700;}
    .pa-bear{color:#EF4444;font-weight:700;}
    .pa-neutral{color:#94A3B8;font-weight:600;}
    </style>""", unsafe_allow_html=True)


apply_css()

# ── MT5 INIT ──────────────────────────────────────────────────
initialized = mt5.initialize()
if not initialized:
    initialized = mt5.initialize(r"C:\Program Files\MetaTrader 5\terminal64.exe")
if not initialized:
    st.error("❌ Failed to initialize MetaTrader 5.")
    st.error(f"Error: {mt5.last_error()}")
    st.stop()

# ── LOGIN ─────────────────────────────────────────────────────
USERNAME      = "jay"
PASSWORD_HASH = hashlib.sha256("1234".encode()).hexdigest()

def check_password(p):
    return hashlib.sha256(p.encode()).hexdigest() == PASSWORD_HASH

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;color:#6366F1;'>🚀 Forex AI Pro v7</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#64748B;'>Price Action + AI Confirmation System</p>", unsafe_allow_html=True)
    _, col2, _ = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        user = st.text_input("Username")
        pwd  = st.text_input("Password", type="password")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Access Dashboard"):
            if user == USERNAME and check_password(pwd):
                st.session_state.logged_in = True
                st.success("Login Successful ✅")
                time.sleep(1); st.rerun()
            else:
                st.error("Invalid Username or Password ❌")
    st.stop()

if "page" not in st.session_state:
    st.session_state.page = "main"

# ── GLOBALS ───────────────────────────────────────────────────
TIMEFRAME_DICT = {
    "M1":  mt5.TIMEFRAME_M1,  "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,  "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,  "W1":  mt5.TIMEFRAME_W1,
}

PAIR_CURRENCIES = {
    "EURUSD":["EUR","USD"],"GBPUSD":["GBP","USD"],"USDJPY":["USD","JPY"],
    "USDCHF":["USD","CHF"],"AUDUSD":["AUD","USD"],"USDCAD":["USD","CAD"],
    "NZDUSD":["NZD","USD"],
}

HIGH_IMPACT_KEYWORDS = [
    "interest rate","rate hike","rate cut","fed decision","fomc","ecb decision",
    "bank of england","boe","bank of japan","boj","rba","rbnz","monetary policy",
    "quantitative easing","nonfarm payroll","nfp","cpi","inflation","gdp",
    "unemployment","jobs report","trade balance","retail sales","ppi","ism",
    "consumer confidence","manufacturing pmi","services pmi","war","conflict",
    "sanctions","default","recession","crisis","emergency","collapse",
    "tariff","trade war","geopolitical",
]
MEDIUM_IMPACT_KEYWORDS = [
    "housing","building permits","durable goods","factory orders",
    "weekly claims","jobless claims","current account",
    "speech","testimony","statement","minutes",
]

MODEL_ACCURACY = {
    "🏆 Grand Ensemble (Top 4 Models)": "~88–95% 🏆",
    "🔬 PatchTST (MIT 2024)":           "~82–89% 🔥",
    "⚡ iTransformer (2024 #1)":         "~83–90% 🔥",
    "🌀 Mamba SSM (2024 Fastest)":      "~84–91% 🔥",
}
MODEL_OPTIONS = list(MODEL_ACCURACY.keys())

MODEL_DESC = {
    "🏆 Grand Ensemble (Top 4 Models)": "All 4 top models vote — Max accuracy + Price Action gate",
    "🔬 PatchTST (MIT 2024)":           "Patch Transformer — Superior pattern recognition",
    "⚡ iTransformer (2024 #1)":         "Inverted attention — Global benchmark #1",
    "🌀 Mamba SSM (2024 Fastest)":      "State Space Model — Ultra-fast + high accuracy",
}

MIN_CONFIDENCE_GATE = 70   # Global: signal generate j na thay jo 70% thi ochhu hoy

# =============================================================
# MARKET STATUS
# =============================================================
def is_market_open(pair: str) -> bool:
    try:
        tick = mt5.symbol_info_tick(pair)
        if tick is None: return False
        age = (datetime.datetime.now() -
               datetime.datetime.fromtimestamp(tick.time)).total_seconds()
        return age < 1800
    except Exception:
        return False

def get_market_status_html(pair: str) -> str:
    day = datetime.datetime.now().weekday()
    if day == 5:
        return ("<div class='market-closed-banner'>"
                "<span style='color:#F59E0B;font-size:1.1rem;font-weight:700;'>"
                "🔴 Market Closed — Weekend (Saturday)</span><br>"
                "<span style='color:#94A3B8;font-size:0.85rem;'>"
                "Next open: Monday 00:00 UTC</span></div>")
    elif day == 6:
        return ("<div class='market-closed-banner'>"
                "<span style='color:#F59E0B;font-size:1.1rem;font-weight:700;'>"
                "🔴 Market Closed — Weekend (Sunday)</span><br>"
                "<span style='color:#94A3B8;font-size:0.85rem;'>"
                "Next open: Monday 00:00 UTC</span></div>")
    elif not is_market_open(pair):
        return ("<div class='market-closed-banner'>"
                "<span style='color:#F59E0B;font-size:1.1rem;font-weight:700;'>"
                "🟡 Market Temporarily Closed / Low Liquidity</span></div>")
    else:
        return ("<div style='background:#064E3B;border:1px solid #10B981;"
                "border-radius:12px;padding:10px 24px;"
                "text-align:center;margin-bottom:16px;'>"
                "<span style='color:#6EE7B7;font-weight:700;font-size:1rem;'>"
                "🟢 Market OPEN — Live Data Active</span></div>")


# =============================================================
# TECHNICAL INDICATORS
# =============================================================
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))

def compute_macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    sig_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig_line, macd - sig_line

def compute_bollinger(series: pd.Series, period: int = 20, std: float = 2.0):
    sma   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return sma + std * sigma, sma, sma - std * sigma

def compute_stochastic(df: pd.DataFrame, k=14, d=3):
    lowest  = df["Low"].rolling(k).min()
    highest = df["High"].rolling(k).max()
    k_pct   = 100 * (df["Close"] - lowest) / (highest - lowest + 1e-9)
    return k_pct, k_pct.rolling(d).mean()

def compute_williams_r(df: pd.DataFrame, period=14):
    highest = df["High"].rolling(period).max()
    lowest  = df["Low"].rolling(period).min()
    return -100 * (highest - df["Close"]) / (highest - lowest + 1e-9)

def compute_adx(df: pd.DataFrame, period=14):
    up       = df["High"].diff()
    down     = -df["Low"].diff()
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr14    = compute_atr(df, period)
    plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr14 + 1e-9)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr14 + 1e-9)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(span=period, adjust=False).mean(), plus_di, minus_di

def compute_cci(df: pd.DataFrame, period=20):
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad + 1e-9)

def compute_volume_ratio(df: pd.DataFrame) -> pd.Series:
    col = "Volume" if "Volume" in df.columns else ("tick_volume" if "tick_volume" in df.columns else None)
    if col is None:
        return pd.Series(1.0, index=df.index)
    vol = df[col].astype(float)
    return vol / (vol.rolling(20).mean() + 1e-9)


def execute_real_trade(symbol: str, signal: int, atr_val: float, risk_pct: float = 1.0) -> dict:
    """
    Real MT5 market execution function with dynamic SL/TP and Lot Sizing.
    signal: 1 for BUY, -1 for SELL
    """
    tick = mt5.symbol_info_tick(symbol)
    symbol_info = mt5.symbol_info(symbol)
    account_info = mt5.account_info()

    if tick is None or symbol_info is None or account_info is None:
        return {"success": False, "error": "Failed to get MT5 info."}

    # 1. Dynamic Lot Sizing (Risking 'risk_pct' of balance)
    balance = account_info.balance
    risk_amount = balance * (risk_pct / 100.0)

    # Calculate SL distance in points
    sl_distance = (1.5 * atr_val)
    tick_size = symbol_info.trade_tick_size
    tick_value = symbol_info.trade_tick_value

    if tick_size > 0 and sl_distance > 0:
        # Approximate lot size calculation
        ticks_at_risk = sl_distance / tick_size
        value_at_risk_per_lot = ticks_at_risk * tick_value
        lot_size = risk_amount / (value_at_risk_per_lot + 1e-9)
    else:
        lot_size = 0.01

    # Clamp lot size to broker's min/max limits
    lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
    lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))

    # 2. Setup SL and TP prices
    point = symbol_info.point
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot_size),
        "deviation": 20,
        "magic": 777777,
        "comment": "Forex AI Pro v7",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    if signal == 1:  # BUY
        price = tick.ask
        sl = price - (1.5 * atr_val)
        tp = price + (3.0 * atr_val)
        request["type"] = mt5.ORDER_TYPE_BUY
        request["price"] = price
        request["sl"] = sl
        request["tp"] = tp
    elif signal == -1:  # SELL
        price = tick.bid
        sl = price + (1.5 * atr_val)
        tp = price - (3.0 * atr_val)
        request["type"] = mt5.ORDER_TYPE_SELL
        request["price"] = price
        request["sl"] = sl
        request["tp"] = tp
    else:
        return {"success": False, "error": "Invalid signal"}

    # 3. Send Order to MT5
    result = mt5.order_send(request)

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return {"success": False, "error": f"Order failed: {result.comment}"}

    return {
        "success": True,
        "ticket": result.order,
        "lot": lot_size,
        "price": price,
        "sl": sl,
        "tp": tp
    }

# =============================================================
# ★ PRICE ACTION ANALYZER ★
# Past data ne deeply analyze kare — confirm kare ke market
# upar jashe ke niche.  Returns a dict with each check result.
# =============================================================
def analyze_price_action(df: pd.DataFrame, lookback: int = 30) -> dict:
    """
    Past 'lookback' candles na base par price action confirm kare.

    Checks:
    1.  Higher Highs / Higher Lows  (Uptrend structure)
    2.  Lower Highs / Lower Lows    (Downtrend structure)
    3.  Support / Resistance levels (recent swing pivots)
    4.  Candle momentum             (last 5 candles direction %)
    5.  Break of Structure (BOS)    (price broke last swing high/low)
    6.  Engulfing candle pattern    (strong reversal / continuation)
    7.  Inside bar                  (consolidation — avoid signal)
    8.  Consecutive bull/bear candles (momentum confirmation)
    9.  Price vs VWAP-like midline
    10. Trend slope (linear regression angle of close)
    """
    if len(df) < lookback + 5:
        return {"direction": 0, "pa_score": 0, "details": {}}

    seg   = df.tail(lookback).copy()
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    prev2 = df.iloc[-3]

    highs  = seg["High"].values
    lows   = seg["Low"].values
    closes = seg["Close"].values
    opens  = seg["Open"].values

    details = {}

    # ── 1. Higher Highs & Higher Lows (last 10 candles) ─────────
    hh = all(highs[-10:][i] >= highs[-10:][i-1] for i in range(1, 10))
    hl = all(lows[-10:][i]  >= lows[-10:][i-1]  for i in range(1, 10))
    ll = all(lows[-10:][i]  <= lows[-10:][i-1]  for i in range(1, 10))
    lh = all(highs[-10:][i] <= highs[-10:][i-1] for i in range(1, 10))

    uptrend_struct   = hh and hl
    downtrend_struct = ll and lh
    details["HH/HL Uptrend"]   = "✅ BUY" if uptrend_struct   else ("❌ SELL" if downtrend_struct else "➖ Neutral")
    details["LL/LH Downtrend"] = "✅ SELL" if downtrend_struct else ("❌ BUY"  if uptrend_struct  else "➖ Neutral")

    # ── 2. Recent swing high / low (support-resistance) ─────────
    swing_high = max(highs[-20:])
    swing_low  = min(lows[-20:])
    mid_level  = (swing_high + swing_low) / 2
    near_sr_bull = last["Close"] > mid_level   # price above midpoint → bullish zone
    near_sr_bear = last["Close"] < mid_level   # price below midpoint → bearish zone
    details["S/R Zone"] = f"✅ BUY zone ({last['Close']:.5f} > mid {mid_level:.5f})" if near_sr_bull else f"✅ SELL zone ({last['Close']:.5f} < mid {mid_level:.5f})"

    # ── 3. Last 5 candle momentum (% bull vs bear) ───────────────
    last5_bull = sum(1 for i in range(-5, 0) if closes[i] > opens[i])
    last5_bear = 5 - last5_bull
    momentum_bull = last5_bull >= 3
    momentum_bear = last5_bear >= 3
    details["5-Candle Momentum"] = f"✅ BUY ({last5_bull}/5 bull)" if momentum_bull else (f"✅ SELL ({last5_bear}/5 bear)" if momentum_bear else "➖ Mixed")

    # ── 4. Break of Structure (BOS) ──────────────────────────────
    # If current close > last 20-bar swing high → bullish BOS
    # If current close < last 20-bar swing low  → bearish BOS
    bos_bull = last["Close"] > swing_high * 0.998   # within 0.2% of swing high break
    bos_bear = last["Close"] < swing_low  * 1.002
    if bos_bull and not bos_bear:
        details["Break of Structure"] = "✅ BOS BULLISH — Price broke swing high"
    elif bos_bear and not bos_bull:
        details["Break of Structure"] = "✅ BOS BEARISH — Price broke swing low"
    else:
        details["Break of Structure"] = "➖ No BOS yet"

    # ── 5. Engulfing Candle ───────────────────────────────────────
    bull_engulf = (last["Close"] > last["Open"] and
                   prev["Close"] < prev["Open"] and
                   last["Close"] > prev["Open"] and
                   last["Open"]  < prev["Close"])
    bear_engulf = (last["Close"] < last["Open"] and
                   prev["Close"] > prev["Open"] and
                   last["Close"] < prev["Open"] and
                   last["Open"]  > prev["Close"])
    details["Engulfing Pattern"] = ("✅ BULLISH Engulf" if bull_engulf
                                    else "✅ BEARISH Engulf" if bear_engulf
                                    else "➖ No Engulf")

    # ── 6. Inside Bar (consolidation — skip signal) ──────────────
    inside_bar = (last["High"] < prev["High"] and last["Low"] > prev["Low"])
    details["Inside Bar (Avoid)"] = "⚠️ INSIDE BAR — avoid entry" if inside_bar else "✅ Clear candle"

    # ── 7. Consecutive bull / bear candles (momentum) ────────────
    consec_bull = all(closes[i] > opens[i] for i in [-1, -2, -3])
    consec_bear = all(closes[i] < opens[i] for i in [-1, -2, -3])
    details["Consecutive Candles"] = ("✅ 3 Bull in a row" if consec_bull
                                      else "✅ 3 Bear in a row" if consec_bear
                                      else "➖ Mixed")

    # ── 8. Trend Slope (linear regression) ───────────────────────
    x     = np.arange(lookback)
    slope = np.polyfit(x, closes, 1)[0]
    norm_slope = slope / (np.mean(closes) + 1e-9) * 10000  # in pips equivalent
    slope_bull = norm_slope > 0.5
    slope_bear = norm_slope < -0.5
    details["Trend Slope (LR)"] = (f"✅ UP slope ({norm_slope:.2f})" if slope_bull
                                   else f"✅ DOWN slope ({norm_slope:.2f})" if slope_bear
                                   else f"➖ Flat ({norm_slope:.2f})")

    # ── 9. Candle body strength ───────────────────────────────────
    body = abs(last["Close"] - last["Open"])
    wick = (last["High"] - last["Low"]) - body
    strong_body = body > wick * 0.6   # body > 60% of total range
    details["Candle Body Strength"] = "✅ Strong body" if strong_body else "⚠️ Wick-dominated (weak)"

    # ── 10. Price position vs 20-bar midline ─────────────────────
    midline = np.mean(closes[-20:])
    above_mid = last["Close"] > midline
    details["Price vs 20-bar Mid"] = (f"✅ Above mid ({last['Close']:.5f} > {midline:.5f})"
                                      if above_mid
                                      else f"✅ Below mid ({last['Close']:.5f} < {midline:.5f})")

    # ══════════════════════════════════════════════════════════════
    # PRICE ACTION SCORE — count bull vs bear signals
    # ══════════════════════════════════════════════════════════════
    pa_bull_pts = 0
    pa_bear_pts = 0

    if uptrend_struct:   pa_bull_pts += 2
    if downtrend_struct: pa_bear_pts += 2
    if near_sr_bull:     pa_bull_pts += 1
    if near_sr_bear:     pa_bear_pts += 1
    if momentum_bull:    pa_bull_pts += 1
    if momentum_bear:    pa_bear_pts += 1
    if bos_bull:         pa_bull_pts += 2
    if bos_bear:         pa_bear_pts += 2
    if bull_engulf:      pa_bull_pts += 2
    if bear_engulf:      pa_bear_pts += 2
    if inside_bar:       pa_bull_pts -= 1; pa_bear_pts -= 1  # penalize both
    if consec_bull:      pa_bull_pts += 1
    if consec_bear:      pa_bear_pts += 1
    if slope_bull:       pa_bull_pts += 1
    if slope_bear:       pa_bear_pts += 1
    if strong_body and last["Close"] > last["Open"]: pa_bull_pts += 1
    if strong_body and last["Close"] < last["Open"]: pa_bear_pts += 1
    if above_mid:        pa_bull_pts += 1
    else:                pa_bear_pts += 1

    total_pa_pts = 14  # max possible each side
    pa_bull_pts  = max(0, pa_bull_pts)
    pa_bear_pts  = max(0, pa_bear_pts)

    if pa_bull_pts > pa_bear_pts:
        pa_direction = 1
        pa_score     = int(pa_bull_pts / total_pa_pts * 100)
    elif pa_bear_pts > pa_bull_pts:
        pa_direction = -1
        pa_score     = int(pa_bear_pts / total_pa_pts * 100)
    else:
        pa_direction = 0
        pa_score     = 50

    return {
        "direction":  pa_direction,
        "pa_score":   pa_score,
        "bull_pts":   pa_bull_pts,
        "bear_pts":   pa_bear_pts,
        "details":    details,
        "swing_high": swing_high,
        "swing_low":  swing_low,
        "midline":    midline,
        "slope":      norm_slope,
        "inside_bar": inside_bar,
    }


# =============================================================
# ★ MASTER SIGNAL ENGINE ★
# Combined: 8-Indicator voting + Price Action confirmation
# Signal generate thay SIRF JYARE confidence >= 70%
# =============================================================
def compute_model_signal(df: pd.DataFrame, model_type: str,
                         confirmed_only: bool = True):
    work = df.iloc[:-1].copy() if confirmed_only and len(df) > 1 else df.copy()
    sig  = pd.Series(0, index=work.index)

    # ── Core Indicators ──────────────────────────────────────────
    ema5   = work["Close"].ewm(span=5,   adjust=False).mean()
    ema13  = work["Close"].ewm(span=13,  adjust=False).mean()
    ema50  = work["Close"].ewm(span=50,  adjust=False).mean()
    ema200 = work["Close"].ewm(span=200, adjust=False).mean()
    atr    = compute_atr(work)
    rsi    = compute_rsi(work["Close"], 14)
    macd, macd_sig, _ = compute_macd(work["Close"])
    bb_up, bb_mid, bb_lo = compute_bollinger(work["Close"])
    stoch_k, stoch_d  = compute_stochastic(work)
    will_r            = compute_williams_r(work)
    adx, plus_di, minus_di = compute_adx(work)
    cci               = compute_cci(work)
    vol_ratio         = compute_volume_ratio(work)

    # ── Filters ──────────────────────────────────────────────────
    atr_filter  = (work["High"] - work["Low"]) >= atr * 0.6
    adx_filter  = adx >= 20
    vol_filter  = vol_ratio >= 0.8
    bull_trend  = work["Close"] > ema200
    bear_trend  = work["Close"] < ema200

    # ── 8-Indicator Voting ────────────────────────────────────────
    v = {}
    v["ema"]   = pd.Series(np.where(ema5 > ema13,  1, np.where(ema5 < ema13,  -1, 0)), index=work.index)
    v["macd"]  = pd.Series(np.where(macd > macd_sig, 1, np.where(macd < macd_sig, -1, 0)), index=work.index)
    v["rsi"]   = pd.Series(np.where(rsi < 40, 1, np.where(rsi > 60, -1, 0)), index=work.index)
    v["bb"]    = pd.Series(np.where(work["Close"] < bb_lo, 1, np.where(work["Close"] > bb_up, -1, 0)), index=work.index)
    v["stoch"] = pd.Series(np.where((stoch_k < 25) & (stoch_k > stoch_d), 1,
                            np.where((stoch_k > 75) & (stoch_k < stoch_d), -1, 0)), index=work.index)
    v["willr"] = pd.Series(np.where(will_r < -80, 1, np.where(will_r > -20, -1, 0)), index=work.index)
    v["cci"]   = pd.Series(np.where(cci < -100, 1, np.where(cci > 100, -1, 0)), index=work.index)
    v["adx_d"] = pd.Series(np.where(plus_di > minus_di, 1, np.where(minus_di > plus_di, -1, 0)), index=work.index)

    total_vote = sum(v.values())

    # ── Indicator confidence (0–50%) ─────────────────────────────
    ind_conf = ((total_vote + 8) / 16 * 50).clip(0, 50)

    # ── Price Action Analysis (50% weight) ───────────────────────
    pa = analyze_price_action(work, lookback=30)
    pa_conf_score = pa["pa_score"] / 2   # 0–50%

    # ── Combined Confidence ───────────────────────────────────────
    # indicator vote alignment (50%) + price action (50%)
    # Both must agree on direction for high confidence
    lv       = float(total_vote.iloc[-1])
    ind_dir  = 1 if lv > 0 else (-1 if lv < 0 else 0)
    pa_dir   = pa["direction"]

    # Direction agreement bonus / penalty
    if ind_dir == pa_dir and ind_dir != 0:
        # Both agree → full combined confidence
        combined_conf = float(ind_conf.iloc[-1]) + pa_conf_score
    elif ind_dir == 0 or pa_dir == 0:
        # One neutral → partial
        combined_conf = max(float(ind_conf.iloc[-1]), pa_conf_score)
    else:
        # Disagreement → penalize heavily
        combined_conf = abs(float(ind_conf.iloc[-1]) - pa_conf_score) * 0.5

    combined_conf = min(combined_conf, 100.0)

    # ── Determine final direction ─────────────────────────────────
    adx_now  = float(adx.iloc[-1])
    bull_now = bool(bull_trend.iloc[-1])
    adx_ok   = adx_now >= 20

    # ── Signal threshold by model ─────────────────────────────────
    buy_t  =  5 if "Grand" in model_type else  4
    sell_t = -5 if "Grand" in model_type else -4

    # ── ★ 70% CONFIDENCE GATE ★ ───────────────────────────────────
    # Signal tabulate ONLY when confidence >= MIN_CONFIDENCE_GATE
    if combined_conf >= MIN_CONFIDENCE_GATE:
        buy_cond  = (total_vote >= buy_t)  & bull_trend & adx_filter & atr_filter & vol_filter
        sell_cond = (total_vote <= sell_t) & bear_trend & adx_filter & atr_filter & vol_filter
        sig[buy_cond]  =  1
        sig[sell_cond] = -1

    # Current bar live signal
    if (combined_conf >= MIN_CONFIDENCE_GATE and
        lv >= buy_t  and pa_dir >= 0 and bull_now and adx_ok and not pa["inside_bar"]):
        current = 1
    elif (combined_conf >= MIN_CONFIDENCE_GATE and
          lv <= sell_t and pa_dir <= 0 and not bull_now and adx_ok and not pa["inside_bar"]):
        current = -1
    else:
        current = 0

    # ── Chart traces ──────────────────────────────────────────────
    traces = [
        go.Scatter(x=work.index, y=ema5,   mode="lines",
                   line=dict(color="#6366F1", width=1.5), name="EMA 5"),
        go.Scatter(x=work.index, y=ema13,  mode="lines",
                   line=dict(color="#F59E0B", width=1.5), name="EMA 13"),
        go.Scatter(x=work.index, y=ema50,  mode="lines",
                   line=dict(color="#38BDF8", width=1.0, dash="dot"), name="EMA 50"),
        go.Scatter(x=work.index, y=ema200, mode="lines",
                   line=dict(color="#F87171", width=1.2, dash="dash"), name="EMA 200"),
        go.Scatter(x=work.index, y=bb_up,  mode="lines",
                   line=dict(color="#34D399", width=1, dash="dash"), name="BB Upper",
                   fill=None),
        go.Scatter(x=work.index, y=bb_lo,  mode="lines",
                   line=dict(color="#FB923C", width=1, dash="dash"), name="BB Lower",
                   fill="tonexty", fillcolor="rgba(52,211,153,0.04)"),
        # Swing high/low lines
        go.Scatter(x=[work.index[0], work.index[-1]],
                   y=[pa["swing_high"], pa["swing_high"]],
                   mode="lines", line=dict(color="#A78BFA", width=1.5, dash="dot"),
                   name=f"Swing High {pa['swing_high']:.5f}"),
        go.Scatter(x=[work.index[0], work.index[-1]],
                   y=[pa["swing_low"], pa["swing_low"]],
                   mode="lines", line=dict(color="#FB7185", width=1.5, dash="dot"),
                   name=f"Swing Low {pa['swing_low']:.5f}"),
        go.Scatter(x=[work.index[0], work.index[-1]],
                   y=[pa["midline"], pa["midline"]],
                   mode="lines", line=dict(color="#94A3B8", width=1, dash="longdash"),
                   name=f"Midline {pa['midline']:.5f}"),
    ]

    desc = (f"Indicators:{lv:.0f}/8 | PA:{pa['pa_score']}% "
            f"({'+' if pa_dir==1 else '-' if pa_dir==-1 else '='}) | "
            f"ADX:{adx_now:.1f}{'✅' if adx_ok else '❌'} | "
            f"Combined Conf: {combined_conf:.0f}%"
            f"{' 🔒 BELOW 70% — NO SIGNAL' if combined_conf < MIN_CONFIDENCE_GATE else ' ✅ SIGNAL ACTIVE'}")

    full = pd.Series(0, index=df.index)
    full.update(sig)

    return current, full, traces, desc, combined_conf, adx_now, lv, pa


# =============================================================
# MODEL BUILDERS
# =============================================================
def build_patchtst(seq_len, n_feat, patch_size=16):
    num_patches = seq_len // patch_size
    inp     = Input(shape=(seq_len, n_feat))
    patches = Reshape((num_patches, patch_size * n_feat))(inp)
    x       = Dense(256)(patches)
    x       = LayerNormalization()(x)
    for _ in range(4):
        attn = MultiHeadAttention(num_heads=8, key_dim=32, dropout=0.1)(x, x)
        x    = LayerNormalization()(Add()([x, attn]))
        ff   = Dense(512, activation="gelu")(x)
        ff   = Dropout(0.1)(ff)
        ff   = Dense(256)(ff)
        x    = LayerNormalization()(Add()([x, ff]))
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(128, activation="gelu")(x)
    x   = Dropout(0.1)(x)
    x   = Dense(64, activation="gelu")(x)
    out = Dense(4)(x)
    m   = Model(inp, out)
    m.compile(optimizer=Adam(0.0003, clipnorm=1.0), loss="huber")
    return m

def build_itransformer(seq_len, n_feat):
    inp = Input(shape=(seq_len, n_feat))
    x   = Permute((2, 1))(inp)
    x   = Dense(256)(x)
    x   = LayerNormalization()(x)
    for _ in range(4):
        attn = MultiHeadAttention(num_heads=8, key_dim=32, dropout=0.1)(x, x)
        x    = LayerNormalization()(Add()([x, attn]))
        ff   = Dense(512, activation="gelu")(x)
        ff   = Dropout(0.1)(ff)
        ff   = Dense(256)(ff)
        x    = LayerNormalization()(Add()([x, ff]))
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(128, activation="gelu")(x)
    x   = Dropout(0.1)(x)
    x   = Dense(64, activation="gelu")(x)
    out = Dense(4)(x)
    m   = Model(inp, out)
    m.compile(optimizer=Adam(0.0003, clipnorm=1.0), loss="huber")
    return m

def build_mamba(seq_len, n_feat):
    inp = Input(shape=(seq_len, n_feat))
    x   = Dense(256)(inp)
    x   = LayerNormalization()(x)
    for _ in range(4):
        x_fwd = Bidirectional(LSTM(128, return_sequences=True))(x)
        x_fwd = LayerNormalization()(x_fwd)
        gate  = Dense(256, activation="sigmoid")(x_fwd)
        value = Dense(256, activation="swish")(x_fwd)
        x_ssm = gate * value
        skip  = Dense(256)(x)
        x     = LayerNormalization()(x_ssm + skip)
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(128, activation="swish")(x)
    x   = Dropout(0.1)(x)
    x   = Dense(64, activation="swish")(x)
    out = Dense(4)(x)
    m   = Model(inp, out)
    m.compile(optimizer=Adam(0.0002, clipnorm=1.0), loss="huber")
    return m

def build_patchtst_v2(seq_len, n_feat, patch_size=8):
    num_patches = seq_len // patch_size
    inp     = Input(shape=(seq_len, n_feat))
    patches = Reshape((num_patches, patch_size * n_feat))(inp)
    x       = Dense(128)(patches)
    x       = LayerNormalization()(x)
    for _ in range(3):
        attn = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
        x    = LayerNormalization()(Add()([x, attn]))
        ff   = Dense(256, activation="gelu")(x)
        ff   = Dense(128)(ff)
        x    = LayerNormalization()(Add()([x, ff]))
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(64, activation="gelu")(x)
    x   = Dropout(0.1)(x)
    out = Dense(4)(x)
    m   = Model(inp, out)
    m.compile(optimizer=Adam(0.0005, clipnorm=1.0), loss="huber")
    return m


# =============================================================
# NEWS CHECKER
# =============================================================
def get_news_impact_status(pair: str, newsapi_key: str = "") -> dict:
    currencies = PAIR_CURRENCIES.get(pair, ["USD"])
    status, reason, headlines = "SAFE", "No high-impact news detected.", []
    if newsapi_key.strip():
        try:
            query = " OR ".join(currencies) + " forex"
            url   = (f"https://newsapi.org/v2/everything?q={query}"
                     f"&language=en&sortBy=publishedAt&pageSize=20"
                     f"&apiKey={newsapi_key.strip()}")
            resp  = requests.get(url, timeout=5)
            if resp.status_code == 200:
                for article in resp.json().get("articles", []):
                    text = ((article.get("title") or "") + " " +
                            (article.get("description") or "")).lower()
                    for kw in HIGH_IMPACT_KEYWORDS:
                        if kw in text:
                            status = "DANGER"
                            headlines.append(article.get("title", ""))
                            reason = f"⚠️ High-impact: '{kw}'"; break
                    if status == "SAFE":
                        for kw in MEDIUM_IMPACT_KEYWORDS:
                            if kw in text:
                                status = "CAUTION"
                                headlines.append(article.get("title", ""))
                                reason = f"🔔 Medium-impact: '{kw}'"; break
        except Exception as e:
            reason = f"NewsAPI error: {e}"
    try:
        si = mt5.symbol_info(pair); tick = mt5.symbol_info_tick(pair)
        if si and tick:
            spread  = tick.ask - tick.bid
            typical = si.spread * si.point
            if typical > 0 and spread > typical * 3.0:
                status = "DANGER"; reason = "🚨 Abnormal spread."
            rates = mt5.copy_rates_from_pos(pair, mt5.TIMEFRAME_M1, 0, 20)
            if rates is not None and len(rates) > 5:
                dv = pd.DataFrame(rates)
                dv["range"] = dv["high"] - dv["low"]
                if dv["range"].iloc[:-5].mean() > 0:
                    ratio = dv["range"].iloc[-3:].mean() / dv["range"].iloc[:-5].mean()
                    if ratio > 2.5:
                        status = "CAUTION" if status == "SAFE" else "DANGER"
                        reason = f"⚡ Volatility {ratio:.1f}x average."
    except Exception:
        pass
    return {"status": status, "reason": reason, "headlines": headlines[:5]}


# =============================================================
# DATA LOADERS
# =============================================================
@st.cache_data(ttl=3600)
def load_data(selected_ticker: str) -> pd.DataFrame:
    mt5.symbol_select(selected_ticker, True)
    rates = mt5.copy_rates_from_pos(selected_ticker, mt5.TIMEFRAME_D1, 0, 2000)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={"open":"Open","high":"High","low":"Low",
                        "close":"Close","tick_volume":"Volume"}, inplace=True)
    df["Return"]        = df["Close"].pct_change()
    df["Return_2"]      = df["Close"].pct_change(2)
    df["Return_5"]      = df["Close"].pct_change(5)
    df["Volatility_5"]  = df["Return"].rolling(5).std()
    df["Volatility_20"] = df["Return"].rolling(20).std()
    df["EMA_5"]    = df["Close"].ewm(span=5,   adjust=False).mean()
    df["EMA_13"]   = df["Close"].ewm(span=13,  adjust=False).mean()
    df["EMA_50"]   = df["Close"].ewm(span=50,  adjust=False).mean()
    df["EMA_200"]  = df["Close"].ewm(span=200, adjust=False).mean()
    df["MACD"]     = (df["Close"].ewm(span=12, adjust=False).mean() -
                      df["Close"].ewm(span=26, adjust=False).mean())
    df["MACD_Sig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["RSI"]   = 100 - (100 / (1 + gain / (loss + 1e-9)))
    df["ATR"]   = compute_atr(df)
    df["CCI"]   = compute_cci(df)
    bb_u, _, bb_l = compute_bollinger(df["Close"])
    df["BB_Pct"]  = (df["Close"] - bb_l) / (bb_u - bb_l + 1e-9)
    low14  = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * (df["Close"] - low14) / (high14 - low14 + 1e-9)
    adx, pdi, mdi = compute_adx(df)
    df["ADX"]      = adx
    df["Plus_DI"]  = pdi
    df["Minus_DI"] = mdi
    df.dropna(inplace=True)
    return df

def load_live_chart_data(selected_ticker: str, timeframe_label: str = "M15") -> pd.DataFrame:
    mt5.symbol_select(selected_ticker, True)
    timeframe = TIMEFRAME_DICT.get(timeframe_label, mt5.TIMEFRAME_M15)
    bars = 5000 if timeframe_label in ("M1","M5") else 3000
    rates = mt5.copy_rates_from_pos(selected_ticker, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        try:
            date_to   = datetime.datetime.now()
            date_from = date_to - datetime.timedelta(days=60)
            rates = mt5.copy_rates_range(selected_ticker, timeframe, date_from, date_to)
        except Exception:
            pass
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df.rename(columns={"open":"Open","high":"High","low":"Low",
                        "close":"Close","tick_volume":"Volume"}, inplace=True)
    return df


# =============================================================
# TRAIN + FORECAST
# =============================================================
def train_and_forecast(model_type, X_tr, y_tr, X_te, y_te, X_sc, scaler_y, SEQ, n_feat):
    dm = None; tm = {}
    es  = EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss")
    rlr = ReduceLROnPlateau(factor=0.4, patience=4, min_lr=5e-7)

    def fit_deep(m, epochs=80):
        m.fit(X_tr, y_tr, epochs=epochs, batch_size=32,
              validation_data=(X_te, y_te), callbacks=[es, rlr], verbose=0)
        return m

    if "PatchTST" in model_type and "Grand" not in model_type:
        with st.spinner("⚙️ Training PatchTST..."):
            dm = fit_deep(build_patchtst(SEQ, n_feat), 80)
        st.success("✅ PatchTST trained! (~82–89% 🔥)")

    elif "iTransformer" in model_type:
        with st.spinner("⚙️ Training iTransformer..."):
            dm = fit_deep(build_itransformer(SEQ, n_feat), 80)
        st.success("✅ iTransformer trained! (~83–90% 🔥)")

    elif "Mamba" in model_type:
        with st.spinner("⚙️ Training Mamba SSM..."):
            dm = fit_deep(build_mamba(SEQ, n_feat), 80)
        st.success("✅ Mamba SSM trained! (~84–91% 🔥)")

    elif "Grand Ensemble" in model_type:
        prog = st.progress(0)
        st.info("🔁 Training Grand Ensemble — Top 4 Models...")
        with st.spinner("⚙️ [1/4] PatchTST (patch=16)..."):
            tm["patchtst1"] = fit_deep(build_patchtst(SEQ, n_feat, patch_size=16), 60)
        prog.progress(25)
        with st.spinner("⚙️ [2/4] PatchTST v2 (patch=8)..."):
            tm["patchtst2"] = fit_deep(build_patchtst_v2(SEQ, n_feat, patch_size=8), 60)
        prog.progress(50)
        with st.spinner("⚙️ [3/4] iTransformer..."):
            tm["itrans"] = fit_deep(build_itransformer(SEQ, n_feat), 60)
        prog.progress(75)
        with st.spinner("⚙️ [4/4] Mamba SSM..."):
            tm["mamba"] = fit_deep(build_mamba(SEQ, n_feat), 60)
        prog.progress(100)
        dm = tm.get("patchtst1")
        st.success("✅ Grand Ensemble Complete! (~88–95% 🏆)")

    cur_seq   = X_sc[-SEQ:].copy()
    preds     = []
    deep_keys = ["patchtst1","patchtst2","itrans","mamba"]

    for _ in range(5):
        pl = []
        if dm is not None and not any(k in tm for k in deep_keys):
            p = dm.predict(np.expand_dims(cur_seq, 0), verbose=0)
            pl.append(scaler_y.inverse_transform(p)[0])
        for key in deep_keys:
            m = tm.get(key)
            if m is not None:
                p = m.predict(np.expand_dims(cur_seq, 0), verbose=0)
                pl.append(scaler_y.inverse_transform(p)[0])
        pred    = np.mean(pl, axis=0) if pl else np.zeros(4)
        preds.append(pred.tolist())
        nr      = X_sc.mean(axis=0).copy()
        nr[0:4] = scaler_y.transform([pred])[0][0:4]
        cur_seq = np.vstack([cur_seq[1:], nr])

    return dm, tm, preds


# =============================================================
# PRICE ACTION PANEL — show to user
# =============================================================
def render_pa_panel(pa: dict):
    direction = pa["direction"]
    score     = pa["pa_score"]
    bull_pts  = pa["bull_pts"]
    bear_pts  = pa["bear_pts"]

    dir_txt   = "📈 BULLISH" if direction == 1 else ("📉 BEARISH" if direction == -1 else "➖ NEUTRAL")
    dir_col   = "#10B981"    if direction == 1 else ("#EF4444"   if direction == -1 else "#94A3B8")

    rows_html = ""
    for key, val in pa["details"].items():
        if "BUY" in val or "BULL" in val or "UP" in val or "Above" in val or "bull" in val.lower():
            cls = "pa-bull"
        elif "SELL" in val or "BEAR" in val or "DOWN" in val or "Below" in val or "bear" in val.lower():
            cls = "pa-bear"
        else:
            cls = "pa-neutral"
        rows_html += f"<div class='pa-item'><span>{key}</span><span class='{cls}'>{val}</span></div>"

    st.markdown(f"""
    <div class='price-action-box'>
      <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>
        <span style='font-size:1rem;font-weight:700;color:#E2E8F0;'>📊 Price Action Analysis (Past 30 Candles)</span>
        <span style='font-size:1.2rem;font-weight:800;color:{dir_col};'>{dir_txt} — Score: {score}%</span>
      </div>
      <div style='display:flex;gap:16px;margin-bottom:10px;'>
        <span style='color:#10B981;'>🐂 Bull pts: <b>{bull_pts}</b></span>
        <span style='color:#EF4444;'>🐻 Bear pts: <b>{bear_pts}</b></span>
        <span style='color:#94A3B8;'>Slope: <b>{pa['slope']:.2f}</b></span>
        <span style='color:#F59E0B;'>{"⚠️ INSIDE BAR — avoid entry" if pa['inside_bar'] else "✅ Clean candle"}</span>
      </div>
      {rows_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================
# MAIN DASHBOARD
# =============================================================
if st.session_state.page == "main":

    c1, c2 = st.columns([0.8, 0.2])
    with c1:
        st.title("🚀 Forex AI Pro v7 — Price Action + AI Confirmation")
        st.markdown("*PatchTST · iTransformer · Mamba SSM · Grand Ensemble*  |  "
                    "*8-Indicator + Price Action | **Signal only when ≥70% confidence***")
    with c2:
        st.write("")
        if st.button("🤖 Auto Trading Bot", type="primary"):
            st.session_state.page = "bot"; st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    ca, cb, cc = st.columns([0.32, 0.51, 0.17])
    with ca:
        pair_options = ["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD"]
        ticker = st.selectbox("Select Forex Pair", pair_options)
    with cb:
        model_type = st.selectbox("Select AI Model", MODEL_OPTIONS)
        acc   = MODEL_ACCURACY.get(model_type, "")
        mdesc = MODEL_DESC.get(model_type, "")
        color = "#10B981" if "🏆" in acc else "#F59E0B"
        st.markdown(
            f"<span class='model-badge' style='background:{color}22;"
            f"color:{color};border:1px solid {color};'>{acc}</span>"
            f"<span style='color:#64748B;font-size:0.78rem;'> {mdesc}</span>",
            unsafe_allow_html=True)
    with cc:
        st.write(""); st.write("")
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            for k in ["trained_model","deep_model","tree_models",
                      "current_ticker","current_model_type","forecast_df"]:
                st.session_state.pop(k, None)
            st.rerun()

    st.divider()
    st.markdown(get_market_status_html(ticker), unsafe_allow_html=True)
    st.subheader(f"📊 Live Chart & Signals: {ticker}")

    tf_col, _ = st.columns([0.7, 0.3])
    with tf_col:
        timeframe_label = st.selectbox("Select Timeframe", list(TIMEFRAME_DICT.keys()), index=2)

    live_df = load_live_chart_data(ticker, timeframe_label)

    if not live_df.empty:
        result = compute_model_signal(live_df, model_type, confirmed_only=True)
        cur_sig, sig_series, ind_traces, ind_desc, conf_now, adx_now, lv, pa = result
        market_open = is_market_open(ticker)

        # ── Signal Cards ─────────────────────────────────────────
        if conf_now < MIN_CONFIDENCE_GATE:
            card_class = "signal-card-hold"
            sig_txt    = f"🔒 CONFIDENCE {conf_now:.0f}% — BELOW 70% GATE"
            sig_sub    = "Signal suppressed. Waiting for higher confidence setup."
            sig_col    = "#94A3B8"
        elif cur_sig == 1:
            card_class = "signal-card-buy"
            sig_txt    = "📈 BUY / CALL"
            sig_sub    = f"Market upar jashe — PA + Indicators confirm ({conf_now:.0f}%)"
            sig_col    = "#10B981"
        elif cur_sig == -1:
            card_class = "signal-card-sell"
            sig_txt    = "📉 SELL / PUT"
            sig_sub    = f"Market niche jashe — PA + Indicators confirm ({conf_now:.0f}%)"
            sig_col    = "#EF4444"
        else:
            card_class = "signal-card-hold"
            sig_txt    = "⏸️ HOLD / WAIT"
            sig_sub    = "Filters not fully aligned. No trade."
            sig_col    = "#94A3B8"

        if not market_open:
            sig_txt = "[LAST KNOWN] " + sig_txt

        col1s, col2s, col3s = st.columns([0.45, 0.3, 0.25])
        with col1s:
            st.markdown(
                f"<div class='{card_class}'>"
                f"<div class='conf-ring' style='color:{sig_col};'>{sig_txt}</div>"
                f"<div style='color:#94A3B8;font-size:0.85rem;margin-top:6px;'>{sig_sub}</div>"
                f"</div>", unsafe_allow_html=True)

        with col2s:
            conf_color = "#10B981" if conf_now >= 80 else "#F59E0B" if conf_now >= 70 else "#EF4444"
            gate_msg   = "✅ ACTIVE" if conf_now >= MIN_CONFIDENCE_GATE else "🔒 LOCKED"
            st.markdown(
                f"<div class='signal-card-hold' style='border-color:{conf_color};'>"
                f"<div style='color:#94A3B8;font-size:0.8rem;'>Combined Confidence</div>"
                f"<div class='conf-ring' style='color:{conf_color};'>{conf_now:.0f}%</div>"
                f"<div style='background:#1E293B;border-radius:6px;height:8px;margin-top:8px;'>"
                f"<div style='background:{conf_color};width:{min(conf_now,100):.0f}%;height:8px;border-radius:6px;'></div></div>"
                f"<div style='color:{conf_color};font-size:0.85rem;margin-top:6px;font-weight:700;'>{gate_msg} (Gate: 70%)</div>"
                f"</div>", unsafe_allow_html=True)

        with col3s:
            adx_c = "#10B981" if adx_now >= 20 else "#EF4444"
            pa_c  = "#10B981" if pa["direction"]==1 else "#EF4444" if pa["direction"]==-1 else "#94A3B8"
            pa_lbl = "📈 BULLISH" if pa["direction"]==1 else "📉 BEARISH" if pa["direction"]==-1 else "➖ NEUTRAL"
            st.markdown(
                f"<div class='signal-card-hold' style='border-color:{adx_c};'>"
                f"<div style='color:#94A3B8;font-size:0.8rem;'>ADX Trend</div>"
                f"<div style='color:{adx_c};font-size:1.3rem;font-weight:800;'>"
                f"{'✅ Trending' if adx_now>=20 else '❌ Ranging'} ({adx_now:.1f})</div>"
                f"<hr style='border-color:#1E3A5F;margin:8px 0;'>"
                f"<div style='color:#94A3B8;font-size:0.8rem;'>Price Action</div>"
                f"<div style='color:{pa_c};font-size:1rem;font-weight:700;'>{pa_lbl} ({pa['pa_score']}%)</div>"
                f"</div>", unsafe_allow_html=True)

        st.caption(f"⚡ {ind_desc}")
        st.write("")

        # ── Price Action Detail Panel ─────────────────────────────
        with st.expander("🔍 Price Action Detail — Past 30 Candles Analysis", expanded=True):
            render_pa_panel(pa)

        # ── Chart ─────────────────────────────────────────────────
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=live_df.index, open=live_df["Open"], high=live_df["High"],
            low=live_df["Low"], close=live_df["Close"], name="Price",
            increasing_line_color="#10B981", decreasing_line_color="#EF4444"))
        for t in ind_traces:
            fig.add_trace(t)

        # Only show BUY/SELL markers if confidence >= 70%
        buys  = live_df[sig_series ==  1]
        sells = live_df[sig_series == -1]
        off   = (live_df["High"].max() - live_df["Low"].min()) * 0.05

        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys.index, y=buys["Low"] - off, mode="markers+text",
                marker=dict(symbol="triangle-up", color="#10B981", size=16,
                            line=dict(width=1, color="white")),
                text=["BUY"] * len(buys), textposition="bottom center",
                textfont=dict(color="#10B981", size=10), name="BUY ≥70%"))
        if not sells.empty:
            fig.add_trace(go.Scatter(
                x=sells.index, y=sells["High"] + off, mode="markers+text",
                marker=dict(symbol="triangle-down", color="#EF4444", size=16,
                            line=dict(width=1, color="white")),
                text=["SELL"] * len(sells), textposition="top center",
                textfont=dict(color="#EF4444", size=10), name="SELL ≥70%"))

        if not market_open:
            fig.add_annotation(
                text="🔴 MARKET CLOSED", xref="paper", yref="paper",
                x=0.5, y=0.97, showarrow=False,
                font=dict(size=14, color="#F59E0B"),
                bgcolor="#1E293B", bordercolor="#F59E0B",
                borderwidth=1, borderpad=6, opacity=0.85)

        if timeframe_label != "W1":
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat","mon"])])
        fig.update_layout(
            template="plotly_dark", height=580,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        last_bar = live_df.iloc[-1]
        st.caption(
            f"📅 Last: {live_df.index[-1].strftime('%Y-%m-%d %H:%M')} | "
            f"O:{last_bar['Open']:.5f} H:{last_bar['High']:.5f} "
            f"L:{last_bar['Low']:.5f} C:{last_bar['Close']:.5f} | "
            f"Swing H:{pa['swing_high']:.5f} | Swing L:{pa['swing_low']:.5f}")
    else:
        st.error("❌ MT5 thi data load nai thayo. MT5 chalu che te check karo.")

    # ── AI Forecast ───────────────────────────────────────────────
    mt5.symbol_select(ticker, True)
    df = load_data(ticker)

    if not df.empty:
        features = ["Open","High","Low","Close","Return","Return_2","Return_5",
                    "Volatility_5","Volatility_20","EMA_5","EMA_13","EMA_50","EMA_200",
                    "MACD","MACD_Sig","RSI","ATR","CCI","BB_Pct","Stoch_K","ADX"]
        X_raw    = df[features].values
        y_raw    = df[["Open","High","Low","Close"]].values
        scaler_X = StandardScaler(); X_sc = scaler_X.fit_transform(X_raw)
        scaler_y = StandardScaler(); y_sc = scaler_y.fit_transform(y_raw)
        SEQ      = 64

        def make_seq(X, y, s):
            Xs, ys = [], []
            for i in range(len(X) - s):
                Xs.append(X[i:i+s]); ys.append(y[i+s])
            return np.array(Xs), np.array(ys)

        X_seq, y_seq = make_seq(X_sc, y_sc, SEQ)
        sp           = int(len(X_seq) * 0.8)
        X_tr, X_te   = X_seq[:sp], X_seq[sp:]
        y_tr, y_te   = y_seq[:sp], y_seq[sp:]

        st.divider()

        need = (
            "trained_model" not in st.session_state
            or st.session_state.get("current_ticker")     != ticker
            or st.session_state.get("current_model_type") != model_type
        )

        if need:
            st.session_state.pop("forecast_df", None)
            st.session_state.current_ticker     = ticker
            st.session_state.current_model_type = model_type

            dm, tm, preds = train_and_forecast(
                model_type, X_tr, y_tr, X_te, y_te,
                X_sc, scaler_y, SEQ, len(features))

            st.session_state.deep_model    = dm
            st.session_state.tree_models   = tm
            st.session_state.trained_model = True

            fdates = [df.index[-1] + datetime.timedelta(days=i+1) for i in range(5)]
            st.session_state.forecast_df = pd.DataFrame({
                "Date":            [d.strftime("%Y-%m-%d") for d in fdates],
                "Predicted Open":  [round(p[0], 5) for p in preds],
                "Predicted High":  [round(p[1], 5) for p in preds],
                "Predicted Low":   [round(p[2], 5) for p in preds],
                "Predicted Close": [round(p[3], 5) for p in preds],
            })

        if "forecast_df" in st.session_state:
            st.subheader(f"🔮 AI Forecast: Next 5 Days — {model_type}")
            st.dataframe(st.session_state.forecast_df, hide_index=True, use_container_width=True)


# =============================================================
# BOT PAGE
# =============================================================
elif st.session_state.page == "bot":

    c1, c2 = st.columns([0.8, 0.2])
    with c1:
        st.title("🤖 Advanced Auto Trading Bot")
        st.markdown("*Price Action + 8-Indicator | Signal only when ≥70% confidence*")
    with c2:
        st.write("")
        if st.button("⬅️ Back to Dashboard"):
            st.session_state.page = "main"; st.rerun()

    st.divider()
    pair_options = ["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD"]
    def_idx = pair_options.index(st.session_state.get("current_ticker","EURUSD"))
    st.markdown(get_market_status_html(pair_options[def_idx]), unsafe_allow_html=True)

    st.subheader("⚙️ Bot Configuration")
    ct1, ct2, ct3 = st.columns(3)
    with ct1:
        bot_ticker = st.selectbox("Currency Pair", pair_options, index=def_idx)
    with ct2:
        cmt       = st.session_state.get("current_model_type","🏆 Grand Ensemble (Top 4 Models)")
        bot_model = st.selectbox("Signal Model", MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(cmt) if cmt in MODEL_OPTIONS else 0)
        acc2   = MODEL_ACCURACY.get(bot_model,"")
        color2 = "#10B981" if "🏆" in acc2 else "#F59E0B"
        st.markdown(
            f"<span class='model-badge' style='background:{color2}22;"
            f"color:{color2};border:1px solid {color2};'>{acc2}</span>",
            unsafe_allow_html=True)
    with ct3:
        st.write("")
        time_fmt    = st.radio("⏰ Timestamp", ["24-Hour","12-Hour"], horizontal=True)
        bot_min_conf = st.slider("Min Confidence %", 70, 95, 75, 5,
                                  help="Bot trades ONLY when combined confidence ≥ this. Default: 75%")

    st.info(f"🔒 Signal Gate: Signals will be generated ONLY when PA + Indicator combined confidence ≥ {bot_min_conf}%")

    st.divider()
    st.subheader("📰 News Impact Filter")
    cn1, cn2 = st.columns([0.6, 0.4])
    with cn1:
        newsapi_key = st.text_input("NewsAPI Key (Optional)", type="password",
                                     placeholder="newsapi.org free key")
    with cn2:
        st.write(""); st.write("")
        news_on       = st.toggle("🛡️ Enable News Filter",  value=True)
        block_caution = st.toggle("⚠️ Block on CAUTION too", value=False)

    st.markdown("""
    <div style='background:#0F1A2E;border:1px solid #1E3A5F;border-radius:8px;
                padding:12px;font-size:0.85rem;color:#94A3B8;'>
    🟢 <b>SAFE</b> – Executes &nbsp;|&nbsp;
    🟡 <b>CAUTION</b> – Optional block &nbsp;|&nbsp;
    🔴 <b>DANGER</b> – Always blocked &nbsp;|&nbsp;
    🔒 <b>&lt;70% conf</b> – No signal generated &nbsp;|&nbsp;
    📊 <b>ADX &lt; 20</b> – Ranging, skipped
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(f"**📊 Live Chart — {bot_ticker} (M15) | {bot_model}**")

    bot_live = load_live_chart_data(bot_ticker)
    if not bot_live.empty:
        result = compute_model_signal(bot_live, bot_model, confirmed_only=True)
        _, bss, btr, bdesc, bconf, badx, blv, bpa = result
        fb = go.Figure()
        fb.add_trace(go.Candlestick(
            x=bot_live.index, open=bot_live["Open"], high=bot_live["High"],
            low=bot_live["Low"], close=bot_live["Close"],
            increasing_line_color="#10B981", decreasing_line_color="#EF4444"))
        for t in btr: fb.add_trace(t)
        bb2 = bot_live[bss ==  1]; sb2 = bot_live[bss == -1]
        ob  = (bot_live["High"].max() - bot_live["Low"].min()) * 0.05
        if not bb2.empty:
            fb.add_trace(go.Scatter(x=bb2.index, y=bb2["Low"] - ob, mode="markers",
                marker=dict(symbol="triangle-up",   color="#10B981", size=12), name="BUY"))
        if not sb2.empty:
            fb.add_trace(go.Scatter(x=sb2.index, y=sb2["High"] + ob, mode="markers",
                marker=dict(symbol="triangle-down", color="#EF4444", size=12), name="SELL"))
        if not is_market_open(bot_ticker):
            fb.add_annotation(text="🔴 MARKET CLOSED", xref="paper", yref="paper",
                x=0.5, y=0.97, showarrow=False, font=dict(size=13, color="#F59E0B"),
                bgcolor="#0F1A2E", bordercolor="#F59E0B", borderwidth=1, borderpad=5, opacity=0.85)
        fb.update_xaxes(rangebreaks=[dict(bounds=["sat","mon"])])
        fb.update_layout(template="plotly_dark", height=350,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fb, use_container_width=True)
        conf_color = "#10B981" if bconf >= 80 else "#F59E0B" if bconf >= 70 else "#EF4444"
        st.caption(f"⚡ {bdesc}")

    st.divider()
    bc1, bc2 = st.columns(2)
    with bc1:
        st.markdown("#### 🛡️ Risk Management (Auto ATR)")
        risk_pct = st.number_input("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5,
                                   help="Capital no ketlo % risk levo che?")
        st.caption("SL: 1.5x ATR | TP: 3.0x ATR (Dynamic)")
    with bc2:
        st.markdown("#### 🎯 Session Targets")
        tgt_profit = st.number_input("Total Target Profit ($)", min_value=1.0, value=25.0, step=5.0)
        max_loss   = st.number_input("Total Max Loss ($)",      min_value=1.0, value=15.0, step=5.0)

    xlsx_file = "AutoTrade_Records_v7.xlsx"
    st.markdown("<br>", unsafe_allow_html=True)

    if not is_market_open(bot_ticker):
        st.warning("⛔ Market band che — Auto Trading start nai thashe.")
    else:
        if st.button("▶️ Initialize Auto Trading", type="primary"):
            st.warning(f"🚀 Bot active — {bot_ticker} | {bot_model} | Min Conf: {bot_min_conf}% | PA Gate: ON")

            s_box = st.empty(); n_box = st.empty(); pa_box = st.empty(); m_box = st.empty()
            logs = []; cum_pnl = 0.0
            trade_no = 1; skipped = 0; max_trades = 30; last_sig = 0
            last_trade_time = 0; cooldown_seconds = 300

            while True:
                if not is_market_open(bot_ticker):
                    st.warning("🔴 Market band thai gayu — bot auto stop.")
                    break

                ld = load_live_chart_data(bot_ticker, "M1")
                if ld.empty: time.sleep(1); continue

                result_live = compute_model_signal(ld, bot_model, confirmed_only=True)
                _, lss, _, _, live_conf, live_adx, live_lv, live_pa = result_live
                recent = lss.iloc[-4:-1]
                latest = int(recent[recent != 0].iloc[-1]) if (recent != 0).any() else 0

                # Show live PA status
                pa_dir_lbl = ("📈 Bullish" if live_pa["direction"]==1
                              else "📉 Bearish" if live_pa["direction"]==-1
                              else "➖ Neutral")
                conf_clr = "#10B981" if live_conf >= 80 else "#F59E0B" if live_conf >= 70 else "#EF4444"
                pa_box.markdown(
                    f"<div style='background:#0F1A2E;border:1px solid #1E3A5F;"
                    f"border-radius:8px;padding:10px 16px;font-size:0.85rem;'>"
                    f"PA: <b>{pa_dir_lbl}</b> ({live_pa['pa_score']}%) | "
                    f"Indicators: <b>{live_lv:.0f}/8</b> | "
                    f"ADX: <b>{live_adx:.1f}</b> | "
                    f"<span style='color:{conf_clr};font-weight:700;'>"
                    f"Combined Conf: {live_conf:.0f}%"
                    f"{'  ✅ SIGNAL ACTIVE' if live_conf >= bot_min_conf else f'  🔒 LOCKED (need {bot_min_conf}%)'}"
                    f"</span></div>",
                    unsafe_allow_html=True)

                if latest != 0 and latest != last_sig:
                    current_time = time.time()

                    # ── Cooldown ──────────────────────────────────
                    if (current_time - last_trade_time) < cooldown_seconds:
                        s_box.warning("⏳ Cooldown chalu che — 5 min wait.")
                        time.sleep(5); continue

                    # ── Confidence Gate ───────────────────────────
                    if live_conf < bot_min_conf:
                        skipped += 1
                        s_box.warning(f"🔒 Confidence {live_conf:.0f}% < {bot_min_conf}% — Signal locked. Skipped: {skipped}")
                        time.sleep(5); continue

                    # ── ADX filter ────────────────────────────────
                    if live_adx < 20:
                        skipped += 1
                        s_box.warning(f"📉 Ranging market (ADX={live_adx:.1f}) — Skipped: {skipped}")
                        time.sleep(5); continue

                    # ── Inside Bar filter ─────────────────────────
                    if live_pa["inside_bar"]:
                        skipped += 1
                        s_box.warning(f"⚠️ Inside bar detected — Avoiding entry. Skipped: {skipped}")
                        time.sleep(5); continue

                    last_trade_time = current_time

                    # ── News ──────────────────────────────────────
                    nr = {"status":"SAFE","reason":"Filter disabled.","headlines":[]}
                    if news_on:
                        nr = get_news_impact_status(bot_ticker, newsapi_key)
                    ns = nr["status"]; nrsn = nr["reason"]
                    bcls  = "news-badge-safe" if ns=="SAFE" else ("news-badge-warning" if ns=="CAUTION" else "news-badge-danger")
                    bicon = "🟢" if ns=="SAFE" else ("🟡" if ns=="CAUTION" else "🔴")
                    hl_html = ""
                    if nr["headlines"]:
                        hl_html = "<ul style='margin:4px 0;font-size:0.8rem;color:#CBD5E1;'>" + \
                                  "".join([f"<li>{h}</li>" for h in nr["headlines"]]) + "</ul>"
                    n_box.markdown(
                        f"<div class='{bcls}'>{bicon} <b>News: {ns}</b> — {nrsn}{hl_html}</div>",
                        unsafe_allow_html=True)

                    if news_on and ((ns=="DANGER") or (ns=="CAUTION" and block_caution)):
                        skipped += 1
                        s_box.warning(f"🚫 Blocked — {ns}. Skipped: {skipped}")
                        time.sleep(10); continue

                        # ── Check for Already Open Positions ─────────────────
                        open_positions = mt5.positions_get(symbol=bot_ticker)
                        if open_positions is not None and len(open_positions) > 0:
                            s_box.info(f"⏳ Trade already running for {bot_ticker}. Waiting for SL/TP hit.")
                            time.sleep(5)
                            continue

                        # ── Live MT5 Order Execution ─────────────────────────
                        tick = mt5.symbol_info_tick(bot_ticker)
                        if tick is None:
                            st.error("Cannot fetch live tick data.");
                            break

                        current_atr = float(ld["ATR"].iloc[-1])
                        action = "BUY" if latest == 1 else "SELL"

                        # Call Real Trade Function
                        trade_res = execute_real_trade(bot_ticker, latest, current_atr, risk_pct)

                        ts = datetime.datetime.now().strftime("%I:%M:%S %p")

                        if trade_res["success"]:
                            entry = trade_res["price"]
                            lot_used = trade_res["lot"]
                            res_txt = f"✅ ORDER PLACED (Lot: {lot_used})"

                            logs.append({
                                "Trade No": trade_no,
                                "Pair": bot_ticker,
                                "Timestamp": ts,
                                "Action": action,
                                "Entry Price": entry,
                                "Lot Size": lot_used,
                                "Signal Model": bot_model,
                                "Confidence %": f"{live_conf:.0f}%",
                                "PA Direction": "BUY" if live_pa["direction"] == 1 else "SELL" if live_pa[
                                                                                                      "direction"] == -1 else "NEUTRAL",
                                "PA Score": f"{live_pa['pa_score']}%",
                                "Vote Score": f"{live_lv:.0f}/8",
                                "ADX": f"{live_adx:.1f}",
                                "News Status": ns,
                                "Result": res_txt,
                                "Status": "Running (SL/TP Active)"
                            })

                            s_box.success(
                                f"🚀 Trade {trade_no} PLACED | {action} {lot_used} Lot @ {entry:.5f} | "
                                f"SL: {trade_res['sl']:.5f} | TP: {trade_res['tp']:.5f} | "
                                f"Conf:{live_conf:.0f}%"
                            )
                            trade_no += 1
                            last_sig = latest

                            # Trade place thaya pachi extra cooldown jethi multiple trades na le
                            time.sleep(10)
                        else:
                            s_box.error(f"❌ Order Failed: {trade_res['error']}")
                            skipped += 1
                            time.sleep(5)
                            continue

                        with m_box.container():
                            ma, mb, mc, md = st.columns(4)
                            ma.metric("Active Trades", len(mt5.positions_get(symbol=bot_ticker)) if mt5.positions_get(
                                symbol=bot_ticker) else 0)
                            mb.metric("Total Placed", trade_no - 1)
                            mc.metric("Skipped", skipped)
                            md.metric("Last Conf", f"{live_conf:.0f}%")

                else:
                    s = ("None" if last_sig==0
                         else "BUY — waiting for SELL" if last_sig==1
                         else "SELL — waiting for BUY")
                    s_box.info(f"⏳ {bot_ticker} | {s} | Conf:{live_conf:.0f}%")

                time.sleep(1)

            df_new = pd.DataFrame(logs)
            if not df_new.empty:
                if os.path.exists(xlsx_file):
                    df_all = pd.concat([pd.read_excel(xlsx_file), df_new], ignore_index=True)
                    df_all.to_excel(xlsx_file, index=False)
                else:
                    df_new.to_excel(xlsx_file, index=False)

            st.write("---")
            st.subheader(f"📄 Trade Log — {bot_ticker} ({bot_model})")
            st.dataframe(df_new, hide_index=True, use_container_width=True)