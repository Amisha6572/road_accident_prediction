# ============================================================
# India Road Accident Risk Intelligence Platform  v2.0
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime

st.set_page_config(
    page_title="India Road Accident Risk Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Road Safety Intelligence Platform v2.0"}
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS  — high-contrast, consistent typography
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Roboto+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: #1a1a2e !important;
}
.main { background: #f0f2f6; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #1a1a4e 50%, #24243e 100%) !important;
}
section[data-testid="stSidebar"] * {
    color: #e8eaf6 !important;
    font-family: 'Inter', sans-serif !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #c5cae9 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-weight: 700 !important;
}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 6px 8px;
    gap: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: #5c6bc0 !important;
    padding: 8px 18px !important;
    transition: all 0.2s;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1a237e, #283593) !important;
}

.stTabs [aria-selected="true"] p {
    color: #ffffff !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1a237e 0%, #283593 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 0.7rem 2rem !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 15px rgba(26,35,126,0.3) !important;
}

.stButton > button p {
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(26,35,126,0.45) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    border-top: 3px solid #3949ab;
}
[data-testid="stMetricLabel"] {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #5c6bc0 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
[data-testid="stMetricValue"] {
    font-size: 1.7rem !important;
    font-weight: 800 !important;
    color: #1a237e !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: #ffffff !important;
    border-radius: 10px !important;
    border: 1.5px solid #e8eaf6 !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    color: #1a237e !important;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    border-radius: 8px !important;
    border: 1.5px solid #c5cae9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    color: #1a1a2e !important;
    background: #ffffff !important;
}
.stSlider > div > div > div > div {
    background: #3949ab !important;
}

/* ── Alerts ── */
.stSuccess { background: #e8f5e9 !important; border-left: 4px solid #2e7d32 !important; border-radius: 8px !important; }
.stWarning { background: #fff8e1 !important; border-left: 4px solid #f57f17 !important; border-radius: 8px !important; }
.stError   { background: #ffebee !important; border-left: 4px solid #c62828 !important; border-radius: 8px !important; }
.stInfo    { background: #e3f2fd !important; border-left: 4px solid #1565c0 !important; border-radius: 8px !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px !important; box-shadow: 0 2px 10px rgba(0,0,0,0.07) !important; }

/* ── Section headers ── */
h1 { font-size: 2rem !important; font-weight: 800 !important; color: #1a237e !important; }
h2 { font-size: 1.5rem !important; font-weight: 700 !important; color: #283593 !important; }
h3 { font-size: 1.2rem !important; font-weight: 700 !important; color: #3949ab !important; }
p  { font-size: 0.95rem !important; color: #37474f !important; line-height: 1.6 !important; }

/* ── Custom cards ── */
.kpi-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 3px 14px rgba(0,0,0,0.08);
    border-top: 4px solid #3949ab;
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0,0,0,0.12); }
.kpi-card .kpi-label {
    font-size: 0.78rem; font-weight: 700; color: #7986cb;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.4rem;
}
.kpi-card .kpi-value {
    font-size: 2rem; font-weight: 800; color: #1a237e; line-height: 1.1;
}
.kpi-card .kpi-sub {
    font-size: 0.82rem; color: #78909c; margin-top: 0.3rem; font-weight: 500;
}

.result-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 3px 14px rgba(0,0,0,0.09);
    margin-bottom: 1rem;
}
.rec-card {
    background: #f8f9ff;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    border-left: 4px solid #3949ab;
    margin-bottom: 0.6rem;
    font-size: 0.9rem;
    font-weight: 500;
    color: #263238;
}
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1a237e;
    margin: 1.2rem 0 0.6rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #e8eaf6;
}
.badge-high   { background:#ffebee; color:#c62828; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.85rem; }
.badge-mod    { background:#fff8e1; color:#e65100; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.85rem; }
.badge-low    { background:#e8f5e9; color:#2e7d32; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.85rem; }
.divider { border:none; border-top:2px solid #e8eaf6; margin:1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS & LOOKUPS
# ─────────────────────────────────────────────────────────────
CITY_COORDS = {
    "delhi": (28.6139, 77.2090), "mumbai": (19.0760, 72.8777),
    "bangalore": (12.9716, 77.5946), "bengaluru": (12.9716, 77.5946),
    "chennai": (13.0827, 80.2707), "hyderabad": (17.3850, 78.4867),
    "pune": (18.5204, 73.8567), "chandigarh": (30.7333, 76.7794),
    "kolkata": (22.5726, 88.3639), "ahmedabad": (23.0225, 72.5714),
    "jaipur": (26.9124, 75.7873), "lucknow": (26.8467, 80.9462),
    "surat": (21.1702, 72.8311), "nagpur": (21.1458, 79.0882),
    "indore": (22.7196, 75.8577), "bhopal": (23.2599, 77.4126),
    "patna": (25.5941, 85.1376), "agra": (27.1767, 78.0081),
    "varanasi": (25.3176, 82.9739), "coimbatore": (11.0168, 76.9558),
}

WEATHER_RISK   = {'clear': 0, 'cloudy': 1, 'rain': 2, 'fog': 3, 'hail': 3, 'storm': 4, 'snow': 4}
VISIBILITY_MAP = {'high': 0, 'medium': 1, 'low': 2}
TRAFFIC_MAP    = {'low': 0, 'medium': 1, 'high': 2}
ROAD_COND_MAP  = {'good': 0, 'under_construction': 1, 'damaged': 2}
ROAD_TYPES     = ['highway', 'urban', 'rural', 'expressway', 'mountain']
ROAD_CONDS     = ['good', 'under_construction', 'damaged']
FESTIVALS      = ['no_festival', 'diwali', 'holi', 'eid', 'christmas', 'navratri']
NON_DRIVING    = {'Weather-Related', 'Road Infrastructure', 'Visibility-Related'}

WEATHER_EMOJI  = {'clear':'☀️','cloudy':'⛅','rain':'🌧️','fog':'🌫️','hail':'🌨️','storm':'⛈️','snow':'❄️'}
ROAD_TYPE_EMOJI= {'highway':'🛣️','urban':'🏙️','rural':'🌾','expressway':'🚀','mountain':'⛰️'}

FACTOR_DETAILS = {
    'Weather-Related':     ('🌧️', 'Weather-Related',     ['Adverse weather (rain/storm/hail/snow)', 'Wet or icy road surface reducing traction', 'Reduced visibility due to heavy rain or storm']),
    'Visibility-Related':  ('🌫️', 'Visibility-Related',  ['Dense fog reducing sight distance', 'Glare from sun or oncoming headlights', 'Smoke or dust reducing visibility']),
    'Road Infrastructure': ('🛣️', 'Road Infrastructure',  ['Damaged or uneven road surface / potholes', 'Active construction zones with reduced lanes', 'Poor road design, markings or signage']),
    'Driving Behavior':    ('🚗', 'Driving Behavior',    ['Overspeeding or reckless driving', 'Driver distraction, fatigue or phone use', 'Drunk driving or impaired judgment']),
}

RECOMMENDATIONS = {
    'Weather-Related':     ['Reduce speed significantly in rain or storm', 'Use headlights and hazard lights', 'Maintain extra following distance (3-second rule)', 'Avoid travel if conditions are severe'],
    'Visibility-Related':  ['Use fog lights — not high beams in fog', 'Reduce speed and increase following distance', 'Stay in lane and avoid sudden overtaking', 'Pull over safely if visibility drops to near zero'],
    'Road Infrastructure': ['Watch for potholes and uneven surfaces', 'Slow down near construction zones', 'Follow diversion signs carefully', 'Avoid sudden braking on damaged roads'],
    'Driving Behavior':    ['Strictly follow posted speed limits', 'Never use phone while driving', 'Take a 15-min break every 2 hours on long trips', 'Never drive under the influence of alcohol or drugs'],
}

SAFETY_TIPS = [
    "🔦 Always carry a reflective triangle and torch in your vehicle.",
    "🛡️ Wear seatbelts — they reduce fatality risk by 45%.",
    "📱 Phone use while driving increases crash risk by 4×.",
    "🌙 Night driving (10 PM–5 AM) accounts for 35% of fatal accidents.",
    "🌧️ Wet roads increase stopping distance by up to 2×.",
    "⛽ Never drive on an empty tank — plan fuel stops in advance.",
    "🚦 Running red lights causes 22% of urban intersection accidents.",
    "😴 Drowsy driving is as dangerous as drunk driving.",
]

# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    reg       = joblib.load('models/regression_model.pkl')
    cls       = joblib.load('models/classification_model.pkl')
    sc_r      = joblib.load('models/scaler_reg.pkl')
    sc_c      = joblib.load('models/scaler_cls.pkl')
    le        = joblib.load('models/label_encoder.pkl')
    feats     = joblib.load('models/features.pkl')
    cls_feats = joblib.load('models/cls_features.pkl') if os.path.exists('models/cls_features.pkl') else feats
    return reg, cls, sc_r, sc_c, le, feats, cls_feats

@st.cache_data
def load_metrics():
    import json
    if os.path.exists('models/metrics.json'):
        with open('models/metrics.json', 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_dataset():
    if os.path.exists('processed_dataset.csv'):
        return pd.read_csv('processed_dataset.csv')
    return None

models_ready = os.path.exists('models/regression_model.pkl')

# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING HELPERS
# ─────────────────────────────────────────────────────────────
def build_input_row(lat, lon, hour, is_weekend, is_peak, weather, visibility,
                    traffic_density, temperature, lanes, traffic_signal,
                    road_type, road_condition, festival, features):
    is_night        = 1 if (hour >= 20 or hour <= 5) else 0
    is_morning_rush = 1 if 7 <= hour <= 9 else 0
    is_evening_rush = 1 if 17 <= hour <= 19 else 0
    hour_sin        = np.sin(2 * np.pi * hour / 24)
    hour_cos        = np.cos(2 * np.pi * hour / 24)
    month           = 6
    month_sin       = np.sin(2 * np.pi * month / 12)
    month_cos       = np.cos(2 * np.pi * month / 12)
    day_of_year     = 180
    weather_risk    = WEATHER_RISK.get(weather, 1)
    vis_enc         = VISIBILITY_MAP.get(visibility, 1)
    traffic_enc     = TRAFFIC_MAP.get(traffic_density, 1)
    road_cond_enc   = ROAD_COND_MAP.get(road_condition, 0)
    risk_interaction= weather_risk * vis_enc
    night_fog       = is_night * (1 if weather == 'fog' else 0)
    peak_high       = is_peak * traffic_enc
    temp_log        = np.log1p(max(temperature, 0))

    base = {
        'latitude': lat, 'longitude': lon, 'hour': hour,
        'is_weekend': is_weekend, 'is_peak_hour': is_peak,
        'is_night': is_night, 'is_morning_rush': is_morning_rush,
        'is_evening_rush': is_evening_rush,
        'hour_sin': hour_sin, 'hour_cos': hour_cos,
        'month_sin': month_sin, 'month_cos': month_cos,
        'month': month, 'day_of_year': day_of_year,
        'weather_risk': weather_risk, 'visibility_enc': vis_enc,
        'traffic_density_enc': traffic_enc, 'road_cond_enc': road_cond_enc,
        'temperature': temperature, 'lanes': lanes,
        'traffic_signal': traffic_signal,
        'risk_interaction': risk_interaction,
        'night_fog': night_fog, 'peak_high_traffic': peak_high,
        'temperature_log': temp_log,
    }
    for rt in ROAD_TYPES:
        base[f'road_{rt}'] = 1 if road_type == rt else 0
    for rc in ROAD_CONDS:
        base[f'cond_{rc}'] = 1 if road_condition == rc else 0
    for fv in FESTIVALS:
        base[f'festival_{fv}'] = 1 if festival == fv else 0

    row = {f: base.get(f, 0) for f in features}
    return pd.DataFrame([row])


def build_cls_row(input_df, weather, visibility, road_type, road_condition, temperature, cls_features):
    X = input_df.copy()
    is_night = int(X.get('is_night', pd.Series([0])).values[0])
    wr       = float(X.get('weather_risk', pd.Series([0])).values[0])
    ve       = float(X.get('visibility_enc', pd.Series([0])).values[0])

    X['is_fog']            = 1 if weather == 'fog' else 0
    X['is_rain']           = 1 if weather == 'rain' else 0
    X['is_storm']          = 1 if weather in ['storm', 'hail', 'snow'] else 0
    X['is_low_vis']        = 1 if visibility == 'low' else 0
    X['is_highway']        = 1 if road_type == 'highway' else 0
    X['is_rural']          = 1 if road_type == 'rural' else 0
    X['is_damaged']        = 1 if road_condition == 'damaged' else 0
    X['is_construction']   = 1 if road_condition == 'under_construction' else 0
    X['fog_night']         = X['is_fog'] * is_night
    X['rain_highway']      = X['is_rain'] * X['is_highway']
    X['low_vis_night']     = X['is_low_vis'] * is_night
    X['weather_x_vis']     = wr * ve
    X['temp_risk']         = 1 if temperature < 15 else 0
    X['damaged_night']     = X['is_damaged'] * is_night
    X['construction_rain'] = X['is_construction'] * X['is_rain']

    for f in cls_features:
        if f not in X.columns:
            X[f] = 0
    return X[cls_features].fillna(0)


def predict_pipeline(lat, lon, hour, is_weekend, is_peak, weather, visibility,
                     traffic_density, temperature, lanes, traffic_signal,
                     road_type, road_condition, festival,
                     reg, cls, sc_r, sc_c, le, features, cls_features):
    input_df   = build_input_row(lat, lon, hour, is_weekend, is_peak, weather, visibility,
                                 traffic_density, temperature, lanes, traffic_signal,
                                 road_type, road_condition, festival, features)
    risk_score = float(reg.predict(sc_r.transform(input_df[features]))[0])
    risk_score = np.clip(risk_score, 0, 1)

    X_cls      = build_cls_row(input_df, weather, visibility, road_type, road_condition, temperature, cls_features)
    proba      = cls.predict_proba(sc_c.transform(X_cls))[0]
    class_names= list(le.classes_)
    proba_dict = dict(zip(class_names, proba))
    risk_category = class_names[int(np.argmax(proba))]
    return risk_score, risk_category, proba_dict


def risk_color(score):
    if score < 0.4:  return '#2e7d32'
    if score < 0.6:  return '#e65100'
    return '#c62828'

def risk_label(score):
    if score < 0.4:  return 'LOW RISK'
    if score < 0.6:  return 'MODERATE RISK'
    return 'HIGH RISK'

def risk_badge(score):
    if score < 0.4:  return '<span class="badge-low">🟢 Low Risk</span>'
    if score < 0.6:  return '<span class="badge-mod">🟡 Moderate Risk</span>'
    return '<span class="badge-high">🔴 High Risk</span>'

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <div style='font-size:3rem;'>🚦</div>
        <div style='font-size:1.1rem; font-weight:800; color:#ffffff; letter-spacing:0.5px;'>Road Risk Intelligence</div>
        <div style='font-size:0.75rem; color:#9fa8da; margin-top:4px; font-weight:500;'>India Safety Platform v2.0</div>
    </div>
    <hr style='border-color:#3949ab; margin:0.8rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.78rem; font-weight:700; color:#7986cb; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.4rem;'>Navigation</p>", unsafe_allow_html=True)
    page = st.radio(
        "Go to",
        ["🏠 Dashboard", "🔮 Risk Assessment", "📂 Batch Prediction", "📊 Data Analytics", "📈 Model Performance", "ℹ️ About & Model"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#3949ab; margin:1rem 0;'>", unsafe_allow_html=True)

    # Live dataset stats in sidebar
    ds_side = load_dataset()
    if ds_side is not None:
        total   = len(ds_side)
        high_r  = int((ds_side['risk_score'] >= 0.6).sum()) if 'risk_score' in ds_side.columns else 0
        avg_r   = ds_side['risk_score'].mean() if 'risk_score' in ds_side.columns else 0
        cities  = ds_side['city'].nunique() if 'city' in ds_side.columns else 0

        st.markdown("<p style='font-size:0.78rem; font-weight:700; color:#7986cb; text-transform:uppercase; letter-spacing:1px; margin-bottom:0.6rem;'>Dataset Snapshot</p>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.07); border-radius:10px; padding:0.8rem 1rem; margin-bottom:0.5rem;'>
            <div style='font-size:0.78rem; color:#9fa8da; font-weight:600;'>TOTAL RECORDS</div>
            <div style='font-size:1.4rem; font-weight:800; color:#ffffff;'>{total:,}</div>
        </div>
        <div style='background:rgba(255,255,255,0.07); border-radius:10px; padding:0.8rem 1rem; margin-bottom:0.5rem;'>
            <div style='font-size:0.78rem; color:#9fa8da; font-weight:600;'>HIGH RISK INCIDENTS</div>
            <div style='font-size:1.4rem; font-weight:800; color:#ef9a9a;'>{high_r:,} <span style='font-size:0.85rem; color:#9fa8da;'>({high_r/total*100:.1f}%)</span></div>
        </div>
        <div style='background:rgba(255,255,255,0.07); border-radius:10px; padding:0.8rem 1rem; margin-bottom:0.5rem;'>
            <div style='font-size:0.78rem; color:#9fa8da; font-weight:600;'>AVG RISK SCORE</div>
            <div style='font-size:1.4rem; font-weight:800; color:#fff176;'>{avg_r:.1%}</div>
        </div>
        <div style='background:rgba(255,255,255,0.07); border-radius:10px; padding:0.8rem 1rem;'>
            <div style='font-size:0.78rem; color:#9fa8da; font-weight:600;'>CITIES COVERED</div>
            <div style='font-size:1.4rem; font-weight:800; color:#a5d6a7;'>{cities}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#3949ab; margin:1rem 0;'>", unsafe_allow_html=True)

    # Rotating safety tip
    tip_idx = datetime.now().minute % len(SAFETY_TIPS)
    st.markdown(f"""
    <div style='background:rgba(57,73,171,0.35); border-radius:10px; padding:0.9rem 1rem; border-left:3px solid #7986cb;'>
        <div style='font-size:0.72rem; font-weight:700; color:#9fa8da; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:0.4rem;'>Safety Tip</div>
        <div style='font-size:0.83rem; color:#e8eaf6; line-height:1.5; font-weight:500;'>{SAFETY_TIPS[tip_idx]}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; margin-top:1.5rem; font-size:0.72rem; color:#5c6bc0; font-weight:500;'>
        Powered by XGBoost + LightGBM<br>Trained on 20,000 Indian road records
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODEL GUARD
# ─────────────────────────────────────────────────────────────
if not models_ready:
    st.error("🔴 **Models not found.** Please run `retrain_models.py` first to initialize the system.")
    st.stop()

reg, cls, sc_r, sc_c, le, features, cls_features = load_models()

# ═════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    # Hero banner
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0f0c29 0%, #1a237e 50%, #283593 100%);
                border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem;
                box-shadow: 0 8px 32px rgba(26,35,126,0.35);'>
        <div style='display:flex; align-items:center; gap:1rem; margin-bottom:0.8rem;'>
            <span style='font-size:3rem;'>🚦</span>
            <div>
                <div style='font-size:2rem; font-weight:800; color:#ffffff; line-height:1.1; letter-spacing:-0.5px;'>
                    India Road Accident Risk Intelligence
                </div>
                <div style='font-size:1rem; color:#9fa8da; margin-top:0.3rem; font-weight:500;'>
                    AI-Powered Safety Platform &nbsp;|&nbsp; Two-Stage ML Pipeline &nbsp;|&nbsp; Real-Time Risk Assessment
                </div>
            </div>
        </div>
        <div style='display:flex; gap:2rem; margin-top:1.2rem; flex-wrap:wrap;'>
            <div style='background:rgba(255,255,255,0.1); border-radius:8px; padding:0.5rem 1rem;'>
                <span style='color:#a5d6a7; font-weight:700; font-size:0.85rem;'>✅ Models Loaded</span>
            </div>
            <div style='background:rgba(255,255,255,0.1); border-radius:8px; padding:0.5rem 1rem;'>
                <span style='color:#fff176; font-weight:700; font-size:0.85rem;'>⚡ Real-Time Prediction</span>
            </div>
            <div style='background:rgba(255,255,255,0.1); border-radius:8px; padding:0.5rem 1rem;'>
                <span style='color:#ef9a9a; font-weight:700; font-size:0.85rem;'>🗺️ Pan-India Coverage</span>
            </div>
            <div style='background:rgba(255,255,255,0.1); border-radius:8px; padding:0.5rem 1rem;'>
                <span style='color:#80cbc4; font-weight:700; font-size:0.85rem;'>📊 Batch Analytics</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ds = load_dataset()
    if ds is not None:
        # ── KPI Row ──
        total   = len(ds)
        high_r  = int((ds['risk_score'] >= 0.6).sum()) if 'risk_score' in ds.columns else 0
        mod_r   = int(((ds['risk_score'] >= 0.4) & (ds['risk_score'] < 0.6)).sum()) if 'risk_score' in ds.columns else 0
        low_r   = int((ds['risk_score'] < 0.4).sum()) if 'risk_score' in ds.columns else 0
        avg_r   = ds['risk_score'].mean() if 'risk_score' in ds.columns else 0
        cities  = ds['city'].nunique() if 'city' in ds.columns else 0
        fatal   = int((ds['accident_severity'] == 'fatal').sum()) if 'accident_severity' in ds.columns else 0

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        kpis = [
            (c1, "Total Records", f"{total:,}", "Accident incidents", "#3949ab"),
            (c2, "High Risk", f"{high_r:,}", f"{high_r/total*100:.1f}% of total", "#c62828"),
            (c3, "Moderate Risk", f"{mod_r:,}", f"{mod_r/total*100:.1f}% of total", "#e65100"),
            (c4, "Low Risk", f"{low_r:,}", f"{low_r/total*100:.1f}% of total", "#2e7d32"),
            (c5, "Avg Risk Score", f"{avg_r:.1%}", "Across all records", "#6a1b9a"),
            (c6, "Cities Covered", f"{cities}", "Major Indian cities", "#00695c"),
        ]
        for col, label, value, sub, color in kpis:
            with col:
                st.markdown(f"""
                <div class='kpi-card' style='border-top-color:{color};'>
                    <div class='kpi-label'>{label}</div>
                    <div class='kpi-value' style='color:{color};'>{value}</div>
                    <div class='kpi-sub'>{sub}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Row 2: Risk gauge + City map ──
        col_gauge, col_map = st.columns([1, 1.6])

        with col_gauge:
            st.markdown("<div class='section-header'>📊 Overall Risk Distribution</div>", unsafe_allow_html=True)
            fig_donut = go.Figure(go.Pie(
                labels=['🔴 High Risk', '🟡 Moderate Risk', '🟢 Low Risk'],
                values=[high_r, mod_r, low_r],
                hole=0.62,
                marker_colors=['#ef5350', '#ffa726', '#66bb6a'],
                textfont=dict(size=13, family='Inter', color='#1a1a2e'),
                hovertemplate='%{label}<br>Count: %{value:,}<br>Share: %{percent}<extra></extra>'
            ))
            fig_donut.add_annotation(
                text=f"<b>{avg_r:.0%}</b><br><span style='font-size:11px'>Avg Risk</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=22, color='#1a237e', family='Inter')
            )
            fig_donut.update_layout(
                height=320, margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor='white', showlegend=True,
                legend=dict(font=dict(size=12, family='Inter', color='#37474f'),
                            orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5)
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_map:
            st.markdown("<div class='section-header'>🗺️ Accident Hotspot Map</div>", unsafe_allow_html=True)
            if 'latitude' in ds.columns and 'longitude' in ds.columns and 'risk_score' in ds.columns:
                map_df = ds[['latitude', 'longitude', 'risk_score', 'city']].dropna().sample(min(3000, len(ds)), random_state=42)
                fig_map = px.scatter_mapbox(
                    map_df, lat='latitude', lon='longitude',
                    color='risk_score', size='risk_score',
                    color_continuous_scale=['#66bb6a', '#ffa726', '#ef5350'],
                    range_color=[0, 1],
                    size_max=10, zoom=4.2,
                    center=dict(lat=22.5, lon=80.0),
                    mapbox_style='carto-positron',
                    hover_data={'city': True, 'risk_score': ':.2f', 'latitude': False, 'longitude': False},
                    labels={'risk_score': 'Risk Score', 'city': 'City'},
                )
                fig_map.update_layout(
                    height=320, margin=dict(t=0, b=0, l=0, r=0),
                    coloraxis_colorbar=dict(
                        title='Risk', tickfont=dict(size=11, family='Inter'),
                        len=0.7, thickness=12
                    )
                )
                st.plotly_chart(fig_map, use_container_width=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Row 3: Hourly trend + Weather risk ──
        col_hr, col_wx = st.columns(2)

        with col_hr:
            st.markdown("<div class='section-header'>⏰ Hourly Risk Pattern</div>", unsafe_allow_html=True)
            if 'hour' in ds.columns and 'risk_score' in ds.columns:
                hourly = ds.groupby('hour')['risk_score'].agg(['mean', 'count']).reset_index()
                hourly.columns = ['hour', 'avg_risk', 'count']
                fig_hr = go.Figure()
                fig_hr.add_trace(go.Scatter(
                    x=hourly['hour'], y=hourly['avg_risk'],
                    mode='lines+markers',
                    line=dict(color='#3949ab', width=3),
                    marker=dict(size=7, color=hourly['avg_risk'],
                                colorscale=['#66bb6a', '#ffa726', '#ef5350'],
                                cmin=0, cmax=1, line=dict(width=1.5, color='white')),
                    fill='tozeroy', fillcolor='rgba(57,73,171,0.08)',
                    hovertemplate='Hour %{x}:00<br>Avg Risk: %{y:.1%}<extra></extra>'
                ))
                fig_hr.add_hline(y=0.6, line_dash='dash', line_color='#ef5350',
                                 annotation_text='High Risk Threshold', annotation_font_size=11)
                fig_hr.add_hline(y=0.4, line_dash='dash', line_color='#ffa726',
                                 annotation_text='Moderate Threshold', annotation_font_size=11)
                fig_hr.update_layout(
                    height=300, paper_bgcolor='white', plot_bgcolor='#fafafa',
                    xaxis=dict(title='Hour of Day', tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                    yaxis=dict(title='Avg Risk Score', tickformat='.0%', tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                    margin=dict(t=10, b=40, l=50, r=20),
                    hovermode='x unified'
                )
                st.plotly_chart(fig_hr, use_container_width=True)

        with col_wx:
            st.markdown("<div class='section-header'>🌦️ Risk by Weather Condition</div>", unsafe_allow_html=True)
            if 'weather' in ds.columns and 'risk_score' in ds.columns:
                wx_stats = ds.groupby('weather')['risk_score'].agg(['mean', 'count']).reset_index()
                wx_stats.columns = ['weather', 'avg_risk', 'count']
                wx_stats = wx_stats.sort_values('avg_risk', ascending=True)
                wx_stats['emoji'] = wx_stats['weather'].map(WEATHER_EMOJI).fillna('🌤️')
                wx_stats['label'] = wx_stats['emoji'] + ' ' + wx_stats['weather'].str.capitalize()
                fig_wx = px.bar(
                    wx_stats, x='avg_risk', y='label', orientation='h',
                    color='avg_risk', color_continuous_scale=['#66bb6a', '#ffa726', '#ef5350'],
                    range_color=[0, 1],
                    text=wx_stats['avg_risk'].apply(lambda v: f'{v:.0%}'),
                    labels={'avg_risk': 'Avg Risk Score', 'label': ''}
                )
                fig_wx.update_traces(textposition='outside',
                                     hovertemplate='%{y}<br>Avg Risk: %{x:.1%}<extra></extra>',
                                     textfont=dict(size=11, family='Inter', color='#1a1a2e'))
                fig_wx.update_layout(
                    height=300, paper_bgcolor='white', plot_bgcolor='#fafafa',
                    coloraxis_showscale=False, showlegend=False,
                    xaxis=dict(tickformat='.0%', tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                    yaxis=dict(tickfont=dict(size=12, family='Inter', color='#37474f')),
                    margin=dict(t=10, b=40, l=10, r=60)
                )
                st.plotly_chart(fig_wx, use_container_width=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Row 4: City comparison + Category breakdown ──
        col_city, col_cat = st.columns(2)

        with col_city:
            st.markdown("<div class='section-header'>🏙️ City-wise Risk Comparison</div>", unsafe_allow_html=True)
            if 'city' in ds.columns and 'risk_score' in ds.columns:
                city_stats = ds.groupby('city')['risk_score'].agg(['mean', 'count']).reset_index()
                city_stats.columns = ['city', 'avg_risk', 'incidents']
                city_stats = city_stats.sort_values('avg_risk', ascending=False)
                fig_city = px.bar(
                    city_stats, x='city', y='avg_risk',
                    color='avg_risk', color_continuous_scale=['#66bb6a', '#ffa726', '#ef5350'],
                    range_color=[0, 1],
                    text=city_stats['avg_risk'].apply(lambda v: f'{v:.0%}'),
                    labels={'avg_risk': 'Avg Risk Score', 'city': 'City'},
                    custom_data=['incidents']
                )
                fig_city.update_traces(
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Avg Risk: %{y:.1%}<br>Incidents: %{customdata[0]:,}<extra></extra>',
                    textfont=dict(size=10, family='Inter', color='#1a1a2e')
                )
                fig_city.update_layout(
                    height=300, paper_bgcolor='white', plot_bgcolor='#fafafa',
                    coloraxis_showscale=False,
                    xaxis=dict(tickfont=dict(size=11, family='Inter', color='#37474f'), tickangle=-30),
                    yaxis=dict(tickformat='.0%', tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                    margin=dict(t=10, b=60, l=50, r=20)
                )
                st.plotly_chart(fig_city, use_container_width=True)

        with col_cat:
            st.markdown("<div class='section-header'>🔍 Risk Category Breakdown</div>", unsafe_allow_html=True)
            if 'risk_category' in ds.columns:
                cat_counts = ds['risk_category'].value_counts().reset_index()
                cat_counts.columns = ['category', 'count']
                cat_colors = {'Weather-Related': '#42a5f5', 'Visibility-Related': '#ab47bc',
                              'Road Infrastructure': '#ff7043', 'Driving Behavior': '#26a69a'}
                fig_cat = px.bar(
                    cat_counts, x='count', y='category', orientation='h',
                    color='category',
                    color_discrete_map=cat_colors,
                    text='count',
                    labels={'count': 'Number of Incidents', 'category': ''}
                )
                fig_cat.update_traces(
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Incidents: %{x:,}<extra></extra>',
                    textfont=dict(size=11, family='Inter', color='#1a1a2e')
                )
                fig_cat.update_layout(
                    height=300, paper_bgcolor='white', plot_bgcolor='#fafafa',
                    showlegend=False,
                    xaxis=dict(tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                    yaxis=dict(tickfont=dict(size=12, family='Inter', color='#37474f')),
                    margin=dict(t=10, b=40, l=10, r=60)
                )
                st.plotly_chart(fig_cat, use_container_width=True)

    else:
        st.info("📌 Run `retrain_models.py` first to generate the processed dataset.")

# ═════════════════════════════════════════════════════════════
# PAGE 2 — RISK ASSESSMENT
# ═════════════════════════════════════════════════════════════
elif page == "🔮 Risk Assessment":
    st.markdown("<h1 style='text-align:center; margin-bottom:0.3rem;'>🔮 Real-Time Risk Assessment</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1rem; color:#5c6bc0; margin-bottom:2rem;'>Enter trip details to get instant accident risk prediction and safety recommendations</p>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Location ──
    with st.expander("📍 **Location Details**", expanded=True):
        loc_mode = st.radio("Input method", ["City Name", "Coordinates"], horizontal=True)
        if loc_mode == "City Name":
            city_input = st.text_input("🏙️ City name", value="Delhi", placeholder="Enter city (e.g., Delhi, Mumbai, Bangalore)")
            city_key   = city_input.strip().lower()
            if city_key in CITY_COORDS:
                lat, lon = CITY_COORDS[city_key]
                st.success(f"✅ **{city_input.title()}** — Lat: {lat:.4f}, Lon: {lon:.4f}")
            else:
                st.warning(f"⚠️ '{city_input}' not in database. Using Delhi as fallback.")
                lat, lon = CITY_COORDS["delhi"]
        else:
            col_a, col_b = st.columns(2)
            lat = col_a.number_input("📌 Latitude", value=28.6139, min_value=-90.0, max_value=90.0, format="%.4f")
            lon = col_b.number_input("📌 Longitude", value=77.2090, min_value=-180.0, max_value=180.0, format="%.4f")

    # ── Time ──
    with st.expander("⏰ **Time of Travel**", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            hour = st.slider("Hour (24h format)", 0, 23, 8, format="%d:00")
        with col2:
            is_weekend = st.selectbox("Day Type", [0, 1], format_func=lambda x: "🏖️ Weekend" if x else "📅 Weekday")
        with col3:
            is_peak = st.selectbox("Peak Hour?", [0, 1], format_func=lambda x: "🚗 Yes" if x else "☑️ No")
        auto_peak = 1 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
        if auto_peak != is_peak:
            st.info(f"💡 Tip: Hour {hour}:00 is {'a peak' if auto_peak else 'not a peak'} travel hour.")

    # ── Weather ──
    with st.expander("🌦️ **Weather Conditions**", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            weather = st.selectbox("🌤️ Weather Type", list(WEATHER_RISK.keys()), format_func=str.capitalize, index=0)
        with col2:
            visibility = st.selectbox("👁️ Visibility", list(VISIBILITY_MAP.keys()), format_func=str.capitalize, index=0)
        with col3:
            temperature = st.slider("🌡️ Temperature (°C)", 5, 50, 28)

    # ── Road Conditions ──
    with st.expander("🛣️ **Road Conditions**", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            road_type = st.selectbox("🛣️ Road Type", ROAD_TYPES, format_func=str.capitalize)
        with col2:
            road_condition = st.selectbox("🔧 Road Condition", ROAD_CONDS, format_func=lambda x: x.replace('_', ' ').title())

    # ── Traffic & Context ──
    with st.expander("🚗 **Traffic & Driving Context**", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            traffic_den = st.selectbox("🚦 Traffic Level", list(TRAFFIC_MAP.keys()), format_func=str.capitalize, index=1)
        with col2:
            lanes = st.slider("🛣️ Road Lanes", 1, 8, 3)
        with col3:
            traffic_sig = st.selectbox("🚥 Traffic Signal?", [0, 1], format_func=lambda x: "✅ Yes" if x else "❌ No")
        with col4:
            festival = st.selectbox("🎉 Festival", FESTIVALS, format_func=lambda x: x.replace('_', ' ').title())

    # Prediction button
    st.markdown("""
    <style>
    div.stButton > button {
        color: white !important;              /* Text color */
        background-color: #1a237e !important; /* Background color */
        font-size: 18px !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        height: 3em;
    }

    div.stButton > button:hover {
        background-color: #283593 !important; /* Hover color */
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)


    col_btn_left, col_btn_center, col_btn_right = st.columns([1, 2, 1])
    with col_btn_center:
        predict_btn = st.button("🚀 Assess Risk Now", use_container_width=True, type="primary")

    if predict_btn:
        score, category, proba_dict = predict_pipeline(
            lat, lon, hour, is_weekend, is_peak, weather, visibility,
            traffic_den, temperature, lanes, traffic_sig,
            road_type, road_condition, festival,
            reg, cls, sc_r, sc_c, le, features, cls_features
        )
        color = risk_color(score)
        label = risk_label(score)
        icon, cat_name, factors = FACTOR_DETAILS[category]
        recs = RECOMMENDATIONS[category]

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align:center; color:#1a237e;'>📊 Risk Assessment Results</h2>", unsafe_allow_html=True)

    

        # ── Risk Score & Status ──
        col_gauge, col_status = st.columns([1.2, 1.8])

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(score * 100, 1),
                number={'suffix': '%', 'font': {'size': 52, 'color': color, 'family': 'Inter'}},
                title={'text': f"<b>Accident Risk</b><br><span style='font-size:14px; color:#78909c;'>Score</span>",
                       'font': {'size': 18, 'family': 'Inter', 'color': '#1a237e'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': '#e0e0e0', 'tickfont': {'size': 11, 'family': 'Inter'}},
                    'bar': {'color': color, 'thickness': 0.5},
                    'bgcolor': '#f5f5f5',
                    'steps': [
                        {'range': [0, 40],  'color': '#e8f5e9'},
                        {'range': [40, 60], 'color': '#fff8e1'},
                        {'range': [60, 100],'color': '#ffebee'},
                    ],
                    'threshold': {'line': {'color': '#1a237e', 'width': 4}, 'value': 60}
                }
            ))
            fig.update_layout(height=320, margin=dict(t=70, b=10, l=20, r=20), paper_bgcolor='white', font=dict(family='Inter'))
            st.plotly_chart(fig, use_container_width=True)

        with col_status:
            if score >= 0.6:
                st.markdown(f"""
                <div class='result-card' style='border-left:5px solid #c62828;'>
                    <div style='font-size:1.8rem; font-weight:800; color:#c62828; margin-bottom:0.5rem;'>
                        🚨 HIGH ACCIDENT RISK
                    </div>
                    <div style='font-size:2.5rem; font-weight:800; color:#d32f2f; margin:0.5rem 0;'>
                        {score:.0%}
                    </div>
                    <div style='font-size:0.95rem; color:#555; font-weight:500;'>
                        ⚠️ Immediate precautions strongly recommended. Consider postponing travel if possible.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif score >= 0.4:
                st.markdown(f"""
                <div class='result-card' style='border-left:5px solid #e65100;'>
                    <div style='font-size:1.8rem; font-weight:800; color:#e65100; margin-bottom:0.5rem;'>
                        ⚠️ MODERATE RISK
                    </div>
                    <div style='font-size:2.5rem; font-weight:800; color:#f57c00; margin:0.5rem 0;'>
                        {score:.0%}
                    </div>
                    <div style='font-size:0.95rem; color:#555; font-weight:500;'>
                        🛡️ Exercise caution and follow all safety guidelines carefully.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-card' style='border-left:5px solid #2e7d32;'>
                    <div style='font-size:1.8rem; font-weight:800; color:#2e7d32; margin-bottom:0.5rem;'>
                        ✅ LOW RISK
                    </div>
                    <div style='font-size:2.5rem; font-weight:800; color:#388e3c; margin:0.5rem 0;'>
                        {score:.0%}
                    </div>
                    <div style='font-size:0.95rem; color:#555; font-weight:500;'>
                        ✔️ Safe to proceed with standard precautions. Stay alert.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Primary Risk Factor ──
        col_factor, col_proba = st.columns([1, 1])

        with col_factor:
            st.markdown("<div class='section-header'>🔍 Primary Risk Factor</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='result-card'>
                <div style='font-size:1.3rem; font-weight:700; color:#1a237e; margin-bottom:0.8rem;'>
                    {icon} {cat_name}
                </div>
                <ul style='margin:0; padding-left:1.3rem; color:#37474f; font-size:0.92rem; line-height:1.8;'>
            """, unsafe_allow_html=True)
            for factor in factors:
                st.markdown(f"<li style='margin-bottom:0.3rem;'>{factor}</li>", unsafe_allow_html=True)
            st.markdown("</ul></div>", unsafe_allow_html=True)

        with col_proba:
            st.markdown("<div class='section-header'>📈 Risk Category Probabilities</div>", unsafe_allow_html=True)
            prob_df = pd.DataFrame(list(proba_dict.items()), columns=['Category', 'Probability'])
            prob_df = prob_df.sort_values('Probability', ascending=True)
            fig2 = px.bar(
                prob_df, x='Probability', y='Category', orientation='h',
                color='Probability',
                color_continuous_scale=['#66bb6a', '#ffa726', '#ef5350'],
                range_x=[0, 1],
                text=prob_df['Probability'].apply(lambda x: f"{x:.0%}"),
                labels={'Probability': 'Probability', 'Category': ''}
            )
            fig2.update_traces(textposition='outside', hovertemplate='%{y}<br>%{x:.0%}<extra></extra>',
                               textfont=dict(size=12, family='Inter', color='#1a1a2e'))
            fig2.update_layout(
                height=260, showlegend=False, coloraxis_showscale=False,
                margin=dict(t=10, b=10, l=10, r=70), paper_bgcolor='white', plot_bgcolor='#fafafa',
                xaxis=dict(tickformat='.0%', tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                yaxis=dict(tickfont=dict(size=12, family='Inter', color='#37474f')),
                font=dict(family='Inter')
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Safety Recommendations ──
        st.markdown("<div class='section-header'>💡 Safety Recommendations</div>", unsafe_allow_html=True)
        rec_cols = st.columns(2)
        for i, rec in enumerate(recs):
            with rec_cols[i % 2]:
                st.markdown(f"<div class='rec-card'>✓ {rec}</div>", unsafe_allow_html=True)

        # ── Input Summary ──
        with st.expander("📋 Detailed Input Summary", expanded=False):
            summary_data = {
                "📍 Location": f"({lat:.4f}, {lon:.4f})",
                "⏰ Time": f"{hour:02d}:00 • {'Weekend' if is_weekend else 'Weekday'} • {'Peak Hour' if is_peak else 'Off-Peak'}",
                "🌦️ Weather": f"{weather.capitalize()} • Visibility: {visibility.capitalize()} • {temperature}°C",
                "🛣️ Road": f"{road_type.capitalize()} • {road_condition.replace('_', ' ').title()} • {lanes} lanes",
                "🚗 Traffic": f"{traffic_den.capitalize()} • Signal: {'Yes' if traffic_sig else 'No'} • Festival: {festival.replace('_', ' ').title()}",
            }
            for key, value in summary_data.items():
                st.markdown(f"**{key}**  \n{value}")

# ═════════════════════════════════════════════════════════════
# PAGE 3 — BATCH PREDICTION
# ═════════════════════════════════════════════════════════════
elif page == "📂 Batch Prediction":
    st.markdown("<h1 style='text-align:center; margin-bottom:0.3rem;'>📂 Batch Risk Assessment</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1rem; color:#5c6bc0; margin-bottom:2rem;'>Upload a CSV file to assess accident risk for multiple trips simultaneously</p>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Instructions card
    st.markdown("""
    <div style='background:#e8eaf6; border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1.5rem; border-left:4px solid #3949ab;'>
        <div style='font-size:0.95rem; font-weight:700; color:#1a237e; margin-bottom:0.5rem;'>📋 Required CSV Columns</div>
        <div style='font-family:"Roboto Mono", monospace; font-size:0.82rem; color:#37474f; line-height:1.8;'>
            <code>hour</code> &nbsp;|&nbsp; <code>is_weekend</code> &nbsp;|&nbsp; <code>is_peak_hour</code> &nbsp;|&nbsp;
            <code>weather</code> &nbsp;|&nbsp; <code>visibility</code> &nbsp;|&nbsp; <code>temperature</code> &nbsp;|&nbsp;
            <code>traffic_density</code> &nbsp;|&nbsp; <code>road_type</code> &nbsp;|&nbsp; <code>road_condition</code> &nbsp;|&nbsp;
            <code>lanes</code> &nbsp;|&nbsp; <code>traffic_signal</code> &nbsp;|&nbsp; <code>latitude</code> &nbsp;|&nbsp;
            <code>longitude</code> &nbsp;|&nbsp; <code>festival</code> <em>(optional)</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("📁 Upload CSV File", type=['csv'], help="Select a CSV file with road condition data")

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            batch_df.columns = batch_df.columns.str.strip().str.lower().str.replace(' ', '_')

            required_cols = ['hour', 'is_weekend', 'weather', 'visibility', 'temperature',
                             'traffic_density', 'road_type', 'road_condition', 'lanes', 'traffic_signal',
                             'latitude', 'longitude']
            missing = [c for c in required_cols if c not in batch_df.columns]

            if missing:
                st.error(f"❌ Missing required columns: `{'`, `'.join(missing)}`")
            else:
                st.success(f"✅ Loaded **{len(batch_df):,}** records successfully")

                for col in ['weather', 'visibility', 'traffic_density', 'road_type', 'road_condition']:
                    if col in batch_df.columns:
                        batch_df[col] = batch_df[col].str.strip().str.lower()
                if 'festival' in batch_df.columns:
                    batch_df['festival'] = batch_df['festival'].str.strip().str.lower().replace('none', 'no_festival')
                else:
                    batch_df['festival'] = 'no_festival'
                if 'road_condition' not in batch_df.columns:
                    batch_df['road_condition'] = 'good'

                batch_df['is_night']          = batch_df['hour'].apply(lambda x: 1 if (x >= 20 or x <= 5) else 0)
                batch_df['is_morning_rush']   = batch_df['hour'].apply(lambda x: 1 if 7 <= x <= 9 else 0)
                batch_df['is_evening_rush']   = batch_df['hour'].apply(lambda x: 1 if 17 <= x <= 19 else 0)
                batch_df['hour_sin']          = np.sin(2 * np.pi * batch_df['hour'] / 24)
                batch_df['hour_cos']          = np.cos(2 * np.pi * batch_df['hour'] / 24)
                batch_df['month']             = pd.to_datetime(batch_df.get('date', '2024-01-01'), errors='coerce').dt.month.fillna(6).astype(int)
                batch_df['day_of_year']       = pd.to_datetime(batch_df.get('date', '2024-01-01'), errors='coerce').dt.dayofyear.fillna(180).astype(int)
                batch_df['month_sin']         = np.sin(2 * np.pi * batch_df['month'] / 12)
                batch_df['month_cos']         = np.cos(2 * np.pi * batch_df['month'] / 12)
                batch_df['weather_risk']      = batch_df['weather'].map(WEATHER_RISK).fillna(1)
                batch_df['visibility_enc']    = batch_df['visibility'].map(VISIBILITY_MAP).fillna(1)
                batch_df['traffic_density_enc'] = batch_df['traffic_density'].map(TRAFFIC_MAP).fillna(1)
                batch_df['road_cond_enc']     = batch_df['road_condition'].map(ROAD_COND_MAP).fillna(0)
                batch_df['risk_interaction']  = batch_df['weather_risk'] * batch_df['visibility_enc']
                batch_df['night_fog']         = batch_df['is_night'] * (batch_df['weather'] == 'fog').astype(int)
                batch_df['peak_high_traffic'] = batch_df.get('is_peak_hour', 0) * batch_df['traffic_density_enc']
                batch_df['temperature_log']   = np.log1p(batch_df['temperature'].clip(lower=0))

                for rt in ROAD_TYPES:
                    batch_df[f'road_{rt}'] = (batch_df['road_type'] == rt).astype(int)
                for rc in ROAD_CONDS:
                    batch_df[f'cond_{rc}'] = (batch_df['road_condition'] == rc).astype(int)
                for fv in FESTIVALS:
                    batch_df[f'festival_{fv}'] = (batch_df['festival'] == fv).astype(int)

                for f in features:
                    if f not in batch_df.columns:
                        batch_df[f] = 0

                X_batch = batch_df[features].fillna(0)
                scores  = reg.predict(sc_r.transform(X_batch)).clip(0, 1)
                batch_df['predicted_risk_score'] = scores

                batch_df['is_fog']          = (batch_df['weather'] == 'fog').astype(int)
                batch_df['is_rain']         = (batch_df['weather'] == 'rain').astype(int)
                batch_df['is_storm']        = batch_df['weather'].isin(['storm', 'hail', 'snow']).astype(int)
                batch_df['is_low_vis']      = (batch_df['visibility'] == 'low').astype(int)
                batch_df['is_highway']      = (batch_df['road_type'] == 'highway').astype(int)
                batch_df['is_rural']        = (batch_df['road_type'] == 'rural').astype(int)
                batch_df['is_damaged']      = (batch_df['road_condition'] == 'damaged').astype(int)
                batch_df['is_construction'] = (batch_df['road_condition'] == 'under_construction').astype(int)
                batch_df['fog_night']       = batch_df['is_fog'] * batch_df['is_night']
                batch_df['rain_highway']    = batch_df['is_rain'] * batch_df['is_highway']
                batch_df['low_vis_night']   = batch_df['is_low_vis'] * batch_df['is_night']
                batch_df['weather_x_vis']   = batch_df['weather_risk'] * batch_df['visibility_enc']
                batch_df['temp_risk']       = (batch_df['temperature'] < 15).astype(int)
                batch_df['damaged_night']   = batch_df['is_damaged'] * batch_df['is_night']
                batch_df['construction_rain'] = batch_df['is_construction'] * batch_df['is_rain']

                for f in cls_features:
                    if f not in batch_df.columns:
                        batch_df[f] = 0

                X_cls_batch = batch_df[cls_features].fillna(0)
                cls_proba   = cls.predict_proba(sc_c.transform(X_cls_batch))
                class_names = list(le.classes_)

                categories = []
                for i in range(len(batch_df)):
                    pd_ = dict(zip(class_names, cls_proba[i]))
                    non_drv = sum(p for c, p in pd_.items() if c in NON_DRIVING)
                    if non_drv > 0.5:
                        categories.append(max({c: p for c, p in pd_.items() if c in NON_DRIVING}, key=lambda k: pd_[k]))
                    else:
                        categories.append(class_names[int(np.argmax(cls_proba[i]))])

                batch_df['predicted_risk_category'] = categories
                batch_df['risk_level'] = batch_df['predicted_risk_score'].apply(
                    lambda s: 'High' if s >= 0.6 else ('Moderate' if s >= 0.4 else 'Low'))

                # ── Summary KPIs ──
                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>📊 Prediction Summary</div>", unsafe_allow_html=True)

                total_b = len(batch_df)
                high_b  = (batch_df['predicted_risk_score'] >= 0.6).sum()
                mod_b   = ((batch_df['predicted_risk_score'] >= 0.4) & (batch_df['predicted_risk_score'] < 0.6)).sum()
                low_b   = (batch_df['predicted_risk_score'] < 0.4).sum()
                avg_b   = batch_df['predicted_risk_score'].mean()

                c1, c2, c3, c4, c5 = st.columns(5)
                kpis_b = [
                    (c1, "Total Records", f"{total_b:,}", "#3949ab"),
                    (c2, "🔴 High Risk", f"{high_b:,} ({high_b/total_b*100:.1f}%)", "#c62828"),
                    (c3, "🟡 Moderate", f"{mod_b:,} ({mod_b/total_b*100:.1f}%)", "#e65100"),
                    (c4, "🟢 Low Risk", f"{low_b:,} ({low_b/total_b*100:.1f}%)", "#2e7d32"),
                    (c5, "Avg Risk Score", f"{avg_b:.1%}", "#6a1b9a"),
                ]
                for col, label, value, color in kpis_b:
                    with col:
                        st.markdown(f"""
                        <div class='kpi-card' style='border-top-color:{color};'>
                            <div class='kpi-label'>{label}</div>
                            <div class='kpi-value' style='color:{color}; font-size:1.4rem;'>{value}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

                # ── Results table ──
                st.markdown("<div class='section-header'>📋 Results Preview (first 100 rows)</div>", unsafe_allow_html=True)
                display_cols = ['predicted_risk_score', 'risk_level', 'predicted_risk_category']
                for opt in ['hour', 'weather', 'road_type', 'city']:
                    if opt in batch_df.columns:
                        display_cols = [opt] + display_cols

                st.dataframe(
                    batch_df[display_cols].head(100).style.background_gradient(
                        subset=['predicted_risk_score'], cmap='RdYlGn_r', vmin=0, vmax=1
                    ),
                    use_container_width=True, height=380
                )

                csv_out = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Complete Results CSV", csv_out,
                                   "road_accident_predictions.csv", "text/csv", use_container_width=True)

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>📈 Analysis Charts</div>", unsafe_allow_html=True)

                # ── Charts row 1 ──
                col_hist, col_pie = st.columns(2)
                with col_hist:
                    fig_h = px.histogram(
                        batch_df, x='predicted_risk_score', nbins=40,
                        color_discrete_sequence=['#3949ab'],
                        title='Risk Score Distribution',
                        labels={'predicted_risk_score': 'Risk Score', 'count': 'Records'}
                    )
                    fig_h.add_vline(x=0.6, line_dash='dash', line_color='#c62828', annotation_text='High Risk')
                    fig_h.add_vline(x=0.4, line_dash='dash', line_color='#e65100', annotation_text='Moderate')
                    fig_h.update_layout(height=320, paper_bgcolor='white', plot_bgcolor='#fafafa',
                                        showlegend=False, hovermode='x unified',
                                        xaxis=dict(tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                                        yaxis=dict(tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                                        title_font=dict(size=14, family='Inter', color='#1a237e'))
                    st.plotly_chart(fig_h, use_container_width=True)

                with col_pie:
                    cat_counts = batch_df['predicted_risk_category'].value_counts()
                    fig_p = px.pie(
                        values=cat_counts.values, names=cat_counts.index,
                        title='Risk Category Distribution',
                        color_discrete_sequence=['#42a5f5', '#ab47bc', '#ff7043', '#26a69a'],
                        hole=0.4
                    )
                    fig_p.update_traces(textfont=dict(size=12, family='Inter'),
                                        hovertemplate='%{label}<br>Count: %{value:,}<br>%{percent}<extra></extra>')
                    fig_p.update_layout(height=320, paper_bgcolor='white',
                                        legend=dict(font=dict(size=11, family='Inter', color='#37474f')),
                                        title_font=dict(size=14, family='Inter', color='#1a237e'))
                    st.plotly_chart(fig_p, use_container_width=True)

                # ── Charts row 2 ──
                col_rl, col_wx2 = st.columns(2)
                with col_rl:
                    rl_counts = batch_df['risk_level'].value_counts().reset_index()
                    rl_counts.columns = ['level', 'count']
                    color_map = {'High': '#ef5350', 'Moderate': '#ffa726', 'Low': '#66bb6a'}
                    fig_rl = px.bar(rl_counts, x='level', y='count', color='level',
                                    color_discrete_map=color_map, text='count',
                                    title='Records by Risk Level',
                                    labels={'level': 'Risk Level', 'count': 'Records'})
                    fig_rl.update_traces(textposition='outside',
                                         textfont=dict(size=12, family='Inter', color='#1a1a2e'))
                    fig_rl.update_layout(height=300, paper_bgcolor='white', plot_bgcolor='#fafafa',
                                         showlegend=False,
                                         xaxis=dict(tickfont=dict(size=12, family='Inter', color='#37474f')),
                                         yaxis=dict(tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                                         title_font=dict(size=14, family='Inter', color='#1a237e'))
                    st.plotly_chart(fig_rl, use_container_width=True)

                with col_wx2:
                    if 'weather' in batch_df.columns:
                        wx_risk = batch_df.groupby('weather')['predicted_risk_score'].mean().reset_index()
                        wx_risk.columns = ['weather', 'avg_risk']
                        wx_risk = wx_risk.sort_values('avg_risk', ascending=False)
                        fig_wx2 = px.bar(wx_risk, x='weather', y='avg_risk',
                                         color='avg_risk',
                                         color_continuous_scale=['#66bb6a', '#ffa726', '#ef5350'],
                                         range_color=[0, 1],
                                         text=wx_risk['avg_risk'].apply(lambda v: f'{v:.0%}'),
                                         title='Avg Risk by Weather',
                                         labels={'weather': 'Weather', 'avg_risk': 'Avg Risk Score'})
                        fig_wx2.update_traces(textposition='outside',
                                              textfont=dict(size=11, family='Inter', color='#1a1a2e'))
                        fig_wx2.update_layout(height=300, paper_bgcolor='white', plot_bgcolor='#fafafa',
                                              coloraxis_showscale=False,
                                              xaxis=dict(tickfont=dict(size=11, family='Inter', color='#37474f')),
                                              yaxis=dict(tickformat='.0%', tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                                              title_font=dict(size=14, family='Inter', color='#1a237e'))
                        st.plotly_chart(fig_wx2, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")


# ═════════════════════════════════════════════════════════════
# PAGE 4 — DATA ANALYTICS
# ═════════════════════════════════════════════════════════════
elif page == "📊 Data Analytics":
    st.markdown("<h1 style='text-align:center; margin-bottom:0.3rem;'>📊 Data Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1rem; color:#5c6bc0; margin-bottom:2rem;'>Explore patterns and insights from 20,000 Indian road accident records</p>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    ds = load_dataset()
    if ds is None:
        st.info("Run `retrain_models.py` first to generate the processed dataset.")
    else:
        total   = len(ds)
        high_r  = int((ds['risk_score'] >= 0.6).sum()) if 'risk_score' in ds.columns else 0
        avg_r   = ds['risk_score'].mean() if 'risk_score' in ds.columns else 0
        cities  = ds['city'].nunique() if 'city' in ds.columns else 0
        fatal   = int((ds['accident_severity'] == 'fatal').sum()) if 'accident_severity' in ds.columns else 0
        features_n = ds.shape[1]

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, value, color in [
            (c1, "Total Records",   f"{total:,}",    "#3949ab"),
            (c2, "Features",        f"{features_n}", "#00695c"),
            (c3, "High Risk",       f"{high_r:,}",   "#c62828"),
            (c4, "Fatal Accidents", f"{fatal:,}",    "#6a1b9a"),
            (c5, "Cities",          f"{cities}",     "#e65100"),
        ]:
            with col:
                st.markdown(f"""
                <div class='kpi-card' style='border-top-color:{color};'>
                    <div class='kpi-label'>{label}</div>
                    <div class='kpi-value' style='color:{color};'>{value}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        atab1, atab2, atab3, atab4, atab5 = st.tabs([
            "📈 Risk Distribution", "🏙️ City Analysis",
            "⏰ Time Patterns", "🌦️ Weather & Road", "📋 Raw Data"
        ])

        # ── Tab 1: Risk Distribution ──
        with atab1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("<div class='section-header'>Risk Score Distribution</div>", unsafe_allow_html=True)
                if 'risk_score' in ds.columns:
                    fig = px.histogram(ds, x='risk_score', nbins=50,
                                       color_discrete_sequence=['#3949ab'],
                                       labels={'risk_score': 'Risk Score', 'count': 'Frequency'})
                    fig.add_vline(x=0.6, line_dash='dash', line_color='#c62828', annotation_text='High Risk')
                    fig.add_vline(x=0.4, line_dash='dash', line_color='#e65100', annotation_text='Moderate')
                    fig.update_layout(height=340, paper_bgcolor='white', plot_bgcolor='#fafafa',
                                      showlegend=False, hovermode='x unified',
                                      xaxis=dict(tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                                      yaxis=dict(tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'))
                    st.plotly_chart(fig, use_container_width=True)

            with col_b:
                st.markdown("<div class='section-header'>Risk Category Breakdown</div>", unsafe_allow_html=True)
                if 'risk_category' in ds.columns:
                    cat_counts = ds['risk_category'].value_counts().reset_index()
                    cat_counts.columns = ['category', 'count']
                    cat_colors = {'Weather-Related': '#42a5f5', 'Visibility-Related': '#ab47bc',
                                  'Road Infrastructure': '#ff7043', 'Driving Behavior': '#26a69a'}
                    fig = px.pie(cat_counts, values='count', names='category', hole=0.45,
                                 color='category', color_discrete_map=cat_colors)
                    fig.update_traces(textfont=dict(size=12, family='Inter'),
                                      hovertemplate='%{label}<br>Count: %{value:,}<br>%{percent}<extra></extra>')
                    fig.update_layout(height=340, paper_bgcolor='white',
                                      legend=dict(font=dict(size=11, family='Inter', color='#37474f')))
                    st.plotly_chart(fig, use_container_width=True)

            if 'accident_severity' in ds.columns and 'risk_score' in ds.columns:
                st.markdown("<div class='section-header'>Risk Score by Accident Severity</div>", unsafe_allow_html=True)
                sev_colors = {'minor': '#66bb6a', 'major': '#ffa726', 'fatal': '#ef5350'}
                fig = px.box(ds, x='accident_severity', y='risk_score',
                             color='accident_severity', color_discrete_map=sev_colors,
                             category_orders={'accident_severity': ['minor', 'major', 'fatal']},
                             labels={'accident_severity': 'Severity', 'risk_score': 'Risk Score'})
                fig.update_layout(height=320, paper_bgcolor='white', plot_bgcolor='#fafafa',
                                  showlegend=False,
                                  xaxis=dict(tickfont=dict(size=12, family='Inter', color='#37474f')),
                                  yaxis=dict(tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'))
                st.plotly_chart(fig, use_container_width=True)

        # ── Tab 2: City Analysis ──
        with atab2:
            if 'city' in ds.columns and 'risk_score' in ds.columns:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("<div class='section-header'>Average Risk Score by City</div>", unsafe_allow_html=True)
                    city_stats = ds.groupby('city')['risk_score'].agg(['mean', 'count', 'std']).reset_index()
                    city_stats.columns = ['city', 'avg_risk', 'incidents', 'std_risk']
                    city_stats = city_stats.sort_values('avg_risk', ascending=False)
                    fig = px.bar(city_stats, x='city', y='avg_risk',
                                 color='avg_risk',
                                 color_continuous_scale=['#66bb6a', '#ffa726', '#ef5350'],
                                 range_color=[0, 1], error_y='std_risk',
                                 text=city_stats['avg_risk'].apply(lambda v: f'{v:.0%}'),
                                 labels={'avg_risk': 'Avg Risk Score', 'city': 'City'},
                                 custom_data=['incidents'])
                    fig.update_traces(textposition='outside',
                                      hovertemplate='<b>%{x}</b><br>Avg Risk: %{y:.1%}<br>Incidents: %{customdata[0]:,}<extra></extra>',
                                      textfont=dict(size=10, family='Inter', color='#1a1a2e'))
                    fig.update_layout(height=360, paper_bgcolor='white', plot_bgcolor='#fafafa',
                                      coloraxis_showscale=False,
                                      xaxis=dict(tickfont=dict(size=11, family='Inter', color='#37474f'), tickangle=-30),
                                      yaxis=dict(tickformat='.0%', tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'))
                    st.plotly_chart(fig, use_container_width=True)

                with col_b:
                    st.markdown("<div class='section-header'>Incident Count by City</div>", unsafe_allow_html=True)
                    city_stats2 = city_stats.sort_values('incidents', ascending=True)
                    fig = px.bar(city_stats2, x='incidents', y='city', orientation='h',
                                 color='avg_risk',
                                 color_continuous_scale=['#66bb6a', '#ffa726', '#ef5350'],
                                 range_color=[0, 1], text='incidents',
                                 labels={'incidents': 'Total Incidents', 'city': 'City'})
                    fig.update_traces(textposition='outside',
                                      hovertemplate='<b>%{y}</b><br>Incidents: %{x:,}<extra></extra>',
                                      textfont=dict(size=11, family='Inter', color='#1a1a2e'))
                    fig.update_layout(height=360, paper_bgcolor='white', plot_bgcolor='#fafafa',
                                      coloraxis_colorbar=dict(title='Avg Risk', tickformat='.0%',
                                                              tickfont=dict(size=10, family='Inter')),
                                      xaxis=dict(tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                                      yaxis=dict(tickfont=dict(size=12, family='Inter', color='#37474f')))
                    st.plotly_chart(fig, use_container_width=True)

                if 'risk_category' in ds.columns:
                    st.markdown("<div class='section-header'>City vs Risk Category Heatmap (% share)</div>", unsafe_allow_html=True)
                    pivot = ds.groupby(['city', 'risk_category']).size().unstack(fill_value=0)
                    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
                    fig = px.imshow(pivot_pct.round(1),
                                    color_continuous_scale='Blues',
                                    text_auto='.1f',
                                    labels=dict(x='Risk Category', y='City', color='% Share'),
                                    aspect='auto')
                    fig.update_traces(textfont=dict(size=11, family='Inter', color='#1a1a2e'))
                    fig.update_layout(height=380, paper_bgcolor='white',
                                      xaxis=dict(tickfont=dict(size=11, family='Inter', color='#37474f')),
                                      yaxis=dict(tickfont=dict(size=11, family='Inter', color='#37474f')),
                                      coloraxis_colorbar=dict(tickfont=dict(size=10, family='Inter')))
                    st.plotly_chart(fig, use_container_width=True)

        # ── Tab 3: Time Patterns ──
        with atab3:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("<div class='section-header'>Hourly Risk Pattern</div>", unsafe_allow_html=True)
                if 'hour' in ds.columns and 'risk_score' in ds.columns:
                    hourly = ds.groupby('hour')['risk_score'].agg(['mean', 'count']).reset_index()
                    hourly.columns = ['hour', 'avg_risk', 'count']
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hourly['hour'], y=hourly['avg_risk'],
                        mode='lines+markers',
                        line=dict(color='#3949ab', width=3),
                        marker=dict(size=8, color=hourly['avg_risk'],
                                    colorscale=['#66bb6a', '#ffa726', '#ef5350'],
                                    cmin=0, cmax=1, line=dict(width=1.5, color='white')),
                        fill='tozeroy', fillcolor='rgba(57,73,171,0.08)',
                        hovertemplate='Hour %{x}:00<br>Avg Risk: %{y:.1%}<extra></extra>'
                    ))
                    fig.add_hline(y=0.6, line_dash='dash', line_color='#ef5350',
                                  annotation_text='High Risk', annotation_font_size=11)
                    fig.add_hline(y=0.4, line_dash='dash', line_color='#ffa726',
                                  annotation_text='Moderate', annotation_font_size=11)
                    fig.update_layout(height=320, paper_bgcolor='white', plot_bgcolor='#fafafa',
                                      xaxis=dict(title='Hour of Day', tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                                      yaxis=dict(title='Avg Risk Score', tickformat='.0%', tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                                      margin=dict(t=10, b=40, l=50, r=20), hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)

            with col_b:
                st.markdown("<div class='section-header'>Weekday vs Weekend Risk</div>", unsafe_allow_html=True)
                if 'is_weekend' in ds.columns and 'risk_score' in ds.columns:
                    wk = ds.groupby('is_weekend')['risk_score'].agg(['mean', 'count']).reset_index()
                    wk['label'] = wk['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
                    fig = px.bar(wk, x='label', y='mean',
                                 color='mean',
                                 color_continuous_scale=['#66bb6a', '#ffa726', '#ef5350'],
                                 range_color=[0, 1],
                                 text=wk['mean'].apply(lambda v: f'{v:.1%}'),
                                 labels={'mean': 'Avg Risk Score', 'label': ''})
                    fig.update_traces(textposition='outside',
                                      textfont=dict(size=13, family='Inter', color='#1a1a2e'))
                    fig.update_layout(height=320, paper_bgcolor='white', plot_bgcolor='#fafafa',
                                      coloraxis_showscale=False,
                                      xaxis=dict(tickfont=dict(size=13, family='Inter', color='#37474f')),
                                      yaxis=dict(tickformat='.0%', tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'))
                    st.plotly_chart(fig, use_container_width=True)

            if 'is_peak_hour' in ds.columns and 'risk_score' in ds.columns:
                st.markdown("<div class='section-header'>Peak Hour vs Off-Peak Risk Distribution</div>", unsafe_allow_html=True)
                ds_peak = ds.copy()
                ds_peak['period'] = ds_peak['is_peak_hour'].map({0: 'Off-Peak', 1: 'Peak Hour'})
                fig = px.violin(ds_peak, x='period', y='risk_score', color='period',
                                color_discrete_map={'Peak Hour': '#ef5350', 'Off-Peak': '#42a5f5'},
                                box=True, points='outliers',
                                labels={'period': '', 'risk_score': 'Risk Score'})
                fig.update_layout(height=320, paper_bgcolor='white', plot_bgcolor='#fafafa',
                                  showlegend=False,
                                  xaxis=dict(tickfont=dict(size=13, family='Inter', color='#37474f')),
                                  yaxis=dict(tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'))
                st.plotly_chart(fig, use_container_width=True)

        # ── Tab 4: Weather & Road ──
        with atab4:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("<div class='section-header'>Risk Score by Weather</div>", unsafe_allow_html=True)
                if 'weather' in ds.columns and 'risk_score' in ds.columns:
                    fig = px.box(ds, x='weather', y='risk_score', color='weather',
                                 color_discrete_sequence=px.colors.qualitative.Set2,
                                 labels={'weather': 'Weather', 'risk_score': 'Risk Score'})
                    fig.update_layout(height=340, paper_bgcolor='white', plot_bgcolor='#fafafa',
                                      showlegend=False,
                                      xaxis=dict(tickfont=dict(size=11, family='Inter', color='#37474f')),
                                      yaxis=dict(tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'))
                    st.plotly_chart(fig, use_container_width=True)

            with col_b:
                st.markdown("<div class='section-header'>Road Condition vs Road Type</div>", unsafe_allow_html=True)
                if 'road_condition' in ds.columns and 'road_type' in ds.columns:
                    fig = px.histogram(ds, x='road_condition', color='road_type',
                                       barmode='group',
                                       color_discrete_sequence=px.colors.qualitative.Set3,
                                       labels={'road_condition': 'Road Condition', 'count': 'Records'})
                    fig.update_layout(height=340, paper_bgcolor='white', plot_bgcolor='#fafafa',
                                      hovermode='x unified',
                                      xaxis=dict(tickfont=dict(size=11, family='Inter', color='#37474f')),
                                      yaxis=dict(tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                                      legend=dict(font=dict(size=11, family='Inter', color='#37474f')))
                    st.plotly_chart(fig, use_container_width=True)

            if 'festival' in ds.columns and 'risk_score' in ds.columns:
                st.markdown("<div class='section-header'>Festival Impact on Risk Score</div>", unsafe_allow_html=True)
                fest_stats = ds.groupby('festival')['risk_score'].agg(['mean', 'count']).reset_index()
                fest_stats.columns = ['festival', 'avg_risk', 'count']
                fest_stats['festival'] = fest_stats['festival'].str.replace('_', ' ').str.title()
                fest_stats = fest_stats.sort_values('avg_risk', ascending=False)
                fig = px.bar(fest_stats, x='festival', y='avg_risk',
                             color='avg_risk',
                             color_continuous_scale=['#66bb6a', '#ffa726', '#ef5350'],
                             range_color=[0, 1],
                             text=fest_stats['avg_risk'].apply(lambda v: f'{v:.0%}'),
                             custom_data=['count'],
                             labels={'festival': 'Festival', 'avg_risk': 'Avg Risk Score'})
                fig.update_traces(textposition='outside',
                                  hovertemplate='<b>%{x}</b><br>Avg Risk: %{y:.1%}<br>Records: %{customdata[0]:,}<extra></extra>',
                                  textfont=dict(size=11, family='Inter', color='#1a1a2e'))
                fig.update_layout(height=320, paper_bgcolor='white', plot_bgcolor='#fafafa',
                                  coloraxis_showscale=False,
                                  xaxis=dict(tickfont=dict(size=11, family='Inter', color='#37474f')),
                                  yaxis=dict(tickformat='.0%', tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'))
                st.plotly_chart(fig, use_container_width=True)

            if 'temperature' in ds.columns and 'risk_score' in ds.columns:
                st.markdown("<div class='section-header'>Temperature vs Risk Score</div>", unsafe_allow_html=True)
                sample = ds.sample(min(2000, len(ds)), random_state=42)
                fig = px.scatter(sample, x='temperature', y='risk_score',
                                 color='weather' if 'weather' in sample.columns else None,
                                 opacity=0.5, trendline='lowess',
                                 labels={'temperature': 'Temperature (C)', 'risk_score': 'Risk Score'},
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=320, paper_bgcolor='white', plot_bgcolor='#fafafa',
                                  xaxis=dict(tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                                  yaxis=dict(tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                                  legend=dict(font=dict(size=11, family='Inter', color='#37474f')))
                st.plotly_chart(fig, use_container_width=True)

        # ── Tab 5: Raw Data ──
        with atab5:
            st.markdown("<div class='section-header'>Dataset Preview</div>", unsafe_allow_html=True)
            n_rows = st.slider("Rows to display", 10, 200, 50, step=10)
            st.dataframe(ds.head(n_rows), use_container_width=True, height=420)

            st.markdown("<div class='section-header'>Descriptive Statistics</div>", unsafe_allow_html=True)
            num_cols = ds.select_dtypes(include=['number']).columns.tolist()
            if num_cols:
                st.dataframe(ds[num_cols].describe().T.style.background_gradient(
                    subset=['mean', 'std'], cmap='Blues'
                ), use_container_width=True)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                csv_ds = ds.to_csv(index=False).encode('utf-8')
                st.download_button("Download Full Dataset", csv_ds,
                                   "processed_dataset.csv", "text/csv", use_container_width=True)
            with col_dl2:
                stats_csv = ds[num_cols].describe().T.to_csv().encode('utf-8') if num_cols else b""
                st.download_button("Download Statistics", stats_csv,
                                   "dataset_statistics.csv", "text/csv", use_container_width=True)

# ═════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    # Hero banner
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0f0c29 0%, #1a237e 50%, #283593 100%);
                border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem;
                box-shadow: 0 8px 32px rgba(26,35,126,0.35);'>
        <div style='text-align:center;'>
            <div style='font-size:3rem; margin-bottom:1rem;'>🤖</div>
            <div style='font-size:1.8rem; font-weight:800; color:#ffffff; margin-bottom:0.5rem;'>
                Model Performance Comparison
            </div>
            <div style='font-size:1rem; color:#9fa8da; margin-top:0.3rem; font-weight:500;'>
                Regression & Classification Model Metrics
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    metrics = load_metrics()
    
    if metrics:
        perf_tab1, perf_tab2 = st.tabs(["📉 Regression Model", "🎯 Classification Model"])
        
        # ── Regression Performance ───────────────────────────────────────────
        with perf_tab1:
            st.markdown("<div class='section-header'>Regression Model Metrics (Best Model)</div>", unsafe_allow_html=True)
            reg_metrics = metrics.get('regression', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R² Score (Test)", f"{reg_metrics.get('r2_test', 0):.4f}")
            with col2:
                st.metric("RMSE (Test)", f"{reg_metrics.get('rmse_test', 0):.4f}")
            with col3:
                st.metric("MAE (Test)", f"{reg_metrics.get('mae_test', 0):.4f}")
            with col4:
                st.metric("Best Model", reg_metrics.get('best_model', 'N/A'))
            
            # Training vs Test R² comparison
            reg_r2_data = pd.DataFrame({
                'Dataset': ['Training', 'Testing'],
                'R² Score': [reg_metrics.get('r2_train', 0), reg_metrics.get('r2_test', 0)]
            })
            
            col_r2_1, col_r2_2 = st.columns(2)
            with col_r2_1:
                fig_r2 = px.bar(reg_r2_data, x='Dataset', y='R² Score', 
                               color='R² Score', color_continuous_scale='RdYlGn',
                               range_color=[0, 1], title='R² Score: Train vs Test')
                fig_r2.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#fafafa')
                st.plotly_chart(fig_r2, use_container_width=True)
            
            with col_r2_2:
                fig_error = go.Figure(data=[
                    go.Bar(name='RMSE', x=['Train', 'Test'], 
                          y=[reg_metrics.get('rmse_train', 0), reg_metrics.get('rmse_test', 0)],
                          marker_color='#3949ab'),
                    go.Bar(name='MAE', x=['Train', 'Test'], 
                          y=[reg_metrics.get('mae_test', 0), reg_metrics.get('mae_test', 0)],
                          marker_color='#ff7043')
                ])
                fig_error.update_layout(title='Error Metrics', height=350, barmode='group', 
                                       paper_bgcolor='white', plot_bgcolor='#fafafa')
                st.plotly_chart(fig_error, use_container_width=True)
            
            # Algorithm Comparison
            st.markdown("<div class='section-header'>Algorithm Comparison</div>", unsafe_allow_html=True)
            algo_dict = reg_metrics.get('algorithms', {})
            if algo_dict:
                algo_df = pd.DataFrame(algo_dict).T.reset_index().rename(columns={'index': 'Algorithm'})
                
                # Determine the best column to sort by (check multiple possible column names)
                sort_column = None
                for col in ['Test R²', 'test_r2', 'Test R2', 'r2_test', 'r2']:
                    if col in algo_df.columns:
                        sort_column = col
                        break
                
                if sort_column:
                    algo_df = algo_df.sort_values(sort_column, ascending=False)
                
                # Prepare formatting dict with only columns that exist
                format_dict = {}
                for col, fmt in {
                    'Train RMSE': '{:.4f}',
                    'Test RMSE': '{:.4f}',
                    'Train R²': '{:.4f}',
                    'Test R²': '{:.4f}',
                    'Train Adj-R²': '{:.4f}',
                    'Test Adj-R²': '{:.4f}',
                    'train_rmse': '{:.4f}',
                    'test_rmse': '{:.4f}',
                    'train_r2': '{:.4f}',
                    'test_r2': '{:.4f}',
                }.items():
                    if col in algo_df.columns:
                        format_dict[col] = fmt
                
                # Display table with styling
                styled_df = algo_df.style.format(format_dict)
                
                # Apply background gradient if Test R² column exists
                for col in ['Test R²', 'test_r2', 'Test R2']:
                    if col in algo_df.columns:
                        styled_df = styled_df.background_gradient(subset=[col], cmap='RdYlGn')
                        break
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Bar charts - check for required columns
                chart_cols_r2 = None
                chart_cols_rmse = None
                
                for train_col, test_col in [('Train R²', 'Test R²'), ('train_r2', 'test_r2')]:
                    if train_col in algo_df.columns and test_col in algo_df.columns:
                        chart_cols_r2 = [train_col, test_col]
                        break
                
                for train_col, test_col in [('Train RMSE', 'Test RMSE'), ('train_rmse', 'test_rmse')]:
                    if train_col in algo_df.columns and test_col in algo_df.columns:
                        chart_cols_rmse = [train_col, test_col]
                        break
                
                col_algo1, col_algo2 = st.columns(2)
                
                with col_algo1:
                    if chart_cols_r2:
                        fig_r2_algo = px.bar(algo_df, x='Algorithm', y=chart_cols_r2,
                                            barmode='group', title='R² Comparison Across Algorithms',
                                            color_discrete_map={chart_cols_r2[0]: '#3949ab', chart_cols_r2[1]: '#ff7043'})
                        fig_r2_algo.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#fafafa')
                        st.plotly_chart(fig_r2_algo, use_container_width=True)
                    else:
                        st.warning("R² metrics not available in algorithms data")
                
                with col_algo2:
                    if chart_cols_rmse:
                        fig_rmse_algo = px.bar(algo_df, x='Algorithm', y=chart_cols_rmse,
                                              barmode='group', title='RMSE Comparison Across Algorithms',
                                              color_discrete_map={chart_cols_rmse[0]: '#42a5f5', chart_cols_rmse[1]: '#ef9a9a'})
                        fig_rmse_algo.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#fafafa')
                        st.plotly_chart(fig_rmse_algo, use_container_width=True)
                    else:
                        st.warning("RMSE metrics not available in algorithms data")
            
            # Summary metrics
            st.markdown("<div class='section-header'>Best Model Performance Summary</div>", unsafe_allow_html=True)
            summary_data = {
                'Metric': ['R² (Train)', 'R² (Test)', 'RMSE (Train)', 'RMSE (Test)', 'MAE (Test)', 'Train Samples', 'Test Samples'],
                'Value': [
                    f"{reg_metrics.get('r2_train', 0):.4f}",
                    f"{reg_metrics.get('r2_test', 0):.4f}",
                    f"{reg_metrics.get('rmse_train', 0):.4f}",
                    f"{reg_metrics.get('rmse_test', 0):.4f}",
                    f"{reg_metrics.get('mae_test', 0):.4f}",
                    f"{reg_metrics.get('training_samples', 0):,}",
                    f"{reg_metrics.get('test_samples', 0):,}"
                ]
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            
            with st.expander("📋 Metric Interpretation"):
                st.markdown("""
                - **R² Score**: Proportion of variance explained (0-1, higher is better)
                - **RMSE**: Root Mean Squared Error - average prediction error magnitude
                - **MAE**: Mean Absolute Error - average absolute prediction difference
                - **Adj-R²**: Adjusted R² accounting for number of features
                - **Train-Test Gap**: Difference indicates model generalization
                """)
        
        # ── Classification Performance ────────────────────────────────────────
        with perf_tab2:
            st.markdown("<div class='section-header'>Classification Model Metrics (Best Model)</div>", unsafe_allow_html=True)
            cls_metrics = metrics.get('classification', {})
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{cls_metrics.get('accuracy', 0):.4f}")
            with col2:
                st.metric("Precision (Weighted)", f"{cls_metrics.get('precision_weighted', 0):.4f}")
            with col3:
                st.metric("Recall (Weighted)", f"{cls_metrics.get('recall_weighted', 0):.4f}")
            with col4:
                st.metric("Best Model", cls_metrics.get('best_model', 'N/A'))
            
            # Classification metrics comparison
            cls_perf_data = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'Score': [
                    cls_metrics.get('accuracy', 0),
                    cls_metrics.get('precision_weighted', 0),
                    cls_metrics.get('recall_weighted', 0),
                    cls_metrics.get('f1_weighted', 0)
                ]
            })
            
            col_cls_1, col_cls_2 = st.columns(2)
            with col_cls_1:
                fig_cls = px.bar(cls_perf_data, x='Metric', y='Score',
                                color='Score', color_continuous_scale='Blues',
                                range_color=[0, 1], title='Best Model Metrics')
                fig_cls.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#fafafa')
                st.plotly_chart(fig_cls, use_container_width=True)
            
            with col_cls_2:
                # Radar chart for multi-metric comparison
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=[
                        cls_metrics.get('accuracy', 0),
                        cls_metrics.get('precision_weighted', 0),
                        cls_metrics.get('recall_weighted', 0),
                        cls_metrics.get('f1_weighted', 0)
                    ],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    fill='toself',
                    marker_color='rgba(57, 73, 171, 0.5)'
                ))
                fig_radar.update_layout(title='Multi-Metric Performance Profile', height=350)
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Algorithm Comparison
            st.markdown("<div class='section-header'>Algorithm Comparison</div>", unsafe_allow_html=True)
            algo_dict = cls_metrics.get('algorithms', {})
            if algo_dict:
                algo_df = pd.DataFrame(algo_dict).T.reset_index().rename(columns={'index': 'Algorithm'})
                
                # Determine the best column to sort by (check multiple possible column names)
                sort_column = None
                for col in ['Accuracy', 'accuracy', 'Acc', 'acc']:
                    if col in algo_df.columns:
                        sort_column = col
                        break
                
                if sort_column:
                    algo_df = algo_df.sort_values(sort_column, ascending=False)
                
                # Prepare formatting dict with only columns that exist
                format_dict = {}
                for col, fmt in {
                    'Accuracy': '{:.4f}',
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1': '{:.4f}',
                    'accuracy': '{:.4f}',
                    'precision': '{:.4f}',
                    'recall': '{:.4f}',
                    'f1': '{:.4f}',
                }.items():
                    if col in algo_df.columns:
                        format_dict[col] = fmt
                
                # Display table with styling
                styled_df = algo_df.style.format(format_dict)
                
                # Apply background gradient if Accuracy column exists
                for col in ['Accuracy', 'accuracy', 'Acc']:
                    if col in algo_df.columns:
                        styled_df = styled_df.background_gradient(subset=[col], cmap='RdYlGn')
                        break
                
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Bar charts - check for required columns
                acc_col = None
                for col in ['Accuracy', 'accuracy', 'Acc']:
                    if col in algo_df.columns:
                        acc_col = col
                        break
                
                perf_cols = []
                for col in ['Precision', 'Recall', 'F1']:
                    if col in algo_df.columns:
                        perf_cols.append(col)
                
                col_algo1, col_algo2 = st.columns(2)
                with col_algo1:
                    if acc_col:
                        fig_acc_algo = px.bar(algo_df, x='Algorithm', y=[acc_col],
                                             title='Accuracy Comparison Across Algorithms',
                                             color=acc_col, color_continuous_scale='Blues')
                        fig_acc_algo.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#fafafa')
                        st.plotly_chart(fig_acc_algo, use_container_width=True)
                    else:
                        st.warning("Accuracy metrics not available in algorithms data")
                
                with col_algo2:
                    if perf_cols:
                        fig_f1_algo = px.bar(algo_df, x='Algorithm', y=perf_cols,
                                            barmode='group', title='Precision, Recall & F1 Comparison',
                                            color_discrete_map={perf_cols[i]: c for i, c in enumerate(['#42a5f5', '#66bb6a', '#ff7043'])})
                        fig_f1_algo.update_layout(height=350, paper_bgcolor='white', plot_bgcolor='#fafafa')
                        st.plotly_chart(fig_f1_algo, use_container_width=True)
                    else:
                        st.warning("Classification metrics (Precision, Recall, F1) not available in algorithms data")
            
            # Class information
            st.markdown("<div class='section-header'>Classification Classes</div>", unsafe_allow_html=True)
            classes = cls_metrics.get('classes', [])
            num_classes = cls_metrics.get('num_classes', 0)
            
            class_info = {
                'Class': classes,
                'Category ID': list(range(num_classes))
            }
            st.dataframe(pd.DataFrame(class_info), use_container_width=True, hide_index=True)
            
            # Summary metrics
            st.markdown("<div class='section-header'>Best Model Performance Summary</div>", unsafe_allow_html=True)
            summary_data = {
                'Metric': ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)', 'Number of Classes', 'Train Samples', 'Test Samples'],
                'Value': [
                    f"{cls_metrics.get('accuracy', 0):.4f}",
                    f"{cls_metrics.get('precision_weighted', 0):.4f}",
                    f"{cls_metrics.get('recall_weighted', 0):.4f}",
                    f"{cls_metrics.get('f1_weighted', 0):.4f}",
                    f"{cls_metrics.get('num_classes', 0)}",
                    f"{cls_metrics.get('training_samples', 0):,}",
                    f"{cls_metrics.get('test_samples', 0):,}"
                ]
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            
            with st.expander("📋 Metric Interpretation"):
                st.markdown("""
                - **Accuracy**: Percentage of correct predictions across all classes
                - **Precision**: True positives / (True positives + False positives)
                - **Recall**: True positives / (True positives + False negatives)
                - **F1-Score**: Harmonic mean of precision and recall
                - **Weighted**: Metrics averaged weighted by class support
                """)
    else:
        st.warning("⚠️ metrics.json not found. Please ensure models are properly trained.")

# ═════════════════════════════════════════════════════════════
# PAGE 6 — ABOUT & MODEL
# ═════════════════════════════════════════════════════════════
elif page == "ℹ️ About & Model":
    # Hero banner
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0f0c29 0%, #1a237e 50%, #283593 100%);
                border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem;
                box-shadow: 0 8px 32px rgba(26,35,126,0.35);'>
        <div style='text-align:center;'>
            <div style='font-size:3rem; margin-bottom:1rem;'>🚦</div>
            <div style='font-size:1.8rem; font-weight:800; color:#ffffff; margin-bottom:0.5rem;'>
                About This Platform
            </div>
            <div style='font-size:1rem; color:#9fa8da; margin-top:0.3rem; font-weight:500;'>
                AI-Powered Road Safety Intelligence for India
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── About Section ──
    st.markdown("<h2 style='color:#1a237e; margin-top:2rem;'>📋 Platform Overview</h2>", unsafe_allow_html=True)
    st.markdown("""
    **Road Risk Intelligence** is an advanced machine learning platform designed to predict accident risk on Indian roads 
    in real-time. It combines data science, deep learning, and road safety expertise to provide actionable insights 
    for drivers, fleet managers, and transportation authorities.

    ### Key Features:
    - **Real-Time Risk Assessment**: Predict accident risk for any location and time
    - **Batch Processing**: Analyze thousands of trips simultaneously
    - **Rich Data Insights**: Explore patterns across 20,000 historical Indian road records
    - **Multi-Factor Analysis**: Weather, traffic, road conditions, time, location, and driving behavior
    - **Safety Recommendations**: Get personalized safety tips based on risk factors
    """)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Model Section ──
    st.markdown("<h2 style='color:#1a237e; margin-top:2rem;'>🤖 Machine Learning Pipeline</h2>", unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("""
        <div class='result-card' style='border-left:4px solid #3949ab;'>
            <div style='font-size:1.3rem; font-weight:700; color:#1a237e; margin-bottom:0.8rem;'>
                📊 Stage 1: Risk Scoring (Regression)
            </div>
            <ul style='margin:0; padding-left:1.3rem; color:#37474f; font-size:0.92rem; line-height:1.8;'>
                <li><b>Algorithm:</b> XGBoost Regressor</li>
                <li><b>Task:</b> Predict continuous risk score (0-1)</li>
                <li><b>Features:</b> 50+ engineered features</li>
                <li><b>Output:</b> Numerical risk score</li>
                <li><b>Performance:</b> R² Score optimized for precision</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class='result-card' style='border-left:4px solid #ab47bc;'>
            <div style='font-size:1.3rem; font-weight:700; color:#1a237e; margin-bottom:0.8rem;'>
                🎯 Stage 2: Risk Categorization (Classification)
            </div>
            <ul style='margin:0; padding-left:1.3rem; color:#37474f; font-size:0.92rem; line-height:1.8;'>
                <li><b>Algorithm:</b> LightGBM Classifier</li>
                <li><b>Task:</b> Classify root cause of risk</li>
                <li><b>Classes:</b> 4 risk categories</li>
                <li><b>Output:</b> Category probabilities</li>
                <li><b>Performance:</b> F1-Score optimized</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Feature Engineering ──
    st.markdown("<h3 style='color:#283593; margin-top:1.5rem;'>🔧 Feature Engineering</h3>", unsafe_allow_html=True)

    tab_feat1, tab_feat2, tab_feat3, tab_feat4 = st.tabs([
        "🗺️ Spatial Features", "⏰ Temporal Features", "🌦️ Environmental", "🚗 Road/Traffic"
    ])

    with tab_feat1:
        st.markdown("""
        - **Latitude & Longitude**: GPS coordinates of the incident location
        - **Distance-based Features**: Proximity calculations to city centers
        - **Regional Patterns**: Risk patterns specific to different geographic areas
        - **Elevation & Terrain**: Derived from coordinates for complex terrain areas
        """)

    with tab_feat2:
        st.markdown("""
        - **Hour of Day**: 24-hour cyclical encoding (sine/cosine transformation)
        - **Temporal Markers**: Is night? Morning rush? Evening rush?
        - **Day Type**: Weekday vs weekend patterns
        - **Peak Hour**: High-traffic periods (7-9 AM, 5-7 PM)
        - **Monthly Seasonality**: Festival impact encoding
        """)

    with tab_feat3:
        st.markdown("""
        - **Weather Risk Score**: Clear → Cloudy → Rain → Fog → Storm (0-4 scale)
        - **Visibility Levels**: High, Medium, Low encoding
        - **Temperature**: Both raw and logarithmic transformations
        - **Interaction Terms**: Weather × Visibility, Night × Fog, etc.
        - **Environmental Composites**: Risk interaction features
        """)

    with tab_feat4:
        st.markdown("""
        - **Road Type**: Highway, Urban, Rural, Expressway, Mountain
        - **Road Condition**: Good, Under Construction, Damaged (one-hot)
        - **Traffic Density**: Low, Medium, High encoding
        - **Traffic Signals**: Presence indicator
        - **Number of Lanes**: Raw lane count
        - **Complex Interactions**: Peak traffic, construction in rain, etc.
        """)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Risk Categories ──
    st.markdown("<h2 style='color:#1a237e; margin-top:2rem;'>🎯 Risk Categories</h2>", unsafe_allow_html=True)

    cat_cols = st.columns(4)
    categories_info = {
        'Weather-Related': ('🌧️', 'Heavy rain, storm, hail, or snow affecting visibility and traction'),
        'Visibility-Related': ('🌫️', 'Fog, glare, smoke reducing sight distance and reaction time'),
        'Road Infrastructure': ('🛣️', 'Potholes, damaged roads, construction, poor signage'),
        'Driving Behavior': ('🚗', 'Overspeeding, distraction, impairment, reckless driving'),
    }
    colors_cat = {'Weather-Related': '#42a5f5', 'Visibility-Related': '#ab47bc',
                  'Road Infrastructure': '#ff7043', 'Driving Behavior': '#26a69a'}

    for col, (cat_name, (emoji, desc)) in zip(cat_cols, categories_info.items()):
        with col:
            st.markdown(f"""
            <div class='result-card' style='border-left:4px solid {colors_cat[cat_name]}; text-align:center;'>
                <div style='font-size:2.5rem; margin-bottom:0.5rem;'>{emoji}</div>
                <div style='font-weight:700; color:#1a237e; margin-bottom:0.3rem;'>{cat_name}</div>
                <div style='font-size:0.85rem; color:#37474f; line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("<h2 style='color:#1a237e; margin-top:2rem;'>📈 Model Performance Metrics</h2>", unsafe_allow_html=True)

    metrics = load_metrics()
    
    if metrics:
        reg_metrics = metrics.get('regression', {})
        cls_metrics = metrics.get('classification', {})
        
        perf_col1, perf_col2 = st.columns(2)

        with perf_col1:
            st.markdown(f"""
            <div class='result-card' style='border-left:4px solid #3949ab;'>
                <div style='font-size:1.2rem; font-weight:700; color:#1a237e; margin-bottom:1rem;'>
                    Regression Model (Risk Score)
                </div>
                <div style='background:#f5f5f5; padding:1rem; border-radius:8px; margin-bottom:0.8rem;'>
                    <div style='font-size:0.9rem; color:#37474f; margin-bottom:0.4rem;'><b>Test R² Score:</b> {reg_metrics.get('r2_test', 0):.3f}</div>
                    <div style='font-size:0.9rem; color:#37474f; margin-bottom:0.4rem;'><b>Train R² Score:</b> {reg_metrics.get('r2_train', 0):.3f}</div>
                    <div style='font-size:0.9rem; color:#37474f; margin-bottom:0.4rem;'><b>Test MAE:</b> {reg_metrics.get('mae_test', 0):.4f}</div>
                    <div style='font-size:0.9rem; color:#37474f; margin-bottom:0.4rem;'><b>Test RMSE:</b> {reg_metrics.get('rmse_test', 0):.4f}</div>
                    <div style='font-size:0.9rem; color:#37474f;'><b>Train-Test Gap:</b> {reg_metrics.get('train_test_r2_gap', 0):.4f}</div>
                </div>
                <div style='font-size:0.85rem; color:#5c6bc0;'>✓ Training Samples: {reg_metrics.get('training_samples', 0):,}</div>
            </div>
            """, unsafe_allow_html=True)

        with perf_col2:
            st.markdown(f"""
            <div class='result-card' style='border-left:4px solid #ab47bc;'>
                <div style='font-size:1.2rem; font-weight:700; color:#1a237e; margin-bottom:1rem;'>
                    Classification Model (Risk Category)
                </div>
                <div style='background:#f5f5f5; padding:1rem; border-radius:8px; margin-bottom:0.8rem;'>
                    <div style='font-size:0.9rem; color:#37474f; margin-bottom:0.4rem;'><b>Accuracy:</b> {cls_metrics.get('accuracy', 0):.3f}</div>
                    <div style='font-size:0.9rem; color:#37474f; margin-bottom:0.4rem;'><b>Weighted F1:</b> {cls_metrics.get('f1_weighted', 0):.3f}</div>
                    <div style='font-size:0.9rem; color:#37474f; margin-bottom:0.4rem;'><b>Precision:</b> {cls_metrics.get('precision_weighted', 0):.3f}</div>
                    <div style='font-size:0.9rem; color:#37474f; margin-bottom:0.4rem;'><b>Recall:</b> {cls_metrics.get('recall_weighted', 0):.3f}</div>
                    <div style='font-size:0.9rem; color:#37474f;'><b>Classes:</b> {cls_metrics.get('num_classes', 0)}</div>
                </div>
                <div style='font-size:0.85rem; color:#5c6bc0;'>✓ Training Samples: {cls_metrics.get('training_samples', 0):,}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📌 Run `retrain_models.py` first to generate and save model metrics.")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Data & Technology ──
    st.markdown("<h2 style='color:#1a237e; margin-top:2rem;'>🔧 Technology Stack</h2>", unsafe_allow_html=True)

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:
        st.markdown("""
        <div class='result-card' style='border-left:4px solid #00695c;'>
            <div style='font-weight:700; color:#1a237e; margin-bottom:0.8rem;'>📚 ML Frameworks</div>
            <ul style='margin:0; padding-left:1.3rem; color:#37474f; font-size:0.92rem; line-height:2;'>
                <li>XGBoost 2.0</li>
                <li>LightGBM</li>
                <li>scikit-learn</li>
                <li>pandas, NumPy</li>
                <li>Plotly, Streamlit</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tech_col2:
        st.markdown("""
        <div class='result-card' style='border-left:4px solid #e65100;'>
            <div style='font-weight:700; color:#1a237e; margin-bottom:0.8rem;'>📊 Data Processing</div>
            <ul style='margin:0; padding-left:1.3rem; color:#37474f; font-size:0.92rem; line-height:2;'>
                <li>Min-Max Scaling</li>
                <li>Label Encoding</li>
                <li>One-Hot Encoding</li>
                <li>Cyclical Transform</li>
                <li>Log Transformation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tech_col3:
        st.markdown("""
        <div class='result-card' style='border-left:4px solid #6a1b9a;'>
            <div style='font-weight:700; color:#1a237e; margin-bottom:0.8rem;'>🎯 Deployment</div>
            <ul style='margin:0; padding-left:1.3rem; color:#37474f; font-size:0.92rem; line-height:2;'>
                <li>Real-time Inference</li>
                <li>Batch Processing</li>
                <li>API-Ready Models</li>
                <li>Joblib Serialization</li>
                <li>Production Optimized</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── How to Use ──
    st.markdown("<h2 style='color:#1a237e; margin-top:2rem;'>🚀 How to Use This Platform</h2>", unsafe_allow_html=True)

    use_col1, use_col2 = st.columns(2)

    with use_col1:
        st.markdown("""
        <div class='section-header'>For Individual Travelers</div>

        1. **Dashboard** → Get overview of current road accident patterns
        2. **Risk Assessment** → Enter your trip details (location, time, weather, road)
        3. **Get Prediction** → See accident risk score and safety recommendations
        4. **Follow Tips** → Adjust driving behavior based on suggestions

        **Example Use Case:**
        - Planning a late-night drive in rainy conditions?
        - Platform predicts HIGH RISK and suggests alternatives
        - Follow recommendations to stay safe
        """)

    with use_col2:
        st.markdown("""
        <div class='section-header'>For Fleet Managers</div>

        1. **Batch Prediction** → Upload fleet's daily trip data (CSV)
        2. **Mass Analysis** → Get risk scores for all routes
        3. **Route Optimization** → Identify high-risk routes and times
        4. **Safety Policies** → Set policies based on risk categories
        5. **Driver Briefings** → Warn drivers of risky conditions

        **Example Use Case:**
        - Managing 100 delivery routes daily?
        - Batch analyze all routes for risk
        - Alert drivers on risky weather or road conditions
        """)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Disclaimer & Contact ──
    st.markdown("""
    <div style='background:#fff8e1; border-radius:10px; padding:1.2rem 1.5rem; border-left:4px solid #f57f17; margin-top:2rem;'>
        <div style='font-weight:700; color:#e65100; margin-bottom:0.5rem;'>⚠️ Important Disclaimer</div>
        <div style='font-size:0.9rem; color:#37474f; line-height:1.6;'>
            This platform provides <b>predictive insights only</b> and should not be considered as absolute risk assessment. 
            Always follow traffic rules, maintain vigilance, and make informed driving decisions. The platform's predictions 
            are based on historical patterns and may not account for unforeseen circumstances. Drivers remain responsible 
            for their safety and the safety of others on the road.
        </div>
    </div>

    <div style='text-align:center; margin-top:2rem; font-size:0.85rem; color:#5c6bc0; font-weight:500;'>
        <b>India Road Accident Risk Intelligence Platform v2.0</b><br>
        Empowering safer roads through AI & Data Science<br>
        <span style='font-size:0.75rem; color:#78909c;'>Built with XGBoost, LightGBM & Streamlit</span>
    </div>
    """, unsafe_allow_html=True)
