import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os

st.set_page_config(page_title="Smart Traffic Risk Predictor", page_icon="🚦", layout="wide")

# ── Load artifacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    reg      = joblib.load('models/regression_model.pkl')
    cls      = joblib.load('models/classification_model.pkl')
    sc_r     = joblib.load('models/scaler_reg.pkl')
    sc_c     = joblib.load('models/scaler_cls.pkl')
    le       = joblib.load('models/label_encoder.pkl')
    feats    = joblib.load('models/features.pkl')
    cls_feats= joblib.load('models/cls_features.pkl') if os.path.exists('models/cls_features.pkl') else feats
    return reg, cls, sc_r, sc_c, le, feats, cls_feats

models_ready = os.path.exists('models/regression_model.pkl')

# ── Helpers ─────────────────────────────────────────────────────────────────
WEATHER_RISK  = {'clear': 0, 'cloudy': 1, 'rain': 2, 'fog': 3, 'hail': 3, 'storm': 4, 'snow': 4}
VISIBILITY_MAP = {'high': 0, 'medium': 1, 'low': 2}
TRAFFIC_MAP    = {'low': 0, 'medium': 1, 'high': 2}
ROAD_TYPES     = ['highway', 'urban', 'rural', 'expressway', 'mountain']
FESTIVALS      = ['no_festival', 'diwali', 'holi', 'eid', 'christmas', 'navratri']

def build_input_row(lat, lon, hour, is_weekend, is_peak, weather, visibility,
                    traffic_density, temperature, lanes, traffic_signal,
                    road_type, festival, features):
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
    risk_interaction = weather_risk * vis_enc
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
        'traffic_density_enc': traffic_enc,
        'temperature': temperature, 'lanes': lanes,
        'traffic_signal': traffic_signal,
        'risk_interaction': risk_interaction,
        'night_fog': night_fog, 'peak_high_traffic': peak_high,
        'temperature_log': temp_log,
    }

    # road dummies
    for rt in ROAD_TYPES:
        base[f'road_{rt}'] = 1 if road_type == rt else 0
    # festival dummies
    for fv in FESTIVALS:
        base[f'festival_{fv}'] = 1 if festival == fv else 0

    row = {f: base.get(f, 0) for f in features}
    return pd.DataFrame([row])


NON_DRIVING_CATS = {'Visibility-Related', 'Weather-Related', 'Road Infrastructure'}

def build_cls_features(input_df, features):
    """Add classification-specific features matching training pipeline."""
    X = input_df[features].copy()
    weather  = input_df.get('weather',      pd.Series(['clear']  * len(input_df))).values
    vis      = input_df.get('visibility',   pd.Series(['high']   * len(input_df))).values
    road     = input_df.get('road_type',    pd.Series(['urban']  * len(input_df))).values
    temp     = input_df.get('temperature',  pd.Series([25]       * len(input_df))).values
    is_night = X.get('is_night',            pd.Series([0]        * len(input_df))).values
    wr       = X.get('weather_risk',        pd.Series([0]        * len(input_df))).values
    ve       = X.get('visibility_enc',      pd.Series([0]        * len(input_df))).values

    X['is_fog']       = (weather == 'fog').astype(int)
    X['is_rain']      = (weather == 'rain').astype(int)
    X['is_storm']     = np.isin(weather, ['storm','hail','snow']).astype(int)
    X['is_low_vis']   = (vis == 'low').astype(int)
    X['is_highway']   = (road == 'highway').astype(int)
    X['is_rural']     = (road == 'rural').astype(int)
    X['fog_night']    = X['is_fog']  * is_night
    X['rain_highway'] = X['is_rain'] * X['is_highway']
    X['low_vis_night']= X['is_low_vis'] * is_night
    X['weather_x_vis']= wr * ve
    X['temp_risk']    = (temp < 15).astype(int)
    return X.fillna(0)


def predict_pipeline(input_df, reg, cls, sc_r, sc_c, le, features):
    X_reg = input_df[features].copy()
    risk_score = float(reg.predict(sc_r.transform(X_reg))[0])
    risk_score = np.clip(risk_score, 0, 1)

    # Always run classification to identify potential cause
    X_cls = build_cls_features(input_df, features)
    x_sc  = sc_c.transform(X_cls[cls_features] if hasattr(sc_c, 'n_features_in_') and sc_c.n_features_in_ == len(cls_features) else X_cls)
    proba = cls.predict_proba(x_sc)[0]
    class_names = list(le.classes_)
    proba_dict  = dict(zip(class_names, proba))

    # Domain rule: Good infrastructure -> Driving Behavior
    lanes        = input_df.get('lanes', pd.Series([0])).values[0]
    traffic_sig  = input_df.get('traffic_signal', pd.Series([0])).values[0]
    road_type    = input_df.get('road_type', pd.Series(['urban'])).values[0]
    domain_rule_fired = bool(lanes > 3 and traffic_sig == 1 and road_type in ['highway', 'expressway'])

    if domain_rule_fired:
        risk_category = 'Driving Behavior'
    else:
        non_driving_prob = sum(p for c, p in proba_dict.items() if c in NON_DRIVING_CATS)
        if non_driving_prob > 0.5:
            risk_category = max(
                {c: p for c, p in proba_dict.items() if c in NON_DRIVING_CATS},
                key=lambda k: proba_dict[k]
            )
        else:
            risk_category = class_names[np.argmax(proba)]

    return risk_score, risk_category, proba_dict, domain_rule_fired


def risk_color(score):
    if score < 0.4:  return '#2ecc71'
    if score < 0.6:  return '#f39c12'
    return '#e74c3c'

def risk_label(score):
    if score < 0.4:  return '🟢 Low Risk'
    if score < 0.6:  return '🟡 Moderate Risk'
    return '🔴 High Risk'

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🚦 Smart Traffic & Accident Risk Prediction System")
st.caption("Two-stage ML pipeline: Regression → Classification")

if not models_ready:
    st.error("Models not found. Please run `code.ipynb` first to train and save models.")
    st.stop()

reg, cls, sc_r, sc_c, le, features, cls_features = load_models()

tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📂 Batch Prediction", "📊 Dataset Insights"])

# ── Tab 1: Single Prediction ─────────────────────────────────────────────────
with tab1:
    st.subheader("Enter Road & Environment Conditions")
    c1, c2, c3 = st.columns(3)

    with c1:
        lat         = st.number_input("Latitude",  value=28.61, format="%.4f")
        lon         = st.number_input("Longitude", value=77.20, format="%.4f")
        hour        = st.slider("Hour of Day", 0, 23, 8)
        is_weekend  = st.selectbox("Weekend?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        is_peak     = st.selectbox("Peak Hour?", [0, 1], format_func=lambda x: "Yes" if x else "No")

    with c2:
        weather     = st.selectbox("Weather", list(WEATHER_RISK.keys()))
        visibility  = st.selectbox("Visibility", list(VISIBILITY_MAP.keys()))
        traffic_den = st.selectbox("Traffic Density", list(TRAFFIC_MAP.keys()))
        temperature = st.slider("Temperature (°C)", 5, 50, 28)
        traffic_sig = st.selectbox("Traffic Signal Present?", [0, 1], format_func=lambda x: "Yes" if x else "No")

    with c3:
        lanes       = st.slider("Number of Lanes", 1, 8, 3)
        road_type   = st.selectbox("Road Type", ROAD_TYPES)
        festival    = st.selectbox("Festival", FESTIVALS)

    if st.button("🚀 Predict Risk", use_container_width=True):
        input_df = build_input_row(lat, lon, hour, is_weekend, is_peak,
                                   weather, visibility, traffic_den,
                                   temperature, lanes, traffic_sig,
                                   road_type, festival, features)

        score, category, proba_dict, rule_fired = predict_pipeline(input_df, reg, cls, sc_r, sc_c, le, features)
        color = risk_color(score)
        label = risk_label(score)

        st.divider()
        m1, m2 = st.columns(2)
        with m1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(score, 3),
                title={'text': "Risk Score"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 0.4], 'color': '#d5f5e3'},
                        {'range': [0.4, 0.6], 'color': '#fdebd0'},
                        {'range': [0.6, 1.0], 'color': '#fadbd8'},
                    ],
                    'threshold': {'line': {'color': 'black', 'width': 3}, 'value': 0.6}
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        with m2:
            st.markdown(f"### {label}")
            st.metric("Risk Score", f"{score:.4f}")
            if score > 0.6:
                # High risk — show category as primary finding
                cat_labels = {
                    'Weather-Related':    '🌧️ Weather-Related',
                    'Visibility-Related': '🌫️ Visibility-Related',
                    'Road Infrastructure':'🛣️ Road Infrastructure',
                    'Driving Behavior':   '🚗 Driving Behavior',
                }
                display_cat = cat_labels.get(category, category)
                st.success(f"**Risk Category:** {display_cat}")
            else:
                # Low risk — still show potential cause as advisory
                cat_labels = {
                    'Weather-Related':    'weather conditions',
                    'Visibility-Related': 'visibility issues',
                    'Road Infrastructure':'road conditions',
                    'Driving Behavior':   'driving behaviour',
                }
                advisory = cat_labels.get(category, category.lower())
                st.warning(f"⚠️ Low risk, but accidents may occur due to **{advisory}**.")

            if proba_dict:
                if rule_fired:
                    st.info("ℹ️ Domain rule applied: lanes > 3 + traffic signal + highway/expressway → **Driving Behavior** override. Chart shows raw model probabilities.")
                st.markdown("**Category Probabilities:**")
                prob_df = pd.DataFrame(list(proba_dict.items()), columns=['Category', 'Probability'])
                prob_df = prob_df.sort_values('Probability', ascending=True)
                fig2 = px.bar(prob_df, x='Probability', y='Category', orientation='h',
                              color='Probability', color_continuous_scale='RdYlGn_r',
                              range_x=[0, 1])
                fig2.update_layout(height=250, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

# ── Tab 2: Batch Prediction ──────────────────────────────────────────────────
with tab2:
    st.subheader("Upload CSV for Batch Prediction")
    st.caption("CSV must contain the same pre-accident columns as the training data.")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded:
        batch_df = pd.read_csv(uploaded)
        batch_df.columns = batch_df.columns.str.strip().str.lower().str.replace(' ', '_')

        # Feature engineering on batch
        batch_df['weather'] = batch_df['weather'].str.strip().str.lower()
        batch_df['visibility'] = batch_df['visibility'].str.strip().str.lower()
        batch_df['traffic_density'] = batch_df['traffic_density'].str.strip().str.lower()
        batch_df['road_type'] = batch_df['road_type'].str.strip().str.lower()
        batch_df['festival'] = batch_df['festival'].str.strip().str.lower().replace('none', 'no_festival')

        batch_df['is_night']        = batch_df['hour'].apply(lambda x: 1 if (x >= 20 or x <= 5) else 0)
        batch_df['is_morning_rush'] = batch_df['hour'].apply(lambda x: 1 if 7 <= x <= 9 else 0)
        batch_df['is_evening_rush'] = batch_df['hour'].apply(lambda x: 1 if 17 <= x <= 19 else 0)
        batch_df['hour_sin']        = np.sin(2 * np.pi * batch_df['hour'] / 24)
        batch_df['hour_cos']        = np.cos(2 * np.pi * batch_df['hour'] / 24)
        batch_df['month']           = pd.to_datetime(batch_df['date'], errors='coerce').dt.month.fillna(6).astype(int)
        batch_df['day_of_year']     = pd.to_datetime(batch_df['date'], errors='coerce').dt.dayofyear.fillna(180).astype(int)
        batch_df['month_sin']       = np.sin(2 * np.pi * batch_df['month'] / 12)
        batch_df['month_cos']       = np.cos(2 * np.pi * batch_df['month'] / 12)
        batch_df['weather_risk']    = batch_df['weather'].map(WEATHER_RISK).fillna(1)
        batch_df['visibility_enc']  = batch_df['visibility'].map(VISIBILITY_MAP).fillna(1)
        batch_df['traffic_density_enc'] = batch_df['traffic_density'].map(TRAFFIC_MAP).fillna(1)
        batch_df['risk_interaction'] = batch_df['weather_risk'] * batch_df['visibility_enc']
        batch_df['night_fog']       = batch_df['is_night'] * (batch_df['weather'] == 'fog').astype(int)
        batch_df['peak_high_traffic'] = batch_df['is_peak_hour'] * batch_df['traffic_density_enc']
        batch_df['temperature_log'] = np.log1p(batch_df['temperature'].clip(lower=0))

        road_dummies = pd.get_dummies(batch_df['road_type'], prefix='road')
        festival_dummies = pd.get_dummies(batch_df['festival'], prefix='festival')
        batch_df = pd.concat([batch_df, road_dummies, festival_dummies], axis=1)

        # Align columns
        for f in features:
            if f not in batch_df.columns:
                batch_df[f] = 0

        X_batch = batch_df[features].fillna(0)
        scores  = reg.predict(sc_r.transform(X_batch)).clip(0, 1)
        batch_df['predicted_risk_score'] = scores

        categories = []
        for i, score in enumerate(scores):
            row = batch_df.iloc[i]
            # Domain rule check
            b_lanes = row.get('lanes', 0)
            b_sig   = row.get('traffic_signal', 0)
            b_road  = row.get('road_type', 'urban')
            if b_lanes > 3 and b_sig == 1 and b_road in ['highway', 'expressway']:
                cat = 'Driving Behavior'
            else:
                x_sc = sc_c.transform(build_cls_features(batch_df.iloc[[i]], features))
                proba = cls.predict_proba(x_sc)[0]
                class_names = list(le.classes_)
                proba_dict  = dict(zip(class_names, proba))
                non_driving_prob = sum(p for c, p in proba_dict.items() if c in NON_DRIVING_CATS)
                if non_driving_prob > 0.5:
                    cat = max(
                        {c: p for c, p in proba_dict.items() if c in NON_DRIVING_CATS},
                        key=lambda k: proba_dict[k]
                    )
                else:
                    cat = class_names[np.argmax(proba)]
            categories.append(cat)

        batch_df['predicted_risk_category'] = categories

        st.success(f"Processed {len(batch_df)} records.")
        st.dataframe(batch_df[['predicted_risk_score', 'predicted_risk_category']].head(50))

        csv_out = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Results", csv_out, "predictions.csv", "text/csv")

        fig = px.histogram(batch_df, x='predicted_risk_score', nbins=30,
                           title='Predicted Risk Score Distribution',
                           color_discrete_sequence=['steelblue'])
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 3: Dataset Insights ──────────────────────────────────────────────────
with tab3:
    if os.path.exists('processed_dataset.csv'):
        ds = pd.read_csv('processed_dataset.csv')
        st.subheader("Processed Dataset Overview")
        st.write(f"Shape: {ds.shape}")
        st.dataframe(ds.head(20))

        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(ds, x='risk_score', nbins=30, title='Risk Score Distribution',
                               color_discrete_sequence=['steelblue'])
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if 'risk_category' in ds.columns:
                fig = px.pie(ds, names='risk_category', title='Risk Category Distribution')
                st.plotly_chart(fig, use_container_width=True)

        if 'weather' in ds.columns:
            fig = px.box(ds, x='weather', y='risk_score', title='Risk Score by Weather',
                         color='weather')
            st.plotly_chart(fig, use_container_width=True)

        if 'hour' in ds.columns:
            hourly = ds.groupby('hour')['risk_score'].mean().reset_index()
            fig = px.line(hourly, x='hour', y='risk_score', title='Avg Risk Score by Hour',
                          markers=True)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the notebook first to generate processed_dataset.csv")
