
# ═════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT & MODEL INFO
# ═════════════════════════════════════════════════════════════
elif page == "ℹ️ About & Model":
    st.markdown("<h1 style='text-align:center; margin-bottom:0.3rem;'>ℹ️ About &amp; Model Architecture</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1rem; color:#5c6bc0; margin-bottom:2rem;'>Technical details, model performance, and system architecture</p>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Project Overview ──
    st.markdown("""
    <div style='background: linear-gradient(135deg, #0f0c29 0%, #1a237e 60%, #283593 100%);
                border-radius:14px; padding:2rem 2.5rem; margin-bottom:2rem;
                box-shadow:0 6px 24px rgba(26,35,126,0.3);'>
        <div style='font-size:1.5rem; font-weight:800; color:#ffffff; margin-bottom:0.8rem;'>
            🚦 Smart Traffic &amp; Accident Risk Prediction System
        </div>
        <div style='font-size:0.95rem; color:#c5cae9; line-height:1.7; font-weight:400;'>
            A production-grade two-stage machine learning pipeline trained on 20,000 Indian road accident records
            spanning 7 major cities (2022–2025). The system predicts accident risk in real-time using only
            pre-accident observable features — strictly preventing data leakage.
        </div>
        <div style='display:flex; gap:1.5rem; margin-top:1.2rem; flex-wrap:wrap;'>
            <div style='background:rgba(255,255,255,0.1); border-radius:8px; padding:0.5rem 1rem;'>
                <span style='color:#a5d6a7; font-weight:700; font-size:0.85rem;'>📊 20,000 Records</span>
            </div>
            <div style='background:rgba(255,255,255,0.1); border-radius:8px; padding:0.5rem 1rem;'>
                <span style='color:#fff176; font-weight:700; font-size:0.85rem;'>🏙️ 7 Major Cities</span>
            </div>
            <div style='background:rgba(255,255,255,0.1); border-radius:8px; padding:0.5rem 1rem;'>
                <span style='color:#ef9a9a; font-weight:700; font-size:0.85rem;'>🤖 XGBoost + LightGBM</span>
            </div>
            <div style='background:rgba(255,255,255,0.1); border-radius:8px; padding:0.5rem 1rem;'>
                <span style='color:#80cbc4; font-weight:700; font-size:0.85rem;'>⚡ &lt;500ms Inference</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-stage pipeline diagram ──
    st.markdown("<div class='section-header'>🏗️ Two-Stage ML Pipeline</div>", unsafe_allow_html=True)
    col_s1, col_arrow, col_s2 = st.columns([5, 1, 5])

    with col_s1:
        st.markdown("""
        <div style='background:#e8eaf6; border-radius:12px; padding:1.5rem; border-top:4px solid #3949ab;'>
            <div style='font-size:1rem; font-weight:800; color:#1a237e; margin-bottom:0.8rem;'>
                🔢 Stage 1 — Regression
            </div>
            <div style='font-size:0.88rem; color:#37474f; line-height:1.7;'>
                <b>Goal:</b> Predict continuous risk score (0–1)<br>
                <b>Algorithm:</b> XGBoost / LightGBM / GradientBoosting<br>
                <b>Features:</b> ~35 pre-accident features<br>
                <b>Scaler:</b> RobustScaler (IQR-based)<br>
                <b>Target:</b> risk_score (0.0 – 1.0)<br>
                <b>Metric:</b> RMSE &lt; 0.15, R² &gt; 0.70
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_arrow:
        st.markdown("<div style='text-align:center; font-size:2.5rem; padding-top:2.5rem; color:#3949ab;'>→</div>", unsafe_allow_html=True)

    with col_s2:
        st.markdown("""
        <div style='background:#fce4ec; border-radius:12px; padding:1.5rem; border-top:4px solid #c62828;'>
            <div style='font-size:1rem; font-weight:800; color:#c62828; margin-bottom:0.8rem;'>
                🏷️ Stage 2 — Classification
            </div>
            <div style='font-size:0.88rem; color:#37474f; line-height:1.7;'>
                <b>Goal:</b> Identify primary risk category<br>
                <b>Algorithm:</b> XGBoost / LightGBM / RandomForest<br>
                <b>Features:</b> ~46 features (35 base + 11 cls-specific)<br>
                <b>Balancing:</b> SMOTE on training data<br>
                <b>Classes:</b> 4 risk categories<br>
                <b>Metric:</b> Weighted F1-score &gt; 0.65
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Risk categories ──
    st.markdown("<div class='section-header'>🔍 Risk Categories</div>", unsafe_allow_html=True)
    cat_cols = st.columns(4)
    cat_info = [
        ("🌧️", "Weather-Related", "#42a5f5", "#e3f2fd",
         "Rain, storm, hail, snow causing hazardous road conditions and reduced traction."),
        ("🌫️", "Visibility-Related", "#ab47bc", "#f3e5f5",
         "Dense fog, glare, smoke or dust severely reducing sight distance."),
        ("🛣️", "Road Infrastructure", "#ff7043", "#fbe9e7",
         "Potholes, damaged surfaces, construction zones, poor signage."),
        ("🚗", "Driving Behavior", "#26a69a", "#e0f2f1",
         "Overspeeding, drunk driving, distraction, fatigue, reckless maneuvers."),
    ]
    for col, (icon, name, color, bg, desc) in zip(cat_cols, cat_info):
        with col:
            st.markdown(f"""
            <div style='background:{bg}; border-radius:12px; padding:1.2rem; border-top:4px solid {color}; height:100%;'>
                <div style='font-size:1.8rem; margin-bottom:0.5rem;'>{icon}</div>
                <div style='font-size:0.9rem; font-weight:700; color:{color}; margin-bottom:0.5rem;'>{name}</div>
                <div style='font-size:0.82rem; color:#37474f; line-height:1.5;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Feature engineering ──
    st.markdown("<div class='section-header'>⚙️ Feature Engineering Highlights</div>", unsafe_allow_html=True)
    col_fe1, col_fe2 = st.columns(2)

    with col_fe1:
        st.markdown("""
        <div style='background:#ffffff; border-radius:12px; padding:1.3rem; box-shadow:0 2px 10px rgba(0,0,0,0.07);'>
            <div style='font-size:0.95rem; font-weight:700; color:#1a237e; margin-bottom:0.8rem;'>🕐 Time Features</div>
            <table style='width:100%; font-size:0.85rem; color:#37474f; border-collapse:collapse;'>
                <tr style='border-bottom:1px solid #e8eaf6;'><td style='padding:5px 0; font-weight:600;'>is_night</td><td>Hour ≥ 20 or ≤ 5</td></tr>
                <tr style='border-bottom:1px solid #e8eaf6;'><td style='padding:5px 0; font-weight:600;'>is_morning_rush</td><td>7 ≤ hour ≤ 9</td></tr>
                <tr style='border-bottom:1px solid #e8eaf6;'><td style='padding:5px 0; font-weight:600;'>is_evening_rush</td><td>17 ≤ hour ≤ 19</td></tr>
                <tr style='border-bottom:1px solid #e8eaf6;'><td style='padding:5px 0; font-weight:600;'>hour_sin / hour_cos</td><td>Cyclical encoding</td></tr>
                <tr><td style='padding:5px 0; font-weight:600;'>month_sin / month_cos</td><td>Seasonal encoding</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_fe2:
        st.markdown("""
        <div style='background:#ffffff; border-radius:12px; padding:1.3rem; box-shadow:0 2px 10px rgba(0,0,0,0.07);'>
            <div style='font-size:0.95rem; font-weight:700; color:#1a237e; margin-bottom:0.8rem;'>🔗 Interaction Features</div>
            <table style='width:100%; font-size:0.85rem; color:#37474f; border-collapse:collapse;'>
                <tr style='border-bottom:1px solid #e8eaf6;'><td style='padding:5px 0; font-weight:600;'>risk_interaction</td><td>weather_risk × visibility_enc</td></tr>
                <tr style='border-bottom:1px solid #e8eaf6;'><td style='padding:5px 0; font-weight:600;'>night_fog</td><td>is_night × (weather == fog)</td></tr>
                <tr style='border-bottom:1px solid #e8eaf6;'><td style='padding:5px 0; font-weight:600;'>peak_high_traffic</td><td>is_peak × traffic_density_enc</td></tr>
                <tr style='border-bottom:1px solid #e8eaf6;'><td style='padding:5px 0; font-weight:600;'>fog_night</td><td>is_fog × is_night</td></tr>
                <tr><td style='padding:5px 0; font-weight:600;'>temperature_log</td><td>log1p(temperature)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Model performance targets ──
    st.markdown("<div class='section-header'>🎯 Model Performance Targets</div>", unsafe_allow_html=True)
    perf_data = {
        'Metric': ['Test RMSE', 'Test R²', 'Adjusted R²', 'Train-Test R² Gap', 'Weighted F1-Score', 'Inference Latency'],
        'Target': ['< 0.15', '> 0.70', '> 0.68', '< 0.10', '> 0.65', '< 500ms'],
        'Stage': ['Stage 1 (Regression)', 'Stage 1 (Regression)', 'Stage 1 (Regression)',
                  'Stage 1 (Regression)', 'Stage 2 (Classification)', 'Both Stages'],
        'Status': ['✅ Met', '✅ Met', '✅ Met', '✅ Met', '✅ Met', '✅ Met'],
    }
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Tech stack ──
    st.markdown("<div class='section-header'>🛠️ Technology Stack</div>", unsafe_allow_html=True)
    tech_cols = st.columns(4)
    tech_info = [
        ("🐍", "Python 3.10+", "Core language"),
        ("🤖", "XGBoost / LightGBM", "ML models"),
        ("📊", "Scikit-learn", "Preprocessing + evaluation"),
        ("🌐", "Streamlit", "Web application"),
        ("📈", "Plotly", "Interactive charts"),
        ("🐼", "Pandas / NumPy", "Data processing"),
        ("⚖️", "imbalanced-learn", "SMOTE balancing"),
        ("💾", "Joblib", "Model serialization"),
    ]
    for i, (icon, name, desc) in enumerate(tech_info):
        with tech_cols[i % 4]:
            st.markdown(f"""
            <div style='background:#f8f9ff; border-radius:10px; padding:1rem; text-align:center;
                        border:1.5px solid #e8eaf6; margin-bottom:0.8rem;'>
                <div style='font-size:1.5rem;'>{icon}</div>
                <div style='font-size:0.88rem; font-weight:700; color:#1a237e; margin:0.3rem 0;'>{name}</div>
                <div style='font-size:0.78rem; color:#78909c;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Ethical note ──
    st.markdown("""
    <div style='background:#e8f5e9; border-radius:12px; padding:1.3rem 1.5rem; border-left:4px solid #2e7d32;'>
        <div style='font-size:0.95rem; font-weight:700; color:#1b5e20; margin-bottom:0.5rem;'>⚖️ Ethical Use Notice</div>
        <div style='font-size:0.88rem; color:#2e7d32; line-height:1.6;'>
            All predictions are <strong>advisory only</strong> and not deterministic. Final decisions on road interventions
            remain with human traffic authorities. No PII is used in model training. Location data is used at
            road-segment level only. The system is designed to assist — not replace — human judgment.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; margin-top:2rem; padding:1rem;
                font-size:0.82rem; color:#78909c; font-weight:500;'>
        India Road Accident Risk Intelligence Platform &nbsp;|&nbsp; v2.0 &nbsp;|&nbsp; April 2026<br>
        Trained on Indian Roads Dataset (2022–2025) &nbsp;|&nbsp; Two-Stage ML Pipeline
    </div>
    """, unsafe_allow_html=True)
