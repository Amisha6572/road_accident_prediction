
# ═════════════════════════════════════════════════════════════
# PAGE 4 — DATA ANALYTICS
# ═════════════════════════════════════════════════════════════
elif page == "📊 Data Analytics":
    st.markdown("<h1 style='text-align:center; margin-bottom:0.3rem;'>📊 Data Analytics &amp; EDA</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1rem; color:#5c6bc0; margin-bottom:2rem;'>Explore patterns and insights from 20,000 Indian road accident records</p>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    ds = load_dataset()
    if ds is None:
        st.info("📌 Run `retrain_models.py` first to generate the processed dataset.")
    else:
        # ── Top KPIs ──
        total   = len(ds)
        high_r  = int((ds['risk_score'] >= 0.6).sum()) if 'risk_score' in ds.columns else 0
        avg_r   = ds['risk_score'].mean() if 'risk_score' in ds.columns else 0
        cities  = ds['city'].nunique() if 'city' in ds.columns else 0
        fatal   = int((ds['accident_severity'] == 'fatal').sum()) if 'accident_severity' in ds.columns else 0
        features_n = ds.shape[1]

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, value, color in [
            (c1, "Total Records",    f"{total:,}",       "#3949ab"),
            (c2, "Features",         f"{features_n}",    "#00695c"),
            (c3, "High Risk",        f"{high_r:,}",      "#c62828"),
            (c4, "Fatal Accidents",  f"{fatal:,}",       "#6a1b9a"),
            (c5, "Cities",           f"{cities}",        "#e65100"),
        ]:
            with col:
                st.markdown(f"""
                <div class='kpi-card' style='border-top-color:{color};'>
                    <div class='kpi-label'>{label}</div>
                    <div class='kpi-value' style='color:{color};'>{value}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Analysis tabs ──
        atab1, atab2, atab3, atab4, atab5 = st.tabs([
            "📈 Risk Distribution", "🏙️ City Analysis",
            "⏰ Time Patterns", "🌦️ Weather & Road", "📋 Raw Data"
        ])

        # ── Tab A1: Risk Distribution ──
        with atab1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("<div class='section-header'>Risk Score Distribution</div>", unsafe_allow_html=True)
                if 'risk_score' in ds.columns:
                    fig = px.histogram(ds, x='risk_score', nbins=50,
                                       color_discrete_sequence=['#3949ab'],
                                       labels={'risk_score': 'Risk Score', 'count': 'Frequency'})
                    fig.add_vline(x=0.6, line_dash='dash', line_color='#c62828',
                                  annotation_text='High Risk', annotation_font_size=11)
                    fig.add_vline(x=0.4, line_dash='dash', line_color='#e65100',
                                  annotation_text='Moderate', annotation_font_size=11)
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

            # Severity breakdown
            if 'accident_severity' in ds.columns and 'risk_score' in ds.columns:
                st.markdown("<div class='section-header'>Risk Score by Accident Severity</div>", unsafe_allow_html=True)
                sev_order = ['minor', 'major', 'fatal']
                sev_colors = {'minor': '#66bb6a', 'major': '#ffa726', 'fatal': '#ef5350'}
                fig = px.box(ds, x='accident_severity', y='risk_score',
                             color='accident_severity', color_discrete_map=sev_colors,
                             category_orders={'accident_severity': sev_order},
                             labels={'accident_severity': 'Severity', 'risk_score': 'Risk Score'})
                fig.update_layout(height=320, paper_bgcolor='white', plot_bgcolor='#fafafa',
                                  showlegend=False,
                                  xaxis=dict(tickfont=dict(size=12, family='Inter', color='#37474f')),
                                  yaxis=dict(tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'))
                st.plotly_chart(fig, use_container_width=True)

        # ── Tab A2: City Analysis ──
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
                                 range_color=[0, 1],
                                 error_y='std_risk',
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
                                 range_color=[0, 1],
                                 text='incidents',
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

                # City x Category heatmap
                if 'risk_category' in ds.columns:
                    st.markdown("<div class='section-header'>City vs Risk Category Heatmap</div>", unsafe_allow_html=True)
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

        # ── Tab A3: Time Patterns ──
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
                    wk['label'] = wk['is_weekend'].map({0: '📅 Weekday', 1: '🏖️ Weekend'})
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

            # Peak hour analysis
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

        # ── Tab A4: Weather & Road ──
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

            # Festival impact
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

            # Temperature vs risk scatter
            if 'temperature' in ds.columns and 'risk_score' in ds.columns:
                st.markdown("<div class='section-header'>Temperature vs Risk Score</div>", unsafe_allow_html=True)
                sample = ds.sample(min(2000, len(ds)), random_state=42)
                fig = px.scatter(sample, x='temperature', y='risk_score',
                                 color='weather' if 'weather' in sample.columns else None,
                                 opacity=0.5, trendline='lowess',
                                 labels={'temperature': 'Temperature (°C)', 'risk_score': 'Risk Score'},
                                 color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=320, paper_bgcolor='white', plot_bgcolor='#fafafa',
                                  xaxis=dict(tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                                  yaxis=dict(tickfont=dict(size=11, family='Inter'), gridcolor='#eeeeee'),
                                  legend=dict(font=dict(size=11, family='Inter', color='#37474f')))
                st.plotly_chart(fig, use_container_width=True)

        # ── Tab A5: Raw Data ──
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
                st.download_button("📥 Download Full Dataset", csv_ds,
                                   "processed_dataset.csv", "text/csv", use_container_width=True)
            with col_dl2:
                stats_csv = ds[num_cols].describe().T.to_csv().encode('utf-8') if num_cols else b""
                st.download_button("📥 Download Statistics", stats_csv,
                                   "dataset_statistics.csv", "text/csv", use_container_width=True)
