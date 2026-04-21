"""
🌍 Water Stress & Drought Index Tracker — Streamlit Dashboard
Interactive dashboard for monitoring global water scarcity and drought levels.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title=config.DASHBOARD_TITLE,
    page_icon=config.DASHBOARD_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background-color: #0e1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16192b 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }

    .metric-value {
        font-size: 2.4em;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .metric-label {
        font-size: 0.9em;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 4px 0 0 0;
    }

    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85em;
        text-transform: uppercase;
    }

    .risk-low { background: rgba(46,204,113,0.2); color: #2ecc71; border: 1px solid rgba(46,204,113,0.3); }
    .risk-moderate { background: rgba(243,156,18,0.2); color: #f39c12; border: 1px solid rgba(243,156,18,0.3); }
    .risk-high { background: rgba(231,76,60,0.2); color: #e74c3c; border: 1px solid rgba(231,76,60,0.3); }
    .risk-extreme { background: rgba(142,68,173,0.2); color: #8e44ad; border: 1px solid rgba(142,68,173,0.3); }

    .section-header {
        font-size: 1.5em;
        font-weight: 600;
        color: #e0e0e0;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(102,126,234,0.3);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        background: #1a1f2e;
    }

    div[data-testid="stMetricValue"] > div {
        font-size: 1.8em;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_data():
    """Load the consolidated dataset."""
    data_path = config.CONSOLIDATED_CSV
    if data_path.exists():
        df = pd.read_csv(data_path)
        if "date" in df.columns:
            # Force to consistent datetime64 without mixed tz objects
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_localize(None)
        return df
    return None


@st.cache_resource
def load_model():
    """Load the trained drought classifier."""
    model_files = list(config.MODELS_DIR.glob("drought_classifier_*.joblib"))
    if model_files:
        return joblib.load(model_files[0])
    return None

@st.cache_resource
def load_reg_model():
    """Load the trained drought regressor (stress predictor)."""
    model_files = list(config.MODELS_DIR.glob("stress_predictor_*.joblib"))
    if model_files:
        return joblib.load(model_files[0])
    return None


def render_metric_card(label, value, delta=None):
    """Render a custom styled metric card."""
    delta_html = ""
    if delta is not None:
        color = "#2ecc71" if delta >= 0 else "#e74c3c"
        arrow = "↑" if delta >= 0 else "↓"
        delta_html = f'<p style="color:{color};font-size:0.9em;margin:4px 0 0 0;">{arrow} {abs(delta):.1f}%</p>'

    st.markdown(f"""
    <div class="metric-card">
        <p class="metric-value">{value}</p>
        <p class="metric-label">{label}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def get_risk_badge(risk_class):
    """Return HTML badge for risk class."""
    class_map = {
        "Low": "risk-low",
        "Moderate": "risk-moderate",
        "High": "risk-high",
        "Extreme": "risk-extreme",
    }
    css = class_map.get(risk_class, "risk-moderate")
    return f'<span class="risk-badge {css}">{risk_class}</span>'


# ─── Main App ─────────────────────────────────────────────────────────────────
def main():
    # Title
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 30px 0;">
        <h1 style="font-size:2.5em; font-weight:700; margin:0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            💧 Water Stress & Drought Index Tracker
        </h1>
        <p style="color:#8892a4; font-size:1.1em; margin-top:8px;">
            Global monitoring of water scarcity, drought risk, and water storage anomalies
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df = load_data()

    if df is None:
        st.warning("⚠️ No data found. Run the data extraction pipeline first:")
        st.code("python -m src.pipeline.quarterly_pipeline", language="bash")
        st.info("Or run individual extractors from the `notebooks/` directory.")
        return

    # ─── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎛️ Filters")

        # Country filter
        countries = sorted(df["country"].unique()) if "country" in df.columns else []
        selected_countries = st.multiselect(
            "Select Countries",
            options=countries,
            default=countries[:5] if len(countries) >= 5 else countries,
        )

        # Date range
        if "date" in df.columns and df["date"].notna().any():
            date_min = df["date"].min()
            date_max = df["date"].max()
            date_range = st.date_input(
                "Date Range",
                value=(date_min, date_max),
                min_value=date_min,
                max_value=date_max,
            )
        else:
            date_range = None

        # Risk level filter
        if "drought_risk_class" in df.columns:
            risk_levels = st.multiselect(
                "Risk Levels",
                options=["Low", "Moderate", "High", "Extreme"],
                default=["Low", "Moderate", "High", "Extreme"],
            )
        else:
            risk_levels = None

        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.markdown(f"**Records:** {len(df):,}")
        st.markdown(f"**Countries:** {df['country_code'].nunique() if 'country_code' in df.columns else 'N/A'}")
        if "date" in df.columns and df["date"].notna().any():
            st.markdown(f"**Period:** {df['date'].min().strftime('%Y-%m')} → {df['date'].max().strftime('%Y-%m')}")

    # Apply filters
    filtered = df.copy()
    if selected_countries:
        filtered = filtered[filtered["country"].isin(selected_countries)]
    if date_range and len(date_range) == 2 and "date" in filtered.columns:
        filtered = filtered[
            (filtered["date"] >= pd.Timestamp(date_range[0])) &
            (filtered["date"] <= pd.Timestamp(date_range[1]))
        ]
    if risk_levels and "drought_risk_class" in filtered.columns:
        filtered = filtered[filtered["drought_risk_class"].isin(risk_levels)]

    # ─── Key Metrics ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_score = filtered["drought_composite_score"].mean() if "drought_composite_score" in filtered.columns else 0
        render_metric_card("Avg Drought Score", f"{avg_score:.2f}")

    with col2:
        if "drought_risk_class" in filtered.columns:
            high_risk = (filtered["drought_risk_class"].isin(["High", "Extreme"])).mean() * 100
            render_metric_card("High/Extreme Risk", f"{high_risk:.1f}%")
        else:
            render_metric_card("High/Extreme Risk", "N/A")

    with col3:
        if "tws_anomaly_cm" in filtered.columns:
            avg_tws = filtered["tws_anomaly_cm"].mean()
            render_metric_card("Avg TWS Anomaly", f"{avg_tws:.1f} cm")
        else:
            render_metric_card("Avg TWS Anomaly", "N/A")

    with col4:
        countries_count = filtered["country_code"].nunique() if "country_code" in filtered.columns else 0
        render_metric_card("Countries Tracked", str(countries_count))

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Global Map", "📈 Time Series", "🎯 Risk Analysis",
        "🔮 Predictions", "📥 Data Explorer"
    ])

    # ── Tab 1: Global Map ──
    with tab1:
        st.markdown('<p class="section-header">Global Water Stress Overview</p>', unsafe_allow_html=True)

        map_col = st.selectbox(
            "Select indicator to map:",
            options=[c for c in ["drought_composite_score", "water_stress_score",
                                  "tws_anomaly_cm", "drought_risk_score"]
                     if c in filtered.columns],
            key="map_indicator",
        )

        if map_col:
            country_avg = filtered.groupby(["country_code", "country"]).agg({
                map_col: "mean", "lat": "mean", "lon": "mean"
            }).reset_index()

            fig = px.choropleth(
                country_avg,
                locations="country_code",
                locationmode="ISO-3",
                color=map_col,
                hover_name="country",
                color_continuous_scale="RdYlBu_r",
                title=f"Global {map_col.replace('_', ' ').title()}",
            )
            fig.update_layout(
                geo=dict(showframe=False, showcoastlines=True, projection_type="natural earth",
                         bgcolor="rgba(0,0,0,0)"),
                height=550, margin=dict(l=0, r=0, t=40, b=0),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 2: Time Series ──
    with tab2:
        st.markdown('<p class="section-header">Regional Time Series Analysis</p>', unsafe_allow_html=True)

        if "country" in filtered.columns:
            ts_country = st.selectbox(
                "Select country:", sorted(filtered["country"].unique()), key="ts_country"
            )

            ts_cols = [c for c in ["tws_anomaly_cm", "drought_composite_score",
                                    "groundwater_anomaly_cm", "water_stress_score"]
                       if c in filtered.columns]

            ts_indicators = st.multiselect(
                "Select indicators:", ts_cols, default=ts_cols[:2], key="ts_indicators"
            )

            if ts_country and ts_indicators:
                region = filtered[filtered["country"] == ts_country].sort_values("date")

                for indicator in ts_indicators:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=region["date"], y=region[indicator],
                        mode="lines", name=indicator.replace("_", " ").title(),
                        line=dict(width=2),
                        fill="tozeroy" if "anomaly" in indicator else None,
                    ))

                    # Add trend line
                    if len(region) > 10:
                        z = np.polyfit(range(len(region)), region[indicator].fillna(0), 1)
                        p = np.poly1d(z)
                        fig.add_trace(go.Scatter(
                            x=region["date"], y=p(range(len(region))),
                            mode="lines", name="Trend",
                            line=dict(width=2, dash="dash", color="rgba(255,255,255,0.5)"),
                        ))

                    fig.update_layout(
                        title=f"{indicator.replace('_', ' ').title()} — {ts_country}",
                        template="plotly_dark", height=350,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Risk Analysis ──
    with tab3:
        st.markdown('<p class="section-header">Drought Risk Analysis</p>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            if "drought_risk_class" in filtered.columns:
                dist = filtered["drought_risk_class"].value_counts().reset_index()
                dist.columns = ["Risk Level", "Count"]

                fig = px.pie(
                    dist, values="Count", names="Risk Level",
                    color="Risk Level",
                    color_discrete_map={
                        "Low": "#2ecc71", "Moderate": "#f39c12",
                        "High": "#e74c3c", "Extreme": "#8e44ad"
                    },
                    title="Risk Level Distribution",
                    hole=0.4,
                )
                fig.update_layout(template="plotly_dark", height=400,
                                  paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if "drought_composite_score" in filtered.columns:
                fig = px.histogram(
                    filtered, x="drought_composite_score",
                    nbins=30, title="Drought Composite Score Distribution",
                    color_discrete_sequence=["#667eea"],
                )
                fig.update_layout(template="plotly_dark", height=400,
                                  paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)

        # Top risk countries
        if "drought_composite_score" in filtered.columns:
            st.markdown("#### 🔴 Highest Risk Countries")
            top_risk = (
                filtered.groupby("country")["drought_composite_score"]
                .mean().sort_values(ascending=False).head(15)
                .reset_index()
            )
            top_risk.columns = ["Country", "Avg Drought Score"]

            fig = px.bar(
                top_risk, x="Avg Drought Score", y="Country",
                orientation="h", color="Avg Drought Score",
                color_continuous_scale="RdYlBu_r",
                title="Top 15 Countries by Drought Composite Score",
            )
            fig.update_layout(
                template="plotly_dark", height=500,
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Predictions ──
    with tab4:
        st.markdown('<p class="section-header">Machine Learning Predictors</p>', unsafe_allow_html=True)

        pred_tab1, pred_tab2 = st.tabs(["🚦 Drought Risk Classification", "📈 3-Month Stress Forecast"])

        # ── Sub-tab 1: Classification ──
        with pred_tab1:
            model_data = load_model()
            if model_data:
                st.success("✅ Classification model loaded successfully")

                st.markdown("#### Input Parameters")
                pcol1, pcol2, pcol3 = st.columns(3)

                with pcol1:
                    water_stress = st.slider("Water Stress Score (0-5)", 0.0, 5.0, 2.5, 0.1, key="clf_ws")
                    drought_risk = st.slider("Drought Risk Score (0-5)", 0.0, 5.0, 2.0, 0.1, key="clf_dr")
                with pcol2:
                    tws_anomaly = st.slider("TWS Anomaly (cm)", -30.0, 30.0, 0.0, 0.5, key="clf_tws")
                    gw_anomaly = st.slider("Groundwater Anomaly (cm)", -20.0, 20.0, 0.0, 0.5, key="clf_gw")
                with pcol3:
                    water_depletion = st.slider("Water Depletion Score", 0.0, 5.0, 1.5, 0.1, key="clf_wd")
                    composite = st.slider("Drought Composite", 0.0, 5.0, 2.0, 0.1, key="clf_comp")

                if st.button("🔮 Predict Drought Risk Class", type="primary"):
                    model = model_data["model"]
                    features = model_data["feature_names"]

                    # Build input vector
                    input_dict = {f: 0.0 for f in features}
                    field_map = {
                        "water_stress_score": water_stress,
                        "drought_risk_score": drought_risk,
                        "tws_anomaly_cm": tws_anomaly,
                        "groundwater_anomaly_cm": gw_anomaly,
                        "water_depletion_score": water_depletion,
                        "drought_composite_score": composite,
                    }
                    for k, v in field_map.items():
                        if k in input_dict:
                            input_dict[k] = v

                    X = pd.DataFrame([input_dict])
                    pred = model.predict(X)
                    label_encoder = model_data["label_encoder"]
                    risk_class = label_encoder.inverse_transform(pred)[0]

                    proba = model.predict_proba(X)[0]
                    proba_dict = dict(zip(label_encoder.classes_, proba))

                    st.markdown(f"### Predicted Risk: {get_risk_badge(risk_class)}", unsafe_allow_html=True)

                    # Probability chart
                    prob_df = pd.DataFrame({
                        "Risk Level": list(proba_dict.keys()),
                        "Probability": list(proba_dict.values()),
                    })
                    fig = px.bar(
                        prob_df, x="Risk Level", y="Probability",
                        color="Risk Level",
                        color_discrete_map={
                            "Low": "#2ecc71", "Moderate": "#f39c12",
                            "High": "#e74c3c", "Extreme": "#8e44ad"
                        },
                        title="Prediction Probabilities",
                    )
                    fig.update_layout(
                        template="plotly_dark", height=350,
                        paper_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No classification model found. Run the ML pipeline first.")
                st.code("python train_models.py", language="bash")

        # ── Sub-tab 2: Regression ──
        with pred_tab2:
            reg_data = load_reg_model()
            if reg_data:
                st.success(f"✅ Regression model loaded successfully ({reg_data.get('model_name', 'Regressor')})")

                st.markdown("#### Input Parameters")
                rcol1, rcol2, rcol3 = st.columns(3)

                with rcol1:
                    r_water_stress = st.slider("Current Water Stress (0-5)", 0.0, 5.0, 2.5, 0.1, key="reg_ws")
                    r_drought_risk = st.slider("Current Drought Risk (0-5)", 0.0, 5.0, 2.0, 0.1, key="reg_dr")
                with rcol2:
                    r_tws_anomaly = st.slider("Current TWS Anomaly (cm)", -30.0, 30.0, 0.0, 0.5, key="reg_tws")
                    r_precip = st.slider("Annual Precipitation (mm)", 0.0, 3000.0, 500.0, 50.0, key="reg_precip")
                with rcol3:
                    r_composite = st.slider("Current Drought Composite", 0.0, 5.0, 2.0, 0.1, key="reg_comp")
                    r_lag3 = st.slider("Drought Composite (3 mos ago)", 0.0, 5.0, 2.0, 0.1, key="reg_lag3")

                if st.button("🔮 Forecast 3-Month Drought Score", type="primary"):
                    model = reg_data["model"]
                    scaler = reg_data["scaler"]
                    features = reg_data["feature_names"]

                    # Build input vector
                    input_dict = {f: 0.0 for f in features}
                    field_map = {
                        "water_stress_score": r_water_stress,
                        "drought_risk_score": r_drought_risk,
                        "tws_anomaly_cm": r_tws_anomaly,
                        "precipitation_mm": r_precip,
                        "drought_composite_score": r_composite,
                        "drought_composite_score_lag1": r_composite,
                        "drought_composite_score_lag3": r_lag3,
                        "drought_composite_score_rolling_mean_3": (r_composite + r_lag3) / 2.0,
                    }
                    for k, v in field_map.items():
                        if k in input_dict:
                            input_dict[k] = v

                    X = pd.DataFrame([input_dict])
                    if scaler:
                        X_scaled = scaler.transform(X)
                        pred_score = model.predict(X_scaled)[0]
                    else:
                        pred_score = model.predict(X)[0]

                    # Clip to valid range 0-5
                    pred_score = max(0.0, min(5.0, pred_score))

                    st.markdown(f"### Forecasted 3-Month Drought Composite Score: **{pred_score:.2f}**")
                    
                    # Also classify it for easy reading
                    if pred_score < 1.0: forecasted_risk = "Low"
                    elif pred_score < 2.0: forecasted_risk = "Moderate"
                    elif pred_score < 3.5: forecasted_risk = "High"
                    else: forecasted_risk = "Extreme"
                    
                    st.markdown(f"**Implied Risk Class:** {get_risk_badge(forecasted_risk)}", unsafe_allow_html=True)
            else:
                st.warning("⚠️ No regression model found. Run the ML pipeline first.")
                st.code("python train_models.py", language="bash")

    # ── Tab 5: Data Explorer ──
    with tab5:
        st.markdown('<p class="section-header">Data Explorer</p>', unsafe_allow_html=True)

        # Column selector
        display_cols = st.multiselect(
            "Select columns to display:",
            options=list(filtered.columns),
            default=[c for c in ["country", "date", "drought_composite_score",
                                  "drought_risk_class", "tws_anomaly_cm", "water_stress_score"]
                     if c in filtered.columns],
        )

        if display_cols:
            st.dataframe(
                filtered[display_cols].head(500),
                use_container_width=True,
                height=400,
            )

        # Download button
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="drought_water_stress_filtered.csv",
            mime="text/csv",
        )

        # Summary statistics
        with st.expander("📊 Summary Statistics"):
            st.dataframe(
                filtered.select_dtypes(include=[np.number]).describe().round(3),
                use_container_width=True,
            )

    # ─── Footer ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:40px 0 20px 0; color:#555; font-size:0.85em;">
        <p>Data sources: WRI Aqueduct 4.0 | FAO AQUASTAT | NASA GRACE/GRACE-FO</p>
        <p>EDA-15: Water Stress & Drought Index Tracker</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
