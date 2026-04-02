"""
Visualization Module
Creates maps, charts, and plots for water stress & drought analysis.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

logger = logging.getLogger(__name__)


def create_global_choropleth(df, value_col="drought_composite_score", title=None):
    """
    Create a global choropleth map of water stress indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Data with country_code column
    value_col : str
        Column to visualize
    title : str
        Map title

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.express as px

    # Aggregate to country level
    if "country_code" in df.columns:
        country_data = df.groupby(["country_code", "country"]).agg({
            value_col: "mean"
        }).reset_index()
    else:
        country_data = df

    fig = px.choropleth(
        country_data,
        locations="country_code",
        locationmode="ISO-3",
        color=value_col,
        hover_name="country" if "country" in country_data.columns else "country_code",
        color_continuous_scale="RdYlBu_r",
        title=title or f"Global {value_col.replace('_', ' ').title()}",
        labels={value_col: value_col.replace("_", " ").title()},
    )

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        template="plotly_dark",
    )

    return fig


def create_time_series_plot(df, country_code, value_cols=None, title=None):
    """
    Create time series plot for a specific country/region.

    Parameters
    ----------
    df : pd.DataFrame
        Time series data with date column
    country_code : str
        ISO-3 country code
    value_cols : list
        Columns to plot
    title : str
        Plot title

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    region_data = df[df["country_code"] == country_code].copy()
    if "date" in region_data.columns:
        region_data["date"] = pd.to_datetime(region_data["date"])
        region_data = region_data.sort_values("date")

    if value_cols is None:
        value_cols = [c for c in ["tws_anomaly_cm", "drought_composite_score", "water_stress_score"]
                      if c in region_data.columns]

    country_name = region_data["country"].iloc[0] if "country" in region_data.columns else country_code

    fig = make_subplots(
        rows=len(value_cols), cols=1,
        shared_xaxes=True,
        subplot_titles=[c.replace("_", " ").title() for c in value_cols],
        vertical_spacing=0.08,
    )

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    for i, col in enumerate(value_cols):
        if col in region_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=region_data["date"],
                    y=region_data[col],
                    name=col.replace("_", " ").title(),
                    line=dict(color=colors[i % len(colors)], width=2),
                    fill="tozeroy" if "anomaly" in col else None,
                ),
                row=i + 1, col=1,
            )

    fig.update_layout(
        title=title or f"Water Stress Indicators — {country_name}",
        height=250 * len(value_cols),
        template="plotly_dark",
        showlegend=True,
    )

    return fig


def create_drought_distribution(df):
    """Create drought risk distribution chart."""
    import plotly.express as px

    if "drought_risk_class" not in df.columns:
        return None

    dist = df["drought_risk_class"].value_counts().reset_index()
    dist.columns = ["Risk Level", "Count"]

    color_map = {
        "Low": "#2ecc71",
        "Moderate": "#f39c12",
        "High": "#e74c3c",
        "Extreme": "#8e44ad",
    }

    fig = px.bar(
        dist, x="Risk Level", y="Count",
        color="Risk Level",
        color_discrete_map=color_map,
        title="Global Drought Risk Distribution",
    )

    fig.update_layout(
        template="plotly_dark",
        height=400,
        showlegend=False,
    )

    return fig


def create_correlation_heatmap(df):
    """Create a correlation matrix heatmap."""
    import plotly.figure_factory as ff

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Select most relevant columns
    relevant = [c for c in numeric_cols if any(
        kw in c.lower() for kw in ["stress", "drought", "tws", "water", "ground", "precip"]
    )]

    if len(relevant) < 3:
        relevant = list(numeric_cols[:10])

    corr = df[relevant].corr().round(3)

    fig = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale="RdBu_r",
        showscale=True,
    )

    fig.update_layout(
        title="Feature Correlation Matrix",
        height=600,
        width=800,
        template="plotly_dark",
    )

    return fig


def create_folium_map(df, value_col="drought_composite_score"):
    """
    Create an interactive Folium map with markers.

    Returns
    -------
    folium.Map
    """
    import folium
    from folium.plugins import HeatMap

    m = folium.Map(
        location=config.MAP_CENTER,
        zoom_start=config.MAP_ZOOM,
        tiles="CartoDB dark_matter",
    )

    # Color coding
    def get_color(value):
        if value < 1:
            return "green"
        elif value < 2:
            return "orange"
        elif value < 3.5:
            return "red"
        else:
            return "darkred"

    # Add markers for each country
    if "country_code" in df.columns:
        country_data = df.groupby(["country_code", "country"]).agg({
            value_col: "mean",
            "lat": "mean",
            "lon": "mean",
        }).reset_index()

        for _, row in country_data.iterrows():
            if pd.notna(row["lat"]) and pd.notna(row["lon"]):
                score = row[value_col]
                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=max(5, score * 3),
                    color=get_color(score),
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"<b>{row['country']}</b><br>"
                          f"{value_col}: {score:.2f}",
                    tooltip=row["country"],
                ).add_to(m)

    return m


def create_seasonal_comparison(df, country_codes=None):
    """Create seasonal water stress comparison chart."""
    import plotly.express as px

    if "season" not in df.columns or "drought_composite_score" not in df.columns:
        return None

    if country_codes:
        df = df[df["country_code"].isin(country_codes)]

    seasonal = df.groupby(["country", "season"]).agg({
        "drought_composite_score": "mean"
    }).reset_index()

    fig = px.bar(
        seasonal, x="country", y="drought_composite_score",
        color="season",
        barmode="group",
        title="Seasonal Water Stress Comparison",
        color_discrete_map={
            "DJF": "#3498db", "MAM": "#2ecc71",
            "JJA": "#e74c3c", "SON": "#f39c12",
        },
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_tickangle=-45,
    )

    return fig
