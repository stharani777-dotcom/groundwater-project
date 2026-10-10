import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(
    page_title="Tamil Nadu Groundwater Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern UI
st.markdown("""
<style>
    /* Gradient Header styling */
    .header-container {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .header-container h1 {
        color: white !important;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .header-container p {
        color: #e0f2fe !important;
        font-size: 1.1rem;
        margin-bottom: 0;
    }
    
    # Custom styling for recommendation list
    .rec-card {
        background-color: #f8fafc;
        border-left: 4px solid #0284c7;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="header-container">
    <h1>💧 Tamil Nadu Groundwater Sustainability Dashboard</h1>
    <p>Interactive analytics, historical trend tracking, and predictive forecasting for water management.</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    df = pd.read_csv("data/groundwater_data.csv", low_memory=False)
    
    # Deduplicate columns
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    time_col = [c for c in df.columns if "Data Acquisition Time" in c or "Date" in c][0]
    district_col = [c for c in df.columns if "District" in c and "Code" not in c][0]
    gw_col = [c for c in df.columns if "Groundwater Level" in c or "meter" in c][0]
    
    clean_df = pd.DataFrame({
        "Date": pd.to_datetime(df[time_col], errors="coerce"),
        "District": df[district_col].astype(str).str.strip(),
        "Groundwater_Level": pd.to_numeric(df[gw_col], errors="coerce")
    })
    
    clean_df = clean_df.dropna(subset=["Date", "District", "Groundwater_Level"])
    clean_df = clean_df[clean_df["District"] != "nan"]
    clean_df["Year"] = clean_df["Date"].dt.year
    
    if "Rainfall" in df.columns:
        clean_df["Rainfall"] = pd.to_numeric(df["Rainfall"], errors="coerce").fillna(0)
    else:
        clean_df["Rainfall"] = 0.0
        
    return clean_df

df = load_and_process_data()

# Sidebar Controls
st.sidebar.title("🔍 Controls & Filters")

districts = sorted(df["District"].unique())
selected_district = st.sidebar.selectbox("Select District", districts)

min_year = int(df["Year"].min())
max_year = int(df["Year"].max())

if min_year < max_year:
    selected_years = st.sidebar.slider(
        "Select Year Range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year)
    )
else:
    selected_years = (min_year, max_year)

# Filter dataset
filtered = df[
    (df["District"] == selected_district) & 
    (df["Year"] >= selected_years[0]) & 
    (df["Year"] <= selected_years[1])
].copy()

# Annual aggregation
annual = (
    filtered.groupby("Year", as_index=False)
    .agg(
        groundwater=("Groundwater_Level", "mean"),
        rainfall=("Rainfall", "mean")
    )
    .sort_values("Year")
)

if annual.empty:
    st.error("No valid data available for the selected range.")
    st.stop()

# Calculations
latest_val = annual["groundwater"].iloc[-1]
first_val = annual["groundwater"].iloc[0]
diff = latest_val - first_val

# Metric Banner
m1, m2, m3, m4 = st.columns(4)

m1.metric(
    label="Latest Level (m)",
    value=f"{latest_val:.2f} m",
    delta=f"{diff:.2f} m" if diff != 0 else None,
    delta_color="inverse"
)
m2.metric(
    label="Peak Level",
    value=f"{annual['groundwater'].max():.2f} m"
)
m3.metric(
    label="Lowest Level",
    value=f"{annual['groundwater'].min():.2f} m"
)
m4.metric(
    label="Years Analyzed",
    value=f"{len(annual)} yrs"
)

st.write("---")

# Organized Tabs
tab1, tab2, tab3 = st.tabs(["📈 Historical Analysis", "🔮 5-Year Forecast", "🌱 Action Plan"])

with tab1:
    st.subheader(f"Groundwater Trends: {selected_district}")
    
    # Styled Plotly Line Chart
    fig = px.area(
        annual,
        x="Year",
        y="groundwater",
        markers=True,
        title=f"Average Water Table Depth in {selected_district}",
        labels={"groundwater": "Water Level (meters below ground)", "Year": "Year"},
        color_discrete_sequence=["#0284c7"]
    )
    
    fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Rainfall chart if available
    if annual["rainfall"].sum() > 0:
        st.subheader("Annual Rainfall Trend")
        r_fig = px.bar(
            annual,
            x="Year",
            y="rainfall",
            color_discrete_sequence=["#38bdf8"],
            title=f"Rainfall Distribution in {selected_district}"
        )
        r_fig.update_layout(template="plotly_white")
        st.plotly_chart(r_fig, use_container_width=True)

with tab2:
    st.subheader("Predictive Groundwater Level")
    
    if len(annual) >= 3:
        model = LinearRegression()
        model.fit(annual[["Year"]], annual["groundwater"])

        future_years = range(int(annual["Year"].max()) + 1, int(annual["Year"].max()) + 6)
        future = pd.DataFrame({"Year": future_years})
        future["groundwater"] = model.predict(future[["Year"]])

        # Combined Plot
        fig_fc = go.Figure()

        # Historic Line
        fig_fc.add_trace(go.Scatter(
            x=annual["Year"], 
            y=annual["groundwater"],
            mode="lines+markers",
            name="Historical Data",
            line=dict(color="#0284c7", width=3)
        ))

        # Forecast Line
        fig_fc.add_trace(go.Scatter(
            x=future["Year"], 
            y=future["groundwater"],
            mode="lines+markers",
            name="Linear Projection",
            line=dict(color="#e11d48", width=3, dash="dash")
        ))

        fig_fc.update_layout(
            title=f"5-Year Projection Model ({selected_district})",
            xaxis_title="Year",
            yaxis_title="Groundwater Level (m)",
            template="plotly_white",
            hovermode="x unified"
        )

        st.plotly_chart(fig_fc, use_container_width=True)
        st.info("💡 **Note:** Predictions are generated using a Linear Regression model fitted on historical annual trends.")
    else:
        st.warning("At least three years of historical data are required to render projections.")

with tab3:
    st.subheader("Recommended Water Conservation Strategies")
    
    recs = [
        "🌧️ **Rainwater Harvesting:** Mandate rooftop RWH systems across residential and commercial buildings.",
        "🏗️ **Artificial Recharge:** Construct check dams, percolation tanks, and recharge pits in key depletion zones.",
        "🌾 **Smart Irrigation:** Transition agriculture from flood irrigation to high-efficiency drip and sprinkler systems.",
        "💧 **Wastewater Recycling:** Treat and reuse industrial and domestic wastewater for non-potable purposes.",
        "📡 **Borewell Monitoring:** Implement telemetry sensors on high-extraction wells for automated depletion alerts.",
        "🌿 **Ecosystem Protection:** Restore natural wetlands, lakes, and urban recharge areas."
    ]
    
    for r in recs:
        st.markdown(f'<div class="rec-card">{r}</div>', unsafe_allow_html=True)

    st.write("---")
    
    st.download_button(
        label="📥 Download District Dataset (CSV)",
        data=filtered.to_csv(index=False),
        file_name=f"{selected_district}_groundwater_filtered.csv",
        mime="text/csv"
    )