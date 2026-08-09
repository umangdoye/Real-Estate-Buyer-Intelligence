import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real Estate Buyer Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL STYLING
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

[data-testid="stMetric"] {
    background-color: #f8fafc;
    border: 1px solid #d9e2ec;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
}

[data-testid="stMetricLabel"] {
    color: #334155 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 28px !important;
    font-weight: 800 !important;
}

[data-testid="stMetricDelta"] {
    color: #475569 !important;
}

h1 {
    font-weight: 800;
}

h2 {
    font-weight: 700;
}

h3 {
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "real_estate_buyer_segments.csv"
    )

    return df


df = load_data()


# ============================================================
# TITLE
# ============================================================

st.title("🏠 Real Estate Buyer Intelligence")

st.caption(
    "Machine Learning Based Buyer Segmentation & Investment Profiling "
    "for Real Estate Market Intelligence"
)

st.markdown("""
This interactive platform transforms buyer data into actionable
real estate intelligence using **machine learning segmentation,
investment profiling, demographic analysis, financing behavior,
and geographic insights**.
""")


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("🔎 Buyer Filters")

country_filter = st.sidebar.multiselect(
    "Country",
    options=sorted(df["country"].dropna().unique()),
    default=[]
)

region_filter = st.sidebar.multiselect(
    "Region",
    options=sorted(df["region"].dropna().unique()),
    default=[]
)

purpose_filter = st.sidebar.multiselect(
    "Acquisition Purpose",
    options=sorted(
        df["acquisition_purpose"].dropna().unique()
    ),
    default=[]
)

client_filter = st.sidebar.multiselect(
    "Client Type",
    options=sorted(
        df["client_type"].dropna().unique()
    ),
    default=[]
)


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = df.copy()

if country_filter:
    filtered_df = filtered_df[
        filtered_df["country"].isin(country_filter)
    ]

if region_filter:
    filtered_df = filtered_df[
        filtered_df["region"].isin(region_filter)
    ]

if purpose_filter:
    filtered_df = filtered_df[
        filtered_df["acquisition_purpose"].isin(
            purpose_filter
        )
    ]

if client_filter:
    filtered_df = filtered_df[
        filtered_df["client_type"].isin(
            client_filter
        )
    ]


# ============================================================
# NAVIGATION TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "👥 Buyer Segments",
    "💰 Investment Intelligence",
    "🌎 Geographic Analysis"
])


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:

    st.subheader("📊 Buyer Intelligence Overview")

    col1, col2, col3, col4 = st.columns(4)

    # Total Buyers
    with col1:

        st.metric(
            "Total Buyers",
            f"{len(filtered_df):,}"
        )

    # Investment Buyers
    with col2:

        investment_count = (
            filtered_df[
                filtered_df["acquisition_purpose"]
                == "Investment"
            ].shape[0]
        )

        st.metric(
            "Investment Buyers",
            f"{investment_count:,}"
        )

    # Total Spending
    with col3:

        total_spend = filtered_df["total_spend"].sum()

        st.metric(
            "Total Property Spend",
            f"{total_spend:,.0f}"
        )

    # Satisfaction
    with col4:

        avg_satisfaction = (
            filtered_df["satisfaction_score"].mean()
        )

        st.metric(
            "Avg Satisfaction",
            f"{avg_satisfaction:.2f}"
        )

    st.markdown("---")

    # Buyer segment chart

    segment_counts = (
        filtered_df["segment_name"]
        .value_counts()
        .reset_index()
    )

    segment_counts.columns = [
        "Segment",
        "Buyers"
    ]

    fig_segments = px.bar(
        segment_counts,
        x="Segment",
        y="Buyers",
        text="Buyers",
        title="Buyer Distribution by Machine Learning Segment"
    )

    fig_segments.update_layout(
        xaxis_title="Buyer Segment",
        yaxis_title="Number of Buyers"
    )

    st.plotly_chart(
        fig_segments,
        use_container_width=True
    )


# ============================================================
# TAB 2 — BUYER SEGMENTS
# ============================================================

with tab2:

    st.subheader("👥 Buyer Segment Analysis")

    segment_counts = (
        filtered_df["segment_name"]
        .value_counts()
        .reset_index()
    )

    segment_counts.columns = [
        "Segment",
        "Buyers"
    ]

    fig_segments = px.bar(
        segment_counts,
        x="Segment",
        y="Buyers",
        text="Buyers",
        title="Buyers by Machine Learning Segment"
    )

    st.plotly_chart(
        fig_segments,
        use_container_width=True
    )

    st.markdown("### 📋 Segment Performance")

    segment_table = (
        filtered_df
        .groupby("segment_name")
        .agg(
            Buyers=("client_id", "count"),
            Avg_Age=("age", "mean"),
            Avg_Spending=("total_spend", "mean"),
            Avg_Properties=("properties_purchased", "mean"),
            Avg_Satisfaction=("satisfaction_score", "mean")
        )
        .round(2)
        .reset_index()
    )

    st.dataframe(
        segment_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 💡 Select a Segment")

    selected_segment = st.selectbox(
        "Choose a buyer segment",
        sorted(
            filtered_df["segment_name"]
            .dropna()
            .unique()
        )
    )

    segment_data = filtered_df[
        filtered_df["segment_name"]
        == selected_segment
    ]

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Buyers",
            f"{len(segment_data):,}"
        )

    with col2:

        st.metric(
            "Avg Age",
            f"{segment_data['age'].mean():.1f}"
        )

    with col3:

        st.metric(
            "Avg Spending",
            f"{segment_data['total_spend'].mean():,.0f}"
        )

    with col4:

        st.metric(
            "Avg Properties",
            f"{segment_data['properties_purchased'].mean():.1f}"
        )

    investment_rate = (
        segment_data["acquisition_purpose"]
        .eq("Investment")
        .mean()
        * 100
    )

    loan_rate = (
        segment_data["loan_applied"]
        .eq("Yes")
        .mean()
        * 100
    )

    company_rate = (
        segment_data["client_type"]
        .eq("Company")
        .mean()
        * 100
    )

    st.markdown("### 📌 Segment Characteristics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Investment-Oriented",
            f"{investment_rate:.1f}%"
        )

    with col2:
        st.metric(
            "Loan Users",
            f"{loan_rate:.1f}%"
        )

    with col3:
        st.metric(
            "Company Buyers",
            f"{company_rate:.1f}%"
        )

    st.markdown("### 🤖 Automated Segment Intelligence")

    if selected_segment == "Financing-Driven Investors":

        st.info("""
        **Financing-Driven Investors**

        This segment demonstrates strong investment orientation
        combined with significant financing dependence.

        **Recommended strategy:**

        • Promote investment-focused properties  
        • Provide financing and mortgage assistance  
        • Offer ROI-focused property information  
        • Use targeted investment campaigns
        """)

    elif selected_segment == "High-Value Corporate Buyers":

        st.info("""
        **High-Value Corporate Buyers**

        This segment represents high-value buyers with strong
        purchasing activity and significant corporate participation.

        **Recommended strategy:**

        • Dedicated corporate relationship management  
        • Bulk or multiple-property offerings  
        • Premium property recommendations  
        • Customized corporate investment packages
        """)

    else:

        st.info("""
        **Segment Insight**

        This segment should be targeted using its observed
        demographic, financial, geographic, and purchasing behavior.
        """)


# ============================================================
# TAB 3 — INVESTMENT INTELLIGENCE
# ============================================================

with tab3:

    st.subheader("💰 Investment Behavior Analysis")

    col1, col2 = st.columns(2)

    # Home vs Investment

    with col1:

        purpose_counts = (
            filtered_df["acquisition_purpose"]
            .value_counts()
            .reset_index()
        )

        purpose_counts.columns = [
            "Purpose",
            "Buyers"
        ]

        fig_purpose = px.pie(
            purpose_counts,
            names="Purpose",
            values="Buyers",
            hole=0.45,
            title="Home vs Investment Buyers"
        )

        st.plotly_chart(
            fig_purpose,
            use_container_width=True
        )

    # Loan behavior

    with col2:

        loan_counts = (
            filtered_df["loan_applied"]
            .value_counts()
            .reset_index()
        )

        loan_counts.columns = [
            "Loan Applied",
            "Buyers"
        ]

        fig_loan = px.pie(
            loan_counts,
            names="Loan Applied",
            values="Buyers",
            hole=0.45,
            title="Buyer Financing Behavior"
        )

        st.plotly_chart(
            fig_loan,
            use_container_width=True
        )

    st.markdown("### 📈 Investment Performance")

    investment_analysis = (
        filtered_df
        .groupby("segment_name")
        .agg(
            Buyers=("client_id", "count"),
            Average_Spend=("total_spend", "mean"),
            Average_Properties=("properties_purchased", "mean")
        )
        .round(2)
        .reset_index()
    )

    st.dataframe(
        investment_analysis,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 💵 Average Spending by Segment")

    spending_by_segment = (
        filtered_df
        .groupby("segment_name")["total_spend"]
        .mean()
        .reset_index()
    )

    spending_by_segment.columns = [
        "Segment",
        "Average Spending"
    ]

    fig_spending = px.bar(
        spending_by_segment,
        x="Segment",
        y="Average Spending",
        text_auto=".2s",
        title="Average Property Spending by Buyer Segment"
    )

    st.plotly_chart(
        fig_spending,
        use_container_width=True
    )

    st.markdown("### 📊 Investment Orientation by Segment")

    investment_profile = (
        filtered_df
        .assign(
            Investment_Buyer=filtered_df["acquisition_purpose"].eq(
                "Investment")
        )
        .groupby("segment_name")
        .agg(
            Investment_Rate=("Investment_Buyer", "mean")
        )
        .reset_index()
    )

    investment_profile["Investment_Rate"] *= 100

    investment_profile["Investment_Rate"] = (
        investment_profile["Investment_Rate"].round(2)
    )

    fig_investment = px.bar(
        investment_profile,
        x="segment_name",
        y="Investment_Rate",
        text="Investment_Rate",
        title="Investment-Oriented Buyers by Segment",
        labels={
            "segment_name": "Buyer Segment",
            "Investment_Rate": "Investment Buyers (%)"
        }
    )

    fig_investment.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig_investment,
        use_container_width=True
    )
    st.markdown("### 🏦 Financing Dependency by Segment")

    financing_profile = (
        filtered_df
        .assign(
            Loan_User=filtered_df["loan_applied"].eq("Yes")
        )
        .groupby("segment_name")
        .agg(
            Loan_Rate=("Loan_User", "mean")
        )
        .reset_index()
    )

    financing_profile["Loan_Rate"] *= 100

    financing_profile["Loan_Rate"] = (
        financing_profile["Loan_Rate"].round(2)
    )

    fig_financing = px.bar(
        financing_profile,
        x="segment_name",
        y="Loan_Rate",
        text="Loan_Rate",
        title="Loan Dependency by Buyer Segment",
        labels={
            "segment_name": "Buyer Segment",
            "Loan_Rate": "Loan Users (%)"
        }
    )

    fig_financing.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig_financing,
        use_container_width=True
    )
# ============================================================
# TAB 4 — GEOGRAPHIC ANALYSIS
# ============================================================

with tab4:

    st.subheader("🌎 Geographic Buyer Analysis")

    # Country analysis

    country_analysis = (
        filtered_df
        .groupby("country")
        .agg(
            Buyers=("client_id", "count"),
            Total_Spend=("total_spend", "sum")
        )
        .reset_index()
        .sort_values(
            "Buyers",
            ascending=False
        )
    )

    fig_geo = px.bar(
        country_analysis,
        x="country",
        y="Buyers",
        text="Buyers",
        title="Buyer Distribution by Country"
    )

    fig_geo.update_layout(
        xaxis_title="Country",
        yaxis_title="Number of Buyers"
    )

    st.plotly_chart(
        fig_geo,
        use_container_width=True
    )

    # Region analysis

    region_analysis = (
        filtered_df
        .groupby("region")
        .agg(
            Buyers=("client_id", "count"),
            Total_Spend=("total_spend", "sum")
        )
        .reset_index()
        .sort_values(
            "Buyers",
            ascending=False
        )
        .head(15)
    )

    fig_region = px.bar(
        region_analysis.sort_values("Buyers"),
        x="Buyers",
        y="region",
        orientation="h",
        text="Buyers",
        title="Top 15 Regions by Buyer Count"
    )

    fig_region.update_layout(
        yaxis_title="Region",
        xaxis_title="Number of Buyers"
    )

    st.plotly_chart(
        fig_region,
        use_container_width=True
    )
    st.markdown("### 🗺️ Buyer Geographic Intelligence")

    map_data = (
        filtered_df
        .groupby(["country", "segment_name"])
        .agg(
            Buyers=("client_id", "count"),
            Total_Spend=("total_spend", "sum")
        )
        .reset_index()
    )

    fig_map = px.scatter_geo(
        map_data,
        locations="country",
        locationmode="country names",
        size="Buyers",
        color="segment_name",
        hover_name="country",
        hover_data={
            "Buyers": True,
            "Total_Spend": ":,.0f"
        },
        title="Buyer Segments by Country",
        projection="natural earth"
    )

    fig_map.update_layout(
        height=600
    )

    st.plotly_chart(
        fig_map,
        use_container_width=True
    )

# ============================================================
# FILTERED BUYER DATA
# ============================================================

st.markdown("---")

st.subheader("📋 Filtered Buyer Data")

display_columns = [
    "client_id",
    "client_type",
    "country",
    "region",
    "acquisition_purpose",
    "loan_applied",
    "properties_purchased",
    "total_spend",
    "segment_name"
]

available_columns = [
    col for col in display_columns
    if col in filtered_df.columns
]

st.dataframe(
    filtered_df[available_columns],
    use_container_width=True,
    hide_index=True
)
# ============================================================
# DOWNLOAD FILTERED DATA
# ============================================================

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Filtered Buyer Data",
    data=csv_data,
    file_name="filtered_real_estate_buyers.csv",
    mime="text/csv"
)
# ============================================================
# KEY BUSINESS FINDINGS
# ============================================================

st.markdown("---")

st.header("🔍 Key Business Findings")

total_buyers = len(filtered_df)

investment_rate = (
    filtered_df["acquisition_purpose"]
    .eq("Investment")
    .mean() * 100
)

loan_rate = (
    filtered_df["loan_applied"]
    .eq("Yes")
    .mean() * 100
)

top_country = (
    filtered_df["country"]
    .value_counts()
    .idxmax()
)

top_segment = (
    filtered_df["segment_name"]
    .value_counts()
    .idxmax()
)

col1, col2 = st.columns(2)

with col1:

    st.markdown("### 📌 Market Overview")

    st.write(
        f"• **{total_buyers:,}** buyers are represented "
        f"in the current filtered view."
    )

    st.write(
        f"• **{investment_rate:.1f}%** of buyers "
        f"are investment-oriented."
    )

    st.write(
        f"• **{loan_rate:.1f}%** of buyers "
        f"have applied for financing."
    )

with col2:

    st.markdown("### 🎯 Market Concentration")

    st.write(
        f"• The largest buyer market is **{top_country}**."
    )

    st.write(
        f"• The most represented segment is "
        f"**{top_segment}**."
    )

    st.write(
        "• Buyer segmentation can support targeted "
        "marketing and personalized property recommendations."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Real Estate Buyer Intelligence | "
    "Machine Learning • K-Means • Hierarchical Clustering • PCA"
)

st.caption(
    "Developed as part of the Unified Mentor Internship Project"
)
