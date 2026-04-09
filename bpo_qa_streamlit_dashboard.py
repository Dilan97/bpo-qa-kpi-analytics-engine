
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="BPO QA KPI Dashboard", layout="wide")
sns.set_theme(style="whitegrid")

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["evaluation_date"] = pd.to_datetime(df["evaluation_date"], errors="coerce")

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    df = df.drop_duplicates()
    df["month"] = df["evaluation_date"].dt.to_period("M").astype(str)
    return df

st.title("BPO QA KPI Dashboard")
st.caption("Analyze data for better Decision-Making")

uploaded_file = st.sidebar.file_uploader("Upload QA dataset (CSV)", type=["csv"])

default_path = "synthetic_bpo_qa_dataset.csv"
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    try:
        df = load_data(default_path)
        st.sidebar.success("Loaded bundled synthetic dataset")
    except FileNotFoundError:
        st.warning("Upload a CSV file to begin.")
        st.stop()

st.sidebar.header("Filters")
teams = st.sidebar.multiselect("Team", sorted(df["team_manager"].dropna().unique()), default=sorted(df["team_manager"].dropna().unique()))
campaigns = st.sidebar.multiselect("Campaign", sorted(df["campaign"].dropna().unique()), default=sorted(df["campaign"].dropna().unique()))
channels = st.sidebar.multiselect("Channel", sorted(df["channel"].dropna().unique()), default=sorted(df["channel"].dropna().unique()))
coach = st.sidebar.multiselect("Coaching Status", sorted(df["coaching_status"].dropna().unique()), default=sorted(df["coaching_status"].dropna().unique()))

filtered = df[
    df["team_manager"].isin(teams)
    & df["campaign"].isin(campaigns)
    & df["channel"].isin(channels)
    & df["coaching_status"].isin(coach)
].copy()

if filtered.empty:
    st.error("No records match the selected filters.")
    st.stop()

avg_qa = filtered["qa_score"].mean()
avg_aht = filtered["aht_seconds"].mean()
avg_csat = filtered["csat_score"].mean()
avg_compliance = filtered["compliance_score"].mean()
target_rate = (filtered["target_met"].eq("Met").mean()) * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Average QA", f"{avg_qa:.1f}")
c2.metric("Average AHT", f"{avg_aht:.0f}s")
c3.metric("Average CSAT", f"{avg_csat:.2f}")
c4.metric("Compliance", f"{avg_compliance:.1f}")
c5.metric("Target Met %", f"{target_rate:.1f}%")

st.subheader("Filtered Dataset Preview")
st.dataframe(filtered.head(50), use_container_width=True)

left, right = st.columns(2)

with left:
    st.subheader("QA Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(filtered["qa_score"], bins=20, kde=True, ax=ax)
    ax.set_xlabel("QA Score")
    ax.set_ylabel("Count")
    st.pyplot(fig)

with right:
    st.subheader("AHT vs QA Score")
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered, x="aht_seconds", y="qa_score", alpha=0.7, ax=ax)
    ax.set_xlabel("AHT (seconds)")
    ax.set_ylabel("QA Score")
    st.pyplot(fig)

left, right = st.columns(2)

with left:
    st.subheader("Average QA by Team")
    team_perf = filtered.groupby("team_manager")["qa_score"].mean().sort_values()
    fig, ax = plt.subplots()
    team_perf.plot(kind="barh", ax=ax)
    ax.set_xlabel("Average QA Score")
    ax.set_ylabel("Team")
    st.pyplot(fig)

with right:
    st.subheader("QA by Coaching Status")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered, x="coaching_status", y="qa_score", ax=ax)
    ax.set_xlabel("Coaching Status")
    ax.set_ylabel("QA Score")
    st.pyplot(fig)

st.subheader("Monthly QA Trend")
monthly_qa = filtered.groupby("month")["qa_score"].mean()
fig, ax = plt.subplots(figsize=(12, 5))
monthly_qa.plot(marker="o", ax=ax)
ax.set_xlabel("Month")
ax.set_ylabel("Average QA Score")
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Campaign Summary")
campaign_summary = (
    filtered.groupby("campaign")[["qa_score", "aht_seconds", "csat_score", "compliance_score"]]
    .mean()
    .sort_values("qa_score", ascending=False)
    .round(2)
)
st.dataframe(campaign_summary, use_container_width=True)

st.subheader("Insights & Analysis Context")
st.markdown(
    '''
This dashboard enables analysis of key BPO operational metrics including QA Score, AHT, CSAT, and Compliance.

The objective is to support data-driven decision-making by:
- Identifying performance trends across teams and campaigns
- Analyzing relationships between efficiency and quality metrics
- Highlighting opportunities for process improvement and performance optimization

The dataset used is synthetic and structured to simulate real-world BPO operational environments.
'''
)
