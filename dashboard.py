import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Live SRE Radar", layout="wide")

st.title("Infrastructure Health Pulse")

conn = sqlite3.connect("reddit.db")
df = pd.read_sql("SELECT * FROM reddit_items", conn)
conn.close()

if df.empty:
    st.warning("No data collected yet. Run collector.py to populate the database.")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

sentiment_weights = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
df["score_weight"] = df["sentiment"].map(sentiment_weights)

df.set_index("timestamp", inplace=True)
hourly_summary = df.resample("1h").agg(
    net_sentiment=("score_weight", "mean"),
    total_volume=("id", "count"),
    neg_count=("sentiment", lambda s: (s == "negative").sum())
).reset_index()

latest_hour = hourly_summary.iloc[-1] if not hourly_summary.empty else None
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Items Monitored", len(df))
with col2:
    if latest_hour is not None:
        status_color = "OUTAGE RISK" if latest_hour["net_sentiment"] < -0.4 and latest_hour["total_volume"] > 5 else "STABLE"
        st.metric("Current Edge Status", status_color)
with col3:
    if latest_hour is not None:
        st.metric("Latest Hour Net Sentiment", f"{latest_hour['net_sentiment']:.2f}")

st.subheader("Community Sentiment & Volume Over Time")
fig = px.line(
    hourly_summary,
    x="timestamp",
    y="net_sentiment",
    title="Rolling Net Sentiment Score (-1: Outage/Crisis, +1: Healthy/Positive)",
    labels={"net_sentiment": "Net Sentiment", "timestamp": "UTC Time"},
    markers=True
)
fig.add_hline(y=-0.4, line_dash="dash", line_color="red", annotation_text="Outage Threshold Alert")
st.plotly_chart(fig, use_container_width=True)

fig_vol = px.bar(
    hourly_summary,
    x="timestamp",
    y="total_volume",
    title="Hourly Post Volume (Surges often indicate regional outages)",
    labels={"total_volume": "Items", "timestamp": "UTC Time"}
)
st.plotly_chart(fig_vol, use_container_width=True)

st.subheader("Inspect Incidents (Select a Time Window)")
selected_sentiment = st.radio("Filter items by sentiment:", ["All", "Negative", "Neutral", "Positive"], horizontal=True)

filtered_df = df.reset_index()
if selected_sentiment != "All":
    filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment.lower()]

st.dataframe(
    filtered_df[["timestamp", "subreddit", "sentiment", "confidence", "text"]].tail(50),
    use_container_width=True
)