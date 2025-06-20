import streamlit as st
import pandas as pd
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download("punkt")

# ---------------------
# Sidebar Configuration
# ---------------------
st.sidebar.title("Advisor Briefing Tool")
uploaded_file = st.sidebar.file_uploader("Upload client CSV", type=["csv"])
st.sidebar.markdown("Sample columns: `Name`, `Notes`, `Portfolio_Value`, `Last_Contact_Date`")

# ---------------------
# Main Page
# ---------------------
st.title("AI-Powered Advisor Briefing")
st.markdown("This sample prototype demonstrates how financial advisors can use AI and NLP to auto-generate personalized meeting prep summaries based on client data.")

# ---------------------
# Process Uploaded File
# ---------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Client Overview")
    st.dataframe(df)

    # Analyzer
    analyzer = SentimentIntensityAnalyzer()

    summaries = []
    st.subheader("AI-Generated Briefing Notes")

    for index, row in df.iterrows():
        name = row.get("Name", "Client")
        notes = row.get("Notes", "")
        portfolio_value = row.get("Portfolio_Value", "")
        last_contact = row.get("Last_Contact_Date", "N/A")

        # Sentiment Analysis
        vader_score = analyzer.polarity_scores(notes)
        blob = TextBlob(notes)
        polarity = blob.sentiment.polarity

        sentiment_summary = (
            "Positive" if vader_score["compound"] > 0.2 else
            "Negative" if vader_score["compound"] < -0.2 else
            "Neutral"
        )

        # Generate Summary
        try:
            formatted_value = f"${float(portfolio_value):,.2f}"
        except:
            formatted_value = "N/A"

        summary = f"""
        **Client Name:** {name}  
        **Portfolio Value:** {formatted_value}  
        **Last Contact:** {last_contact}  
        **Sentiment Summary:** {sentiment_summary}  
        **Top Insight:** {notes[:150]}...
        """
        st.markdown(summary)
        st.markdown("---")
else:
    st.info("📁 Upload a CSV to begin generating AI briefings.")

# ---------------------
# Footer
# ---------------------
st.markdown("— *Advisor360° AI Briefing Tool Prototype*")