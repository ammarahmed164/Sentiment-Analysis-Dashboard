import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.set_page_config(page_title="Sentiment Analysi s Dashboard", layout="wide")
st.title("📊 Sentiment Analysis Dashboard")

try:
    df = pd.read_csv("sample_reviews.csv")
    st.success("Loaded 'sample_reviews.csv' successfully.")
except FileNotFoundError:
    st.error("CSV file not found! Please upload or add 'sample_reviews.csv' to project folder.")
    st.stop()

# Sentiment function..
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# create sentiment column ..
df["Sentiment"] = df["Text"].apply(get_sentiment)

with st.expander("📄 Show Raw Data"):
    st.dataframe(df)

st.subheader("🔢 Sentiment Distribution")
sentiment_counts = df["Sentiment"].value_counts()
st.bar_chart(sentiment_counts)

st.subheader("📊 Sentiment Pie Chart")
fig1, ax1 = plt.subplots()
ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
ax1.axis("equal")
st.pyplot(fig1)

# Word clouds..
st.subheader("☁️ Word Clouds by Sentiment")
for sentiment in ["Positive", "Neutral", "Negative"]:
    texts = " ".join(df[df["Sentiment"] == sentiment]["Text"])
    if texts.strip():
        wc = WordCloud(background_color="white", width=800, height=400).generate(texts)
        st.markdown(f"**{sentiment}**")
        st.image(wc.to_array(), use_column_width=True)

# User Input..
st.subheader("📝 Real-time Sentiment Checker")
user_input = st.text_input("Type a sentence to analyze:")
if user_input:
    user_sentiment = get_sentiment(user_input)
    st.write(f"Sentiment: **{user_sentiment}**")

# If user want to upload our own manual file...
st.subheader("📂 Upload a Custom Dataset (CSV)")
uploaded_file = st.file_uploader("Upload a CSV file with a 'Text' column", type="csv")
if uploaded_file is not None:
    user_df = pd.read_csv(uploaded_file)
    if "Text" in user_df.columns:
        user_df["Sentiment"] = user_df["Text"].apply(get_sentiment)
        st.dataframe(user_df)
    else:
        st.error("CSV must have a column named 'Text'")
