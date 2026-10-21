"""Campus Comment Sentiment Checker: the in-class demo app.

Run with:  streamlit run app.py
"""

import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Campus Comment Sentiment", page_icon="💬")

st.title("Campus Comment Sentiment Checker")
st.write(
    "Type a comment about campus life. A pre-trained Hugging Face model "
    "will guess whether the writer feels positive or negative."
)


# Streamlit re-runs this whole file every time you click something.
# @st.cache_resource makes the model load ONE time and then reuse it.
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )


classifier = load_model()

text = st.text_area(
    "Enter a comment:",
    "The mac and cheese in the dining hall was amazing today!",
)

if st.button("Analyze"):
    if not text.strip():
        st.warning("Please type a comment first.")
    else:
        result = classifier(text)[0]          # a dict: {"label": ..., "score": ...}
        label = result["label"].lower()
        score = result["score"]

        if label == "positive":
            st.success(f"Positive ({score:.1%} confident)")
        else:
            st.error(f"Negative ({score:.1%} confident)")
        st.progress(score)

        with st.expander("See the raw model output"):
            st.json(result)

        st.caption("A confident score is not the same as a correct answer.")
