
import pickle
import streamlit as st
import re
import nltk

from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


st.title("📩 Spam Message Classifier")
st.write("Enter a message below to check whether it is spam or not.")

message = st.text_area("Enter your message:")

if st.button("Check Message"):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        cleaned_message = clean_text(message)
        message_tfidf = vectorizer.transform([cleaned_message])
        prediction = model.predict(message_tfidf)

        if prediction[0] == 1:
            st.error("⚠️ This message is SPAM!")
        else:
            st.success("✅ This message is NOT SPAM.")
