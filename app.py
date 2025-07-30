import streamlit as st
import pickle
import nltk
print(nltk.__version__)
nltk.download('punkt')
print("Punkt resource downloaded")
nltk.download('stopwords') 
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.set_page_config(page_title="Spam Message Classifier", page_icon="📧", layout="centered")
st.title("📩 Spam Message Classifier")

input_sms = st.text_area("Enter the message")
if st.button('Predict'):
    with st.spinner("Analyzing message..."):
        # Dummy processing time (simulate delay)
        import time
        time.sleep(1.5)
    #preprocess
    transformed_sms = transform_text(input_sms)

    #vectorize
    vector_input = tfidf.transform([transformed_sms])

    #predcit
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("🚨 This looks like spam!")
    else:
        st.header("✅ Safe! No spam detected.")
st.markdown("---")
st.info("⚠️ Note: Our spam detection model aims for accuracy but no model can guarantee 100% correctness. Always review messages if unsure.")

with st.sidebar:
    st.title("ℹ️ About")
    st.write("A fun little ML-powered spam detector built with Streamlit.")
