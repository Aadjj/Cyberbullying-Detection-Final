import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib
import easyocr

svm_model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')


def classify_text(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    return svm_model.predict(text_tfidf)[0]


def extract_text(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    text = ' '.join([entry[1] for entry in result])
    return text


def process_image(image):
    image_bytes = image.read()

    text = extract_text(image_bytes)

    classification = classify_text(text)

    if classification == 1:
        st.error("Offensive Content Detected!")
    else:
        st.success("No Offensive Content Detected!")


st.set_page_config(page_title="Hate Speech Detection", page_icon=":shield:", layout="wide")
st.title('Hate Speech Detection')
st.write("A free to use website for Hate Speech and Offensiveness Detection.")
st.markdown("---")

st.sidebar.title("Hate Speech Detection")
option = st.sidebar.selectbox(
    'Select Input Type',
    ('Text', 'Image')
)

if option == 'Text':
    st.subheader("Enter Text:")
    input_text = st.text_area("", "", height=150)
    if st.button("Predict"):
        with st.spinner('Predicting...'):
            classification = classify_text(input_text)
            if classification == 1:
                st.error("Offensive Content Detected!")
            else:
                st.success("No Offensive Content Detected!")

elif option == 'Image':
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with st.spinner('Processing image...'):
            process_image(uploaded_file)

st.markdown("---")
st.write("Developed by Syed Adnan Ahmed,"
         " Aitham Meghana,"
         " Syed Talal Amjad.")
st.write(" For any inquiries or feedback, please contact us at aadjj41@gmail.com.")
