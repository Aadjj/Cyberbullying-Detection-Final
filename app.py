import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib
import easyocr

# Load the trained SVM model and TF-IDF vectorizer
svm_model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')


# Function to classify text as offensive or non-offensive
def classify_text(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    return svm_model.predict(text_tfidf)[0]


# Function to extract text from an image using EasyOCR
def extract_text(image):
    reader = easyocr.Reader(['en'])  # Specify language (e.g., English)
    result = reader.readtext(image)
    text = ' '.join([entry[1] for entry in result])
    return text


# Function to process an image and display the result
def process_image(image):
    # Convert BytesIO object to bytes array
    image_bytes = image.read()

    # Extract text from the image using EasyOCR
    text = extract_text(image_bytes)

    # Classify the extracted text
    classification = classify_text(text)

    # Display the result
    if classification == 1:
        st.error("Offensive Content Detected!")
    else:
        st.success("No Offensive Content Detected!")


# Streamlit UI
st.set_page_config(page_title="Cyberbullying Detection", page_icon=":shield:", layout="wide")
st.title('Cyberbullying Detection')
st.write("A free to use website for Cyberbullying Detection.")
st.markdown("---")

# Sidebar
st.sidebar.title("Cyberbullying Detection")
option = st.sidebar.selectbox(
    'Select Input Type',
    ('Text', 'Image')
)

# Main content
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

# Footer
st.markdown("---")
st.write("Developed by Syed Adnan Ahmed,"
         " Aitham Meghana,"
         " Syed Talal Amjad."
         " For any inquiries or feedback, please contact us at aadjj41@gmail.com.")

