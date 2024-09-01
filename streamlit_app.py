import streamlit as st
import os
from predictor import WheatDiseasePredictor
from PIL import Image

# Configure the file upload directory
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize the predictor
predictor = WheatDiseasePredictor(
    csv_path='data/class_dict.csv',
    model_path='models/model.h5'
)

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(uploaded_file):
    filename = uploaded_file.name
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# Streamlit app
st.title('Wheat Disease Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=ALLOWED_EXTENSIONS)

if uploaded_file is not None:
    # Save the file
    file_path = save_uploaded_file(uploaded_file)

    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Make a prediction
    class_name, probability = predictor.predict(file_path)
    
    # Display the prediction
    if class_name:
        st.success(f'Prediction: {class_name}')
        st.info(f'Probability: {probability * 100:.2f}%')
    else:
        st.error('Unable to classify image')
