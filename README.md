# Wheat Disease Classifier

## Overview

The Wheat Disease Classifier is a web application that uses a deep learning model to classify wheat leaf images based on their health status. This project leverages TensorFlow for model inference, OpenCV for image processing, and Flask for building a user-friendly web interface. The classifier can detect and predict the disease present in wheat leaves, providing insights into their condition.

## Features

- **Image Upload**: Users can upload an image of a wheat leaf to the web application.
- **Disease Prediction**: The model classifies the leaf image and returns the disease type along with a probability score.
- **Result Display**: The web interface displays the prediction results and the uploaded image.

## Technologies Used

- **Flask**: Web framework for building the application.
- **TensorFlow**: Deep learning framework for loading and running the disease classification model.
- **OpenCV-Python**: Library for image processing tasks.
- **Pandas**: Data manipulation library for handling class information.
- **NumPy**: Library for numerical operations.


## Installation

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/wheat_disease_classifier.git
   cd wheat_disease_classifier
   ```
2. **Create a Virtual Environment (optional but recommended)**:
    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```
3. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```
4. **Download or Place Model and Data Files**:
- Ensure that you have the model.h5 file in the models/ directory.
- Place the class_dict.csv file in the data/ directory.

## Running the Application

1. **Start the Flask Server**:
    ```sh
    python3 app.py
    ```
2. **Access the Web Interface**:
- Open your web browser and navigate to http://127.0.0.1:5000/.

3. **Upload an Image**:
- Use the file upload button to select an image of a wheat leaf.
- Click on "Upload and Predict" to submit the image for classification.

4. **View Results**:
- The result will be displayed on the web page, showing the predicted disease class, probability, and the uploaded image.

## File Upload Restrictions
- Allowed File Types: JPEG, PNG
- Maximum File Size: 5MB

## Model Test Results
- Overall Accuracy: 98.96% on the test set
  
![Training Loss and Accuracy at each Epoch]([http://url/to/img.png](https://github.com/obaidhsn/wheat_leaf_disease_detection/blob/main/static/train_loss_acc.png))

![Confusion Matrix]([http://url/to/img.png](https://github.com/obaidhsn/wheat_leaf_disease_detection/blob/main/static/conf_matrix.png))

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/obaidhsn/wheat_leaf_disease_detection/blob/main/LICENSE) file for details.

