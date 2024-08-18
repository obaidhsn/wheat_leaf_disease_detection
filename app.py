from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from predictor import WheatDiseasePredictor
import mimetypes

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Initialize the predictor
predictor = WheatDiseasePredictor(
    csv_path='data/class_dict.csv',
    model_path='models/model.h5'
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_mime_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Check MIME type
        mime_type = get_mime_type(file_path)
        if mime_type not in ['image/jpeg', 'image/png']:
            return jsonify({'error': 'Invalid file type'})
        
        class_name, probability = predictor.predict(file_path)
        if class_name:
            return jsonify({'class': class_name, 'probability': f'{probability * 100:.2f}%', 'image': file_path})
        else:
            return jsonify({'error': 'Unable to classify image'})
    else:
        return jsonify({'error': 'Invalid file type'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
