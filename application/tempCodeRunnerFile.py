from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('final_plant_disease_model 22.h5')

# Define the image size (ensure it matches the size used during training)
image_size = (128, 128)

# Class labels
class_labels = [
    'Apple Scab Disease',
    'Black Rot Disease',
    'Cedar Apple rust Disease',
    'Healthy Apple Leaf',
    'Healthy BlueBerry',
    'Cherry (including sour) - Healthy',
    'Cherry (including sour) - Powdery mildew',
    'Corn (maize) - Cercospora leaf spot Gray leaf spot',
    'Corn (maize) - Common rust',
    'Healthy Corn Maize',
    'Corn Northern Leaf Blind',
    'Grape Black rot',
    'Grape Esca',
    'Healthy Grape',
    'Grape Leaf Blind',
    'Orange Huanglongbing (Citrus greening)',
    'Peach Bacterial spot',
    'Healthy Peach',
    'Peach Bell Bacterial',
    'Peach Bell Healthy',
    'Potato Early Blight',
    'Potato Late Blight'
]

# Function to preprocess the uploaded image before prediction
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure the image is in RGB mode
    image = image.resize(image_size)  # Resize the image
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make a prediction on the uploaded image
def predict_image(image):
    processed_image = preprocess_image(image)  # Preprocess the image
    predictions = model.predict(processed_image)  # Make predictions
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get the index of the highest prediction
    predicted_class_label = class_labels[predicted_class_index]  # Get the corresponding class label
    confidence_score = np.max(predictions) * 100  # Get the confidence score in percentage
    return predicted_class_label, confidence_score

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle requests to the main page."""
    prediction, probability = None, None

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            try:
                image = Image.open(file)
                prediction, probability = predict_image(image)
            except Exception as e:
                print(f"Error: {e}")

    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)
