
import base64
from flask import Flask, request, render_template
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image, ImageDraw

app = Flask(__name__)

# Load your trained model
model = load_model('target_detection_model.h5')

# Load the class labels (if needed)
classes = ['2S1', 'BRDM_2', 'BTR_60','D7','SLICY','T62','ZIL131','ZSU_23_4']  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"

    if file:
        # Read the uploaded image
        img = Image.open(file)
        img = img.convert('L')  # Convert to grayscale if necessary
        img = img.resize((128, 128))  # Resize image to match model input size
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        
        # Perform prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        accuracy = predictions[0][predicted_class]
        
        # Mark the detected target with a red square (example: center of the image)
        draw = ImageDraw.Draw(img)
        center = (64, 64)
        box_size = 20
        draw.rectangle(
            [(center[0] - box_size, center[1] - box_size), 
             (center[0] + box_size, center[1] + box_size)], 
            outline="white", width=3)

        # Encode the image in base64
        img_io = BytesIO()
        img.save(img_io, 'JPEG')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('ascii')

        # Return the image and prediction results to the webpage
        return render_template(
            'result.html', 
            predicted_class=classes[predicted_class], 
            accuracy=accuracy,
            img_data=img_base64
        )

if __name__ == "__main__":
    app.run(debug=True)
