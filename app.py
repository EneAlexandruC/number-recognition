import numpy as np
from flask import Flask, render_template, request, jsonify          # The web app framework
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import base64                                                       # Decode images sent as base64 strings
import re                                                           # Regular expressions to clean up the base64 string
from io import BytesIO                                              # Treat bytes like a file object for PIL
from PIL import Image                                               # Load, manipulate, and resize images

app = Flask(__name__)                                               
model = load_model('digit_model.h5')

# Home Page

# When user visits /, Flask serves index.html (from templates folder)
@app.route('/')
def index():
    return render_template('index.html')

# Prediction Endpoint

# Receive the image drawn by the user and return the predicted digit
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()                                                   # Reads the JSON body sent from the frontend

    image_data = re.sub('^data:image/.+;base64,', '', data['image'])            # Removes redundant data and leaves only the raw base64 data
    image = Image.open(BytesIO(base64.b64decode(image_data))).convert('L')      # Converts to grayscale ('L')
    image = image.resize((28,28))
    image = np.array(image)                                                     # Converts PIL image to NumPy array
    image = 255 - image                                                         # Invert colors (black background)
    image = image / 255.0
    image = image.reshape(-1, 28, 28, 1)                                        # Reshape the image because the model expects batch dimension

    # Pass the preprocessed imagage into the model and returns a 1x10 array of probabilities for digits 0-9
    prediction = model.predict(image)
    digit = np.argmax(prediction)

    return jsonify({'digit': int(digit)})                                   # Send back a JSON with the predicted digit

# Running the app

if __name__ == '__main__':
    app.run(debug=True)


