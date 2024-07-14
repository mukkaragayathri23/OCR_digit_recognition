''' DIGIT RECOGNIZER '''

# Imports
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('./models/WideResNet28_10.h5')
# Image Processing & Prediction
def recognize_digit(image_path):
    # Load the Keras model for digit recognition
    

    # Load the image
    image = cv2.imread(image_path)



    # Preprocess the image
    # Convert to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to 28x28 pixels
    image_resized = cv2.resize(image_gray, (28, 28))

    # Invert the image (if needed)
    image_processed = cv2.bitwise_not(image_resized)
    num_black_pixels = (image_processed > 200).sum()
    if num_black_pixels>10:

        # Normalize the pixel values to range [0, 1]
        image_normalized = image_processed / 255.0

        # Reshape the image to match model input shape (batch_size, height, width, channels)
        input_image = np.expand_dims(image_normalized, axis=0)
        input_image = np.expand_dims(input_image, axis=-1)

        # Predict the digit using the loaded model
        predictions = model.predict(input_image)

        # Get the predicted digit (index of maximum probability)
        predicted_digit = np.argmax(predictions)
    else:
        predicted_digit = 'NA'

    return predicted_digit
