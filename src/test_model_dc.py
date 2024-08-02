import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the model
model = load_model('../models/shapes_created_model.h5')

# Load label encoder
# Assuming you saved and reloaded the label encoder separately
# If you didn’t save it, you need to recreate it with the same classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['line', 'circle', 'ellipse', 'rectangle', 'rounded_rectangle', 'regular_polygon', 'star'])

# Function to preprocess images
def preprocess_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        img = cv2.resize(img, target_size)
        img = img.reshape((1, 128, 128, 1)) / 255.0
    return img

# Function to predict image class
def predict_image(model, image_path, label_encoder):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_class[0]

# Function to display image with prediction
def display_image_with_prediction(image_path, predicted_class):
    img = cv2.imread(image_path)
    cv2.imshow(f'{predicted_class}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test with user-provided images
user_image_path = '../problems/occlusion2_sol_rec.png'  # Replace with the path to the user image
predicted_class = predict_image(model, user_image_path, label_encoder)
print(f'Predicted Class: {predicted_class}')
display_image_with_prediction(user_image_path, predicted_class)