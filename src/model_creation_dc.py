import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Function to load images from a directory
def load_images_from_directory(base_dir):
    data = []
    labels = []
    shapes = ['line', 'circle', 'ellipse', 'rectangle', 'rounded_rectangle', 'regular_polygon', 'star']
    
    for shape in shapes:
        shape_dir = os.path.join(base_dir, shape)
        for filename in os.listdir(shape_dir):
            img_path = os.path.join(shape_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                data.append(img)
                labels.append(shape)
    
    return np.array(data), np.array(labels)

# Load dataset
data_dir = '../dataset_created'  # Path to the generated dataset
data, labels = load_images_from_directory(data_dir)

# Preprocess data
data = data.reshape((data.shape[0], 128, 128, 1)) / 255.0
label_encoder = LabelEncoder()
labels = to_categorical(label_encoder.fit_transform(labels))

# Split data
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(data_train, labels_train, epochs=20, validation_split=0.2, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(data_test, labels_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save('./models/shapes_created_model.h5')