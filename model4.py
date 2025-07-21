import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2

# Paths to dataset directories
train_dir = "E:\\car_damage\\damage_locate\\train"
val_dir = "E:\\car_damage\\damage_locate\\val"
test_dir = "E:\\car_damage\\damage_locate\\test"

# Load annotations
def load_annotations(json_file):
    with open(json_file) as f:
        return json.load(f)

# Prepare the dataset
def load_data(images_dir, annotations_json):
    annotations = load_annotations(annotations_json)
    images = []
    labels = []

    for img_info in annotations['images']:
        img_id = img_info['id']
        img_path = os.path.join(images_dir, img_info['file_name'])
        
        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to load image at {img_path}. Skipping.")
            continue  # Skip this image if it cannot be loaded

        img = cv2.resize(img, (150, 150))  # Resize to your input size
        images.append(img)

        # Create a label array for the parts
        label_array = np.zeros(5)  # Assuming 5 categories
        for ann in annotations['annotations']:
            if ann['image_id'] == img_id:
                part_id = ann['category_id']  # Adjust if needed
                label_array[part_id - 1] = 1  # Assuming category ids are 1-indexed
        labels.append(label_array)

    # Convert lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Ensure consistent input shapes
    if len(images) == 0 or len(labels) == 0:
        raise ValueError("No valid images/labels found.")

    return images, labels

# Load training and validation data
X_train, y_train = load_data(val_dir, os.path.join(val_dir, "COCO_mul_val_annos.json"))
X_val, y_val = load_data(val_dir, os.path.join(val_dir, "COCO_mul_val_annos.json"))

# Normalize the images
X_train = X_train / 255.0
X_val = X_val / 255.0

# Print the shapes of the loaded datasets
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# Define the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(5, activation='sigmoid')  # Output layer for 5 categories
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save the model
model.save("car_damage_location_model.h5")
print("Model saved as car_damage_location_model.h5")

# Evaluation on test dataset
X_test, y_test = load_data(val_dir, os.path.join(val_dir, "COCO_mul_val_annos.json"))  # Update with your actual test annotations
X_test = X_test / 255.0

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.2f}")
