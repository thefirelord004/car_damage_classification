import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define the base directory containing Cars and Damaged_Cars
base_dir = "C:\\Users\\prana\\Documents\\car_damage\\damage_is"

# Check the number of images in each class
print(f"Number of non-damaged cars: {len(os.listdir(os.path.join(base_dir, 'Car')))}")
print(f"Number of damaged cars: {len(os.listdir(os.path.join(base_dir, 'Car_Damage')))}")

# Create an ImageDataGenerator for loading images
datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values

# Create generators for training data
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150, 150),  # Resize images to 150x150
    batch_size=32,
    class_mode='binary'       # Binary classification
)

print(f"Training samples: {train_generator.samples}, Classes: {train_generator.class_indices}")

# Model Creation
damage_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile Model
damage_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
print("Starting model training...")
damage_model.fit(train_generator, epochs=10)

# Save the model
damage_model.save("car_damage_model.h5")
print("Model saved as car_damage_model.h5")
