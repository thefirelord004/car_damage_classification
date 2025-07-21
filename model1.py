import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Define the paths to the training and validation directories
train_dir = 'car_detect'  # Main directory containing Bikes and Cars

# Data Preparation
train_datagen = ImageDataGenerator(rescale=1./255, 
                                    shear_range=0.2, 
                                    zoom_range=0.2, 
                                    horizontal_flip=True,
                                    validation_split=0.2)  # Split data into training and validation sets

# Create generators for training and validation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),  # Resize images to 150x150
    batch_size=32,
    class_mode='binary',      # Binary classification
    subset='training'         # Set as training data
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',      # Binary classification
    subset='validation'       # Set as validation data
)

# Model Creation
model = models.Sequential([
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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the model
model.save("car_bike_classification_model.h5")
