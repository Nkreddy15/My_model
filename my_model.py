import tensorflow as tf
from tensorflow.keras import layers, models

# Create a simple model (example for demonstration)
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),  # Example input shape (e.g., for MNIST images)
    layers.Dense(10, activation='softmax')  # Example output layer (e.g., for 10 classes)
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Example: Dummy data for illustration (use actual training data for a real scenario)
import numpy as np
x_train = np.random.random((100, 784))  # 100 samples, 784 features (e.g., flattened 28x28 images)
y_train = np.random.randint(10, size=(100,))  # 100 labels for 10 classes

# Train the model (use your own dataset if available)
model.fit(x_train, y_train, epochs=5)
