
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# --- 1. LOAD AND PREPARE THE DATA ---
print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Define the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Preprocessing
# Normalize pixel values from [0, 255] to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the labels
# Example: Label 5 becomes [0,0,0,0,0,1,0,0,0,0]
y_train_categorical = to_categorical(y_train, 10)
y_test_categorical = to_categorical(y_test, 10)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")
print(f"Training labels shape (one-hot): {y_train_categorical.shape}")

# --- 2. DEFINE THE CNN ARCHITECTURE ---
print("Building the CNN model...")

model = Sequential()

# Block 1: First Convolutional and Pooling layer
# 32 filters, 3x3 kernel size, relu activation
# Input shape is 32x32 pixels with 3 color channels (RGB)
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
# Optional: Add Dropout for regularization
model.add(Dropout(0.25))

# Block 2: Second Convolutional and Pooling layer
# 64 filters, 3x3 kernel
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Block 3: Third Convolutional and Pooling layer
# 128 filters, 3x3 kernel
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Flatten the 3D feature maps into a 1D vector
model.add(Flatten())

# Dense (fully-connected) layers for classification
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Output layer with 10 neurons (one for each class) and softmax activation
model.add(Dense(10, activation='softmax'))

# Print a summary of the model architecture
model.summary()

# --- 3. COMPILE THE MODEL ---
print("Compiling the model...")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- 4. TRAIN THE MODEL ---
print("Training the model...")
# For a real run, use more epochs (e.g., 25-50). For a quick lab, 10 is enough to see progress.
EPOCHS = 15 
BATCH_SIZE = 64

history = model.fit(x_train, y_train_categorical, 
                    epochs=EPOCHS, 
                    batch_size=BATCH_SIZE,
                    validation_data=(x_test, y_test_categorical))

print("Model training complete.")

# --- 5. EVALUATE THE MODEL ---
print("Evaluating the model on the test set...")
test_loss, test_acc = model.evaluate(x_test, y_test_categorical, verbose=2)
print(f"Test Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# --- 6. VISUALIZE TRAINING HISTORY & PREDICTIONS ---

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()
print("Training history plot saved to cnn_training_history.png")

# Make predictions on a few test images
predictions = model.predict(x_test)

# Show 5 random images from the test set, their predicted labels, and the true labels
plt.figure(figsize=(15, 7))
for i in range(5):
    idx = np.random.randint(0, x_test.shape[0])
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[idx])
    plt.xticks([])
    plt.yticks([])
    predicted_label_index = np.argmax(predictions[idx])
    true_label_index = y_test[idx][0]
    
    predicted_label = class_names[predicted_label_index]
    true_label = class_names[true_label_index]
    
    color = 'green' if predicted_label == true_label else 'red'
    
    plt.xlabel(f"Pred: {predicted_label} True: {true_label}", color=color)

plt.tight_layout()
plt.show()
print("Prediction examples plot saved to cnn_predictions.png")
