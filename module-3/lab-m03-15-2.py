
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from pathlib import Path

# --- 1. DATA LOADING AND PREPROCESSING ---

def load_and_preprocess_data():
    """Loads and preprocesses the CIFAR-10 dataset."""
    print("Loading and preprocessing CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot encode labels
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    
    return (x_train, y_train_cat), (x_test, y_test_cat), y_test.flatten()

# --- 2. BASELINE MODEL DEFINITION ---

def create_baseline_model(input_shape=(32, 32, 3), num_classes=10):
    """Creates a simple baseline CNN for CIFAR-10."""
    print("Creating the baseline CNN architecture...")
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Classifier Head
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    print("Compiling the model...")
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Main Execution Block ---

if __name__ == "__main__":
    print("--- Project Session 2: Building a Baseline Model ---")
    
    # Define constants
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    RESULTS_DIR = "baseline_results"
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    # 1. Load Data
    (x_train, y_train_cat), (x_test, y_test_cat), y_true_flat = load_and_preprocess_data()
    
    # 2. Create Model
    model = create_baseline_model()
    model.summary()
    
    # 3. Train Model
    print("--- Training the Baseline Model ---")
    history = model.fit(x_train, y_train_cat,
                        epochs=20,
                        batch_size=64,
                        validation_data=(x_test, y_test_cat),
                        verbose=2)
    
    print("--- Evaluating the Baseline Model ---")
    
    # 4. Evaluate on Test Set
    loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"Baseline Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Baseline Test Loss: {loss:.4f}")

    # 5. Generate Predictions for Metrics
    y_pred_probs = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    # 6. Confusion Matrix
    print("Generating and saving Confusion Matrix...")
    cm = confusion_matrix(y_true_flat, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Baseline Model - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = os.path.join(RESULTS_DIR, "baseline_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Saved to {cm_path}")
    plt.close()

    # 7. Classification Report
    print("Generating and saving Classification Report...")
    report = classification_report(y_true_flat, y_pred_classes, target_names=class_names)
    report_path = os.path.join(RESULTS_DIR, "baseline_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Baseline Model Classification Report")
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%")
        f.write(f"Test Loss: {loss:.4f}")
        f.write(report)
    print(report)
    print(f"Saved to {report_path}")

    # 8. Save the trained model
    model_path = os.path.join(RESULTS_DIR, "baseline_model.h5")
    print(f"Saving baseline model to {model_path}...")
    model.save(model_path)
    print("Model saved successfully.")

    # 9. Plot and save training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Baseline: Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Baseline: Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    bth_path = os.path.join(RESULTS_DIR, "baseline_training_history.png")
    plt.savefig(bth_path)
    print(f"Training history plot saved to {bth_path}")

    print("Baseline established. Ready for improvement experiments.")
