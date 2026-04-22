
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np

# --- 1. PROJECT SETUP AND DATA LOADING ---

def load_dataset():
    """
    Loads the CIFAR-10 dataset from Keras and returns the training and test sets.
    """
    print("Attempting to load CIFAR-10 dataset...")
    try:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print("CIFAR-10 dataset loaded successfully.")
        return (x_train, y_train), (x_test, y_test)
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None, None

def get_class_names():
    """Returns the list of class names for the CIFAR-10 dataset."""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# --- Main Execution Block ---

if __name__ == "__main__":
    print("--- Project Kickoff: End-to-End Image Classifier ---")
    print("Selected Dataset: CIFAR-10")
    
    # Load the dataset
    (x_train, y_train), (x_test, y_test) = load_dataset()
    
    # Verify the data has been loaded correctly
    if x_train is not None:
        print("--- Dataset Verification ---")
        print(f"Shape of training images: {x_train.shape}") # Expected: (50000, 32, 32, 3)
        print(f"Shape of training labels: {y_train.shape}") # Expected: (50000, 1)
        print(f"Shape of test images: {x_test.shape}")   # Expected: (10000, 32, 32, 3)
        print(f"Shape of test labels: {y_test.shape}")   # Expected: (10000, 1)
        
        # Check data types and value ranges
        print(f"Data type of images: {x_train.dtype}")
        print(f"Min pixel value: {np.min(x_train)}")
        print(f"Max pixel value: {np.max(x_train)}")
        
        class_names = get_class_names()
        print(f"Number of classes: {len(class_names)}")
        print(f"Classes: {class_names}")
        
        print("Project environment is set up and data is loaded.")
        print("Ready to proceed to building the baseline model in the next session.")
    else:
        print("Data loading failed. Please check your internet connection or Keras configuration.")

