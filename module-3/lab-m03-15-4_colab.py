
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import drive

# --- Google Drive Mounting ---
drive.mount('/content/drive')

# --- Function to load images and extract features (unchanged) ---
def load_images_and_extract_features(folder, label):
    images = []
    labels = []
    features = []
    print(f"Loading images from: {folder}")
    if not os.path.isdir(folder):
        print(f"Error: Directory not found at {folder}")
        return [], [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, (64, 64))
            feature_vector = img_resized.flatten()
            images.append(img_resized)
            labels.append(label)
            features.append(feature_vector)
    print(f"Found and processed {len(features)} images.")
    return images, labels, features

# --- IMPORTANT --- #
# Please upload your 'positive_images' and 'negative_images' folders to this Google Drive directory.
# DRIVE_DATA_DIR = '/content/drive/My Drive/colab_data/inria_person'

# --- Load Data ---
# positive_folder = os.path.join(DRIVE_DATA_DIR, 'positive_images')
# negative_folder = os.path.join(DRIVE_DATA_DIR, 'negative_images')
# For local execution, ensure 'positive_images' and 'negative_images' are in the same directory as the script
positive_folder = 'positive_images'
negative_folder = 'negative_images'

pos_images, pos_labels, pos_features = load_images_and_extract_features(positive_folder, 1)
neg_images, neg_labels, neg_features = load_images_and_extract_features(negative_folder, 0)

# Check if images were loaded
if not pos_features or not neg_features:
    print("Critical Error: No features were loaded. Please check the folder paths and content.")
else:
    all_features = np.array(pos_features + neg_features)
    all_labels = np.array(pos_labels + neg_labels)

    # --- Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, all_labels, test_size=0.25, random_state=42, stratify=all_labels
    )

    # --- Hyperparameter Tuning for SVM using GridSearchCV ---
    print("Starting hyperparameter tuning for SVM with GridSearchCV...")

    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }

    grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=3)
    grid_search.fit(X_train, y_train)

    # --- Best Model Evaluation ---
    print("Hyperparameter tuning complete.")
    print(f"Best parameters found: {grid_search.best_params_}")
    best_svm = grid_search.best_estimator_

    y_pred = best_svm.predict(X_test)

    # --- Final Evaluation ---
    print("--- Final Evaluation of the Tuned SVM Model ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    # --- Visualize the Confusion Matrix ---
    print("Generating confusion matrix for the best SVM model...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix for Tuned SVM')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')

    # Save the plot to Google Drive (saving is disabled as per user request)
    # DRIVE_OUTPUT_DIR = '/content/drive/My Drive/colab_output/session4'
    # os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
    # output_path = os.path.join(DRIVE_OUTPUT_DIR, 'confusion_matrix_svm.png')
    # plt.savefig(output_path)
    # print(f"Confusion matrix saved to {output_path}")
    plt.show() 

    print("Lecture complete. We have successfully trained, tuned, and evaluated a Support Vector Machine classifier.")
