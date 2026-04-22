
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# --- Function to load images and extract features (unchanged) ---
def load_images_and_extract_features(folder, label):
    images = []
    labels = []
    features = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, (64, 64))
            feature_vector = img_resized.flatten()
            images.append(img_resized)
            labels.append(label)
            features.append(feature_vector)
    return images, labels, features

# --- Load Data (same as before) ---
positive_folder = 'positive_images'
negative_folder = 'negative_images'

pos_images, pos_labels, pos_features = load_images_and_extract_features(positive_folder, 1)
neg_images, neg_labels, neg_features = load_images_and_extract_features(negative_folder, 0)

all_features = np.array(pos_features + neg_features)
all_labels = np.array(pos_labels + neg_labels)

# --- Split Data (same as before) ---
X_train, X_test, y_train, y_test = train_test_split(
    all_features, all_labels, test_size=0.25, random_state=42, stratify=all_labels
)

# --- Introduction to SVM Hyperparameters ---
# C: The regularization parameter. It trades off correct classification of training
#    examples against maximization of the decision function's margin.
# gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
# kernel: Specifies the kernel type to be used in the algorithm.

# --- Hyperparameter Tuning for SVM using GridSearchCV ---
print("Starting hyperparameter tuning for SVM with GridSearchCV...")

# Define the parameter grid to search
# We will test a combination of C values, gamma values, and kernels.
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'gamma': [1, 0.1, 0.01, 0.001], # Kernel coefficient
    'kernel': ['rbf', 'linear'] # Kernel type
}

# Instantiate GridSearchCV
# cv=3 means 3-fold cross-validation.
# verbose=2 provides more detailed output during the process.
grid_search = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, cv=3)

# Fit the model to the data to find the best parameters
grid_search.fit(X_train, y_train)

# --- Best Model Evaluation ---
print("Hyperparameter tuning complete.")

# Print the best parameters found by GridSearchCV
print(f"Best parameters found: {grid_search.best_params_}")
best_svm = grid_search.best_estimator_

# Make predictions on the test set using the best found model
y_pred = best_svm.predict(X_test)

# --- Final Evaluation ---
print("--- Final Evaluation of the Tuned SVM Model ---")

# Print final accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# --- Visualize the Confusion Matrix ---
print("Generating confusion matrix for the best SVM model...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title(f'Confusion Matrix for Tuned SVM')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# Save the plot
output_path = '..\\results\\confusion_matrix_svm.png'
plt.savefig(output_path)
print(f"Confusion matrix saved to {output_path}")
plt.show()

print("Lecture complete. We have successfully trained, tuned, and evaluated a Support Vector Machine classifier.")
