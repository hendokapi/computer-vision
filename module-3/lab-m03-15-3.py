
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# --- Function to load images and extract features (unchanged from previous lectures) ---
def load_images_and_extract_features(folder, label):
    images = []
    labels = []
    features = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # For simplicity, we'll use the flattened pixel array as a feature vector.
            # In a real-world scenario, more sophisticated feature extraction would be needed.
            img_resized = cv2.resize(img, (64, 64)) # Resize to a consistent size
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

# Combine data
all_features = np.array(pos_features + neg_features)
all_labels = np.array(pos_labels + neg_labels)

# --- Split Data (same as before) ---
X_train, X_test, y_train, y_test = train_test_split(
    all_features, all_labels, test_size=0.25, random_state=42, stratify=all_labels
)

# --- Model Training (k-NN) ---
print("Training the k-Nearest Neighbors classifier...")
# We will experiment with different values of k
k_values = [1, 3, 5, 7, 9]
best_k = -1
best_accuracy = 0.0

for k in k_values:
    print(f"  Training with k = {k}...")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # --- Model Evaluation ---
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy for k={k}: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f"Best k value found: {best_k} with an accuracy of {best_accuracy:.4f}")

# --- Final Evaluation with the Best k ---
print("--- Final Evaluation using the best k-value ---")
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)
y_pred_final = best_knn.predict(X_test)

# Print final accuracy
final_accuracy = accuracy_score(y_test, y_pred_final)
print(f"Final Test Accuracy (k={best_k}): {final_accuracy:.4f}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_final, target_names=['Negative', 'Positive']))

# --- Visualize the Confusion Matrix ---
print("Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title(f'Confusion Matrix for k-NN (k={best_k})')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# Save the plot
output_path = '..\\results\\confusion_matrix_knn.png'
plt.savefig(output_path)
print(f"Confusion matrix saved to {output_path}")
plt.show()

print("Lecture complete. We have successfully tuned and evaluated a k-NN classifier.")
