
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- 1. LOAD DATA AND TRAIN A SIMPLE CNN ---
# This part is a condensed version of the previous lab (M03-12-4)

print("Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values and one-hot encode labels
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train_categorical = to_categorical(y_train, 10)
y_test_categorical = to_categorical(y_test, 10)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Build a simple CNN (for speed in this lab)
def create_simple_cnn():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

print("Creating and training a simple CNN...")
model = create_simple_cnn()
# Using fewer epochs as the focus is on metrics, not model performance
model.fit(x_train, y_train_categorical, epochs=5, batch_size=64, validation_split=0.1, verbose=2)

print("--- Evaluating Model Performance with Scikit-learn ---")

# --- 2. GET PREDICTIONS ---
# Get predicted probabilities
y_pred_probs = model.predict(x_test)
# Get predicted class labels by finding the index with the highest probability
y_pred_classes = np.argmax(y_pred_probs, axis=1)
# True labels are in y_test, which is shape (10000, 1). We flatten it.
y_true = y_test.flatten()

# --- 3. PART 1: CONFUSION MATRIX ---
print("Generating Confusion Matrix...")
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()
print("Confusion matrix plot saved to confusion_matrix.png")

# --- 4. PART 2: CLASSIFICATION REPORT ---
print("Generating Classification Report...")
report = classification_report(y_true, y_pred_classes, target_names=class_names)
print(report)

# --- 5. PART 3: ROC CURVE and AUC ---
# We will do this for a single class, e.g., 'cat' (class index 3)
# This is a One-vs-Rest ROC curve

CAT_CLASS_INDEX = 3

# Get the probabilities for the positive class ('cat')
y_pred_cat_prob = y_pred_probs[:, CAT_CLASS_INDEX]

# Create binary true labels: 1 if 'cat', 0 otherwise
y_true_cat = (y_true == CAT_CLASS_INDEX).astype(int)

# Calculate ROC curve and AUC score
print(f"Generating ROC Curve for class: '{class_names[CAT_CLASS_INDEX]}'...")
fpr, tpr, thresholds = roc_curve(y_true_cat, y_pred_cat_prob)
auc_score = roc_auc_score(y_true_cat, y_pred_cat_prob)

print(f"AUC Score for '{class_names[CAT_CLASS_INDEX]}' vs. Rest: {auc_score:.4f}")

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f"Receiver Operating Characteristic (ROC) for '{class_names[CAT_CLASS_INDEX]}' vs. Rest")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()
print(f"ROC curve plot for class '{class_names[CAT_CLASS_INDEX]}' saved to roc_curve_cat.png")
