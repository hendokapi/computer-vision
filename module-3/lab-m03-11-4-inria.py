import os
import cv2
import numpy as np
from sklearn.utils import shuffle

# --- HOG FEATURE EXTRACTION ---
def extract_hog_features(image_paths, hog_descriptor, label):
    """
    Loads images from paths, computes HOG features, and returns them along with labels.
    """
    hog_features = []
    hog_labels = []
    print(f'Extracting HOG features for {label} samples...')
    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image at {image_path}")
            continue

        # Resize to the HOG descriptor's window size if not already matching
        if (img.shape[1], img.shape[0]) != hog_descriptor.winSize:
            img = cv2.resize(img, hog_descriptor.winSize)

        features = hog_descriptor.compute(img)
        hog_features.append(features)
        hog_labels.append(label)

    return np.array(hog_features, dtype=np.float32), np.array(hog_labels, dtype=np.int32)

# --- Main Execution ---

if __name__ == "__main__":
    # --- 1. DEFINE PATHS & PARAMETERS ---
    # you can download the dataset from https://hyper.ai/en/datasets/5331
    # ASSUMPTION: The script is run from a directory containing the 'inria_dataset' folder.
    # The INRIA dataset should be organized as follows:
    # ./inria_dataset/
    #   Train/
    #     pos/ ... (positive training images)
    #     neg/ ... (negative training images)
    #   Test/
    #     pos/ ... (positive testing images)
    #     neg/ ... (negative testing images)
    BASE_INRIA_DIR = "inria_dataset"
    TRAIN_POS_DIR = os.path.join(BASE_INRIA_DIR, "Train", "pos")
    TRAIN_NEG_DIR = os.path.join(BASE_INRIA_DIR, "Train", "neg")
    TEST_POS_DIR = os.path.join(BASE_INRIA_DIR, "Test", "pos")
    TEST_NEG_DIR = os.path.join(BASE_INRIA_DIR, "Test", "neg")
    SVM_MODEL_PATH = "svm_inria_pedestrian_detector.yml"

    # Check if dataset paths exist
    if not os.path.isdir(BASE_INRIA_DIR):
        print(f"Error: The dataset directory '{BASE_INRIA_DIR}' was not found.")
        print("Please download the INRIA Person Dataset and place it in the correct directory.")
        exit()

    # HOG Descriptor Parameters (standard for 64x128 pedestrian detection)
    winSize = (64, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    # --- 2. LOAD DATA AND EXTRACT HOG FEATURES ---
    # Load training data
    train_pos_paths = [os.path.join(TRAIN_POS_DIR, f) for f in os.listdir(TRAIN_POS_DIR)]
    train_neg_paths = [os.path.join(TRAIN_NEG_DIR, f) for f in os.listdir(TRAIN_NEG_DIR)]

    pos_hog_features, pos_hog_labels = extract_hog_features(train_pos_paths, hog, 1)
    neg_hog_features, neg_hog_labels = extract_hog_features(train_neg_paths, hog, -1)

    # Combine and shuffle training data
    train_features = np.concatenate((pos_hog_features, neg_hog_features), axis=0)
    train_labels = np.concatenate((pos_hog_labels, neg_hog_labels), axis=0)
    train_features, train_labels = shuffle(train_features, train_labels, random_state=42)

    # Reshape features for SVM
    num_samples, feature_len = train_features.shape
    train_features = train_features.reshape(num_samples, feature_len)

    print(f"Total training samples: {num_samples}")
    print(f"HOG feature vector shape: {train_features.shape}")
    print(f"Labels shape: {train_labels.shape}")

    # --- 3. TRAIN SVM CLASSIFIER ---
    print("Training SVM classifier...")

    # Create and configure the SVM
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-6))

    # Train the SVM
    svm.train(train_features, cv2.ml.ROW_SAMPLE, train_labels)
    print("SVM training complete.")

    # Save the trained model
    svm.save(SVM_MODEL_PATH)
    print(f"Trained SVM model saved to: {SVM_MODEL_PATH}")

    # --- 4. EVALUATE THE CLASSIFIER ---
    print("Evaluating classifier on test data...")

    # Load test data
    test_pos_paths = [os.path.join(TEST_POS_DIR, f) for f in os.listdir(TEST_POS_DIR)]
    test_neg_paths = [os.path.join(TEST_NEG_DIR, f) for f in os.listdir(TEST_NEG_DIR)]
    
    test_pos_features, test_pos_labels = extract_hog_features(test_pos_paths, hog, 1)
    test_neg_features, test_neg_labels = extract_hog_features(test_neg_paths, hog, -1)

    # Combine test data
    test_features = np.concatenate((test_pos_features, test_neg_features), axis=0)
    test_labels = np.concatenate((test_pos_labels, test_neg_labels), axis=0)

    # Reshape test features
    num_test_samples, test_feature_len = test_features.shape
    test_features = test_features.reshape(num_test_samples, test_feature_len)
    
    # Predict using the trained SVM
    _, predictions = svm.predict(test_features)

    # Calculate accuracy
    accuracy = np.mean(predictions.flatten() == test_labels.flatten()) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Example of checking a single image
    if len(test_pos_paths) > 0 and len(test_neg_paths) > 0:
        print("--- Single Image Prediction Examples ---")
        # Positive sample
        sample_pos_path = test_pos_paths[0]
        pos_feature, _ = extract_hog_features([sample_pos_path], hog, 1)
        pos_feature = pos_feature.reshape(1, -1)
        _, pos_pred = svm.predict(pos_feature)
        print(f"Prediction for a positive sample ({os.path.basename(sample_pos_path)}): {'Pedestrian' if pos_pred[0][0] == 1 else 'Background'}")

        # Negative sample
        sample_neg_path = test_neg_paths[0]
        neg_feature, _ = extract_hog_features([sample_neg_path], hog, -1)
        neg_feature = neg_feature.reshape(1, -1)
        _, neg_pred = svm.predict(neg_feature)
        print(f"Prediction for a negative sample ({os.path.basename(sample_neg_path)}): {'Pedestrian' if neg_pred[0][0] == 1 else 'Background'}")
