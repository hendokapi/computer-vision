
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from pathlib import Path

# --- 1. DATA PREPARATION ---

def generate_synthetic_data(base_dir, img_size=(150, 150)):
    """Generates a synthetic dataset for binary classification (rectangles vs. noise)."""
    print("Generating synthetic data...")
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    # Define paths
    train_pos_dir = os.path.join(train_dir, 'positive')
    train_neg_dir = os.path.join(train_dir, 'negative')
    validation_pos_dir = os.path.join(validation_dir, 'positive')
    validation_neg_dir = os.path.join(validation_dir, 'negative')

    # Create directories
    for directory in [train_pos_dir, train_neg_dir, validation_pos_dir, validation_neg_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Generate and save images
    def save_images(num, path, is_positive):
        for i in range(num):
            if is_positive:
                img = np.zeros((*img_size, 3), dtype=np.uint8)
                x1, y1 = np.random.randint(0, img_size[0]//2, 2)
                x2, y2 = np.random.randint(img_size[0]//2, img_size[0], 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
            else:
                img = np.random.randint(0, 255, (*img_size, 3), dtype=np.uint8)
            # VGG16 expects 3-channel images, so we create them even if content is grayscale
            cv2.imwrite(os.path.join(path, f"sample_{i:04d}.png"), img)
            
    save_images(500, train_pos_dir, True) # 500 positive training samples
    save_images(500, train_neg_dir, False) # 500 negative training samples
    save_images(150, validation_pos_dir, True) # 150 positive validation samples
    save_images(150, validation_neg_dir, False) # 150 negative validation samples

    print("Synthetic data generation complete.")

# --- Main Execution ---

if __name__ == "__main__":
    base_data_dir = "transfer_learning_data"
    
    # Generate data if directories are empty
    if not os.listdir(os.path.join(base_data_dir, "train", "positive")):
        generate_synthetic_data(base_data_dir)
    else:
        print("Data already exists, skipping generation.")

    # --- 2. LOAD PRE-TRAINED BASE AND BUILD MODEL ---
    # Image dimensions for VGG16
    IMG_WIDTH, IMG_HEIGHT = 150, 150

    # Load VGG16 convolutional base
    conv_base = VGG16(weights='imagenet',
                      include_top=False, # Do not include the ImageNet classifier
                      input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    # Freeze the convolutional base
    conv_base.trainable = False
    print(f"The VGG16 base has {len(conv_base.layers)} layers.")
    print(f"VGG16 base is now frozen. Trainable status: {conv_base.trainable}")

    # Create the new model on top
    model = Sequential([
        conv_base,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # Sigmoid for binary classification
    ])

    model.summary()

    # --- 3. SETUP DATA GENERATORS & PHASE 1: TRAIN THE HEAD ---

    # Use ImageDataGenerator for rescaling and augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    # Note: validation data should not be augmented!
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(base_data_dir, 'train'),
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=20,
        class_mode='binary') # binary_crossentropy loss needs binary labels

    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(base_data_dir, 'validation'),
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=20,
        class_mode='binary')

    # Compile and train the model (Phase 1)
    print("--- Phase 1: Training the Classifier Head ---")
    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=50,  # 1000 train images / batch_size 20
        epochs=30,
        validation_data=validation_generator,
        validation_steps=15) # 300 validation images / batch_size 20

    # --- 4. PHASE 2: FINE-TUNING ---
    print("--- Phase 2: Fine-Tuning Top Layers ---")

    # Unfreeze the top convolutional block of VGG16
    conv_base.trainable = True
    for layer in conv_base.layers[:-4]: # Keep the first layers frozen
        layer.trainable = False
    
    print("Re-compiling model for fine-tuning with a very low learning rate.")
    model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Resume training
    history_fine = model.fit(
        train_generator,
        steps_per_epoch=50,
        epochs=20, # Train for 20 more epochs
        initial_epoch=history.epoch[-1] + 1, # Resume from where we left off
        validation_data=validation_generator,
        validation_steps=15)

    # --- 5. VISUALIZE RESULTS ---

    # Combine history objects
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.axvline(x=29, color='r', linestyle='--', label='Start Fine-Tuning')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.axvline(x=29, color='r', linestyle='--', label='Start Fine-Tuning')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    print("Fine-tuning history plot saved to finetuning_history.png")
