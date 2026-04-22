
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

# --- 1. DATA PREPARATION ---

def preprocess_data(examples):
    # Resize images and normalize
    examples["image"] = [image.convert("RGB").resize((150, 150)) for image in examples["image"]]
    examples["image"] = [tf.keras.preprocessing.image.img_to_array(image) / 255.0 for image in examples["image"]]
    # One-hot encode labels
    examples["labels"] = tf.one_hot(examples["labels"], 3)
    return examples

if __name__ == "__main__":
    # Load the beans dataset from Hugging Face
    print("Loading 'beans' dataset from Hugging Face...")
    dataset = load_dataset("beans")

    # Preprocess the dataset
    print("Preprocessing the dataset...")
    dataset = dataset.map(preprocess_data, batched=True)

    # Convert to tf.data.Dataset
    train_dataset = dataset["train"].to_tf_dataset(
        columns="image",
        label_cols="labels",
        batch_size=20,
        shuffle=True
    )

    validation_dataset = dataset["validation"].to_tf_dataset(
        columns="image",
        label_cols="labels",
        batch_size=20,
        shuffle=False
    )

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

    # Create the new model on top for 3 classes
    model = Sequential([
        conv_base,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax') # Softmax for multi-class classification
    ])

    model.summary()

    # --- 3. SETUP DATA GENERATORS & PHASE 1: TRAIN THE HEAD ---

    # Compile and train the model (Phase 1)
    print("--- Phase 1: Training the Classifier Head ---")
    model.compile(optimizer=optimizers.RMSprop(learning_rate=2e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_dataset,
        epochs=30,
        validation_data=validation_dataset)

    # --- 4. PHASE 2: FINE-TUNING ---
    print("--- Phase 2: Fine-Tuning Top Layers ---")

    # Unfreeze the top convolutional block of VGG16
    conv_base.trainable = True
    for layer in conv_base.layers[:-4]: # Keep the first layers frozen
        layer.trainable = False
    
    print("Re-compiling model for fine-tuning with a very low learning rate.")
    model.compile(optimizer=optimizers.RMSprop(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Resume training
    history_fine = model.fit(
        train_dataset,
        epochs=50, # Train for 20 more epochs
        initial_epoch=history.epoch[-1] + 1, # Resume from where we left off
        validation_data=validation_dataset)

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
