import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ================================
# CONFIGURATION
# ================================
DATASET_PATH = 'dataset'
TRAIN_DIR = os.path.join(DATASET_PATH, 'train')
TEST_DIR = os.path.join(DATASET_PATH, 'test')
IMG_SIZE = 48  # FER2013 images are 48x48 pixels
BATCH_SIZE = 64
EPOCHS = 50  # Maximum epochs (will stop early if no improvement)
MODEL_SAVE_PATH = 'face_emotionModel.h5'

# Emotion labels (must match folder names in dataset)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print("=" * 60)
print("🎭 FACE EMOTION DETECTION - MODEL TRAINING")
print("=" * 60)
print(f"📂 Training data: {TRAIN_DIR}")
print(f"📂 Testing data: {TEST_DIR}")
print(f"🎯 Emotions to detect: {len(EMOTIONS)} classes")
print(f"📏 Image size: {IMG_SIZE}x{IMG_SIZE} pixels")
print("=" * 60)

# ================================
# DATA PREPARATION
# ================================
print("\n📊 STEP 1: Preparing data...")

# Data augmentation for training (creates variations of images to improve learning)
train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize pixel values to 0-1
    rotation_range=10,            # Randomly rotate images
    width_shift_range=0.1,        # Randomly shift horizontally
    height_shift_range=0.1,       # Randomly shift vertically
    zoom_range=0.1,               # Randomly zoom
    horizontal_flip=True,         # Randomly flip images
    fill_mode='nearest'
)

# Only rescaling for test data (no augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=True
)

# Load testing data
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical',
    shuffle=False
)

print(f"✅ Training samples: {train_generator.samples}")
print(f"✅ Testing samples: {test_generator.samples}")
print(f"✅ Classes detected: {train_generator.class_indices}")

# ================================
# MODEL ARCHITECTURE
# ================================
print("\n🏗️ STEP 2: Building the CNN model...")

model = keras.Sequential([
    # First Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # Second Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # Third Convolutional Block
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    
    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    
    # Output Layer (7 emotions)
    layers.Dense(len(EMOTIONS), activation='softmax')
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model architecture created!")
model.summary()

# ================================
# TRAINING CALLBACKS
# ================================
print("\n🎓 STEP 3: Setting up training callbacks...")

# Early stopping: stop training if validation accuracy doesn't improve
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate when validation accuracy plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]

# ================================
# TRAINING
# ================================
print("\n🚀 STEP 4: Training the model...")
print(f"⏱️ Maximum epochs: {EPOCHS}")
print("💡 Training will stop early if no improvement is detected\n")

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ================================
# SAVE MODEL
# ================================
print("\n💾 STEP 5: Saving the trained model...")
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved successfully as: {MODEL_SAVE_PATH}")

# ================================
# EVALUATION
# ================================
print("\n📈 STEP 6: Evaluating model performance...")

test_loss, test_accuracy = model.evaluate(test_generator)
print(f"\n🎯 Final Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"📉 Final Test Loss: {test_loss:.4f}")

# ================================
# VISUALIZATION
# ================================
print("\n📊 STEP 7: Generating training history plots...")

# Plot training history
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("✅ Training history saved as: training_history.png")

print("\n" + "=" * 60)
print("🎉 MODEL TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"✅ Model file: {MODEL_SAVE_PATH}")
print(f"✅ Training accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
print(f"✅ Validation accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")
print(f"✅ Test accuracy: {test_accuracy * 100:.2f}%")
print("=" * 60)