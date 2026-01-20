import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ... (Previous data loading code should be above this) ...

print("1. Loading CIFAR-10 dataset...")
# Load the dataset (images and labels)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Define constants
NUM_CLASSES = 10
INPUT_SHAPE = x_train.shape[1:] # (32, 32, 3)

print(f"Original Training Data Shape: {x_train.shape}")
print(f"Original Test Data Shape: {x_test.shape}")

# 2. Normalize pixel values to [0, 1]
# We convert to float32 first to ensure precision
print("2. Normalizing pixel values...")
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 3. Convert labels to One-Hot Encoding
# Example: Label '3' becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
print("3. Converting labels to One-Hot Encoding...")
y_train = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

# Output required by the exercise
print(f"\n--- DATA PREPARATION REPORT ---")
print(f"Input data shape: {INPUT_SHAPE}")
print(f"Labels shape after conversion: {y_train.shape}")
print(f"Example Label (One-Hot): {y_train[0]}")

# ==========================================
# PART 2.1: CLASSIC CNN ARCHITECTURE
# ==========================================
def build_basic_cnn(input_shape, num_classes):
    print("\nBuilding Basic CNN Model...")
    model = keras.Sequential([
        # --- Block 1 ---
        # Convolutional Layer 1: 32 filters, 3x3 size, ReLU activation
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        # Pooling Layer 1: Max Pooling 2x2 (Added as per instructions)
        layers.MaxPooling2D(pool_size=(2, 2)),

        # --- Block 2 ---
        # Convolutional Layer 2: 64 filters, 3x3 size, ReLU activation (Added)
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        # Pooling Layer 2: Max Pooling 2x2
        layers.MaxPooling2D(pool_size=(2, 2)),

        # --- Classification Head ---
        # Flatten Layer to transition to Dense layers
        layers.Flatten(),

        # Dense Layer 1: 512 units
        layers.Dense(512, activation='relu'),
        
        # Output Layer: num_classes units
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ==========================================
# TRAINING EXECUTION
# ==========================================
# Ensure data variables (x_train, y_train, etc.) are available from Part 1.2
if 'x_train' in locals():
    # 1. Build
    model = build_basic_cnn(INPUT_SHAPE, NUM_CLASSES)
    model.summary()

    # 2. Compile
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 3. Train
    print("\nStarting Training (10 Epochs)...")
    history = model.fit(
        x_train, 
        y_train,
        batch_size=64,
        epochs=10,
        validation_split=0.1 # 10% held out for validation (Dev set)
    )

    # 4. Evaluate on Test Set
    print("\nEvaluating on Test Set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final Test Accuracy: {test_acc:.4f}")
    
def residual_block(x, filters, kernel_size=(3, 3), stride=1):
    """
    Implements a Residual Block: y = F(x) + x
    """
    # --- Main Path (F(x)) ---
    # First Convolution
    y = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
    
    # Second Convolution (No activation yet)
    y = layers.Conv2D(filters, kernel_size, padding='same')(y)

    # --- Skip Connection Path (x) ---
    # If we change dimensions (stride > 1) or filter count, we must resize x to match y
    if stride > 1 or x.shape[-1] != filters:
        x = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(x)

    # --- Addition (The Residual Connection) ---
    z = layers.Add()([x, y])

    # Final Activation after addition
    z = layers.Activation('relu')(z)
    return z


# Exercise: 2.2
def build_mini_resnet(input_shape, num_classes):
    """
    Builds a small architecture using 3 consecutive residual blocks
    """
    inputs = keras.Input(shape=input_shape)

    # Initial Convolution layer
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)

    # Block 1: 32 filters, stride 1 (Dimensions remain 32x32)
    x = residual_block(x, 32)

    # Block 2: 64 filters, stride 2 (Dimensions reduce to 16x16)
    x = residual_block(x, 64, stride=2)

    # Block 3: 64 filters, stride 1 (Dimensions remain 16x16)
    x = residual_block(x, 64)

    # Classification Head
    x = layers.GlobalAveragePooling2D()(x) # A modern alternative to Flatten
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Mini_ResNet")
    return model

# ==========================================
# TEST BUILD
# ==========================================
if __name__ == "__main__":
    # We define dummy shapes just to test the build
    INPUT_SHAPE = (32, 32, 3) 
    NUM_CLASSES = 10

    print("\nBuilding Mini-ResNet...")
    resnet_model = build_mini_resnet(INPUT_SHAPE, NUM_CLASSES)
    
    # Print summary to verify the "Add" layers exist
    resnet_model.summary()
    
    print("\nResNet built successfully")