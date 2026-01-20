import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==========================================
# 1. CONV BLOCK (Helper Function)
# ==========================================
def conv_block(input_tensor, num_filters):
    """
    Standard U-Net block: Conv2D -> BatchNormalization -> ReLU (applied twice)
    """
    # First Conv Layer
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Second Conv Layer
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

# ==========================================
# 2. U-NET ARCHITECTURE BUILDER
# ==========================================
def build_unet(input_shape=(128, 128, 1)):
    """
    Builds a U-Net model for binary segmentation.
    """
    inputs = keras.Input(input_shape)

    # --- ENCODER PATH (Contracting) ---
    # Step 1: 32 filters
    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Step 2: 64 filters (TODO Completed)
    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Step 3: 128 filters (TODO Completed)
    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # --- BRIDGE / BOTTLENECK ---
    # 256 filters (Deepest point of the network)
    b = conv_block(p3, 256)

    # --- DECODER PATH (Expansive) ---
    # Step 1: Upsample to match c3 size
    u1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(b)
    # Skip Connection: Concatenate with c3 (TODO Completed)
    u1 = layers.Concatenate()([u1, c3])
    d1 = conv_block(u1, 128)

    # Step 2: Upsample to match c2 size (TODO Completed)
    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d1)
    u2 = layers.Concatenate()([u2, c2]) # Skip connection
    d2 = conv_block(u2, 64)

    # Step 3: Upsample to match c1 size (TODO Completed)
    u3 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(d2)
    u3 = layers.Concatenate()([u3, c1]) # Skip connection
    d3 = conv_block(u3, 32)

    # --- OUTPUT LAYER ---
    # 1 filter because output is a binary mask (0 or 1 per pixel)
    # Sigmoid activation forces values between 0 and 1
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d3)

    model = keras.Model(inputs=[inputs], outputs=[outputs], name="U-Net")
    return model

# ==========================================
# TEST THE ARCHITECTURE
# ==========================================
if __name__ == "__main__":
    print("Building U-Net Model...")
    model = build_unet(input_shape=(128, 128, 1))
    
    # Print summary to verify shapes
    # Notice how dimensions go down (128->64->32->16) and then back up (16->32->64->128)
    model.summary()
    print("U-Net built successfully.")

def dice_coeff(y_true, y_pred, smooth=1.):
    """
    Dice = (2 * Intersection) / (|A| + |B|)
    """
    # Flatten the tensors to 1D vectors for easy calculation
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    # Calculate intersection (True Positives)
    intersection = K.sum(y_true_f * y_pred_f)
    
    # TODO: Complete the Dice formula calculation
    # We add 'smooth' to numerator and denominator to prevent division by zero
    numerator = (2. * intersection) + smooth
    denominator = K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    
    return numerator / denominator

# Optional: Dice Loss (Useful for training)
def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)

# ==========================================
# 2. IOU (INTERSECTION OVER UNION)
# ==========================================
def iou_metric(y_true, y_pred, smooth=1.):
    """
    IoU = Intersection / Union
    Union = |A| + |B| - Intersection
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    
    # Calculate Union
    total_area = K.sum(y_true_f) + K.sum(y_pred_f)
    union = total_area - intersection
    
    return (intersection + smooth) / (union + smooth)