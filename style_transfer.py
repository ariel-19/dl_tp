import tensorflow as tf
from tensorflow import keras
import numpy as np

# --- Helper function to make lines 5 and 6 work ---
def load_and_process_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (512, 512))
    return img[tf.newaxis, :] # Add batch dimension


# Load a content image and a style image
content_path = 'turtle.jpg'
style_path = 'kandinsky.jpg'
# Load and pre-process two images of your choice
content_image = load_and_process_img(content_path)
style_image = load_and_process_img(style_path)

# Load the pre-trained VGG16 model
vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
vgg.trainable = False # Important: the VGG model is not being trained here

# Content and style layers for feature extraction
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Function to create an extractor model
def create_extractor(model, style_layers, content_layers):
    # The model outputs will be the activations of the selected layers
    outputs = [model.get_layer(name).output for name in style_layers + content_layers]
    return keras.Model(inputs=model.input, outputs=outputs)

extractor = create_extractor(vgg, style_layers, content_layers)


print("Exercise 4 Completed: Extractor model created successfully.")
print(f"Content Image Shape: {content_image.shape}")