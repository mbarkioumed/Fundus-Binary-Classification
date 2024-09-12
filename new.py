import os
import cv2
import numpy as np
import tensorflow as tf

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

# Define paths
data_dir = 'data'
image_size = (224, 224)  # VGG16 expects 224x224 images
batch_size = 32

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)  # Apply Gaussian blur
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(blurred_image)  # Apply CLAHE
    resized_image = cv2.resize(enhanced_image, image_size)  # Resize image
    return resized_image

# Custom data generator to apply CLAHE filter
class CustomDataGenerator(ImageDataGenerator):
    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros((len(index_array), *image_size, 1), dtype='float32')
        batch_y = np.zeros((len(index_array),), dtype='float32')
        for i, j in enumerate(index_array):
            img_path = self.filepaths[j]
            img = preprocess_image(img_path)
            batch_x[i] = np.expand_dims(img, axis=-1)
            batch_y[i] = self.labels[j]
        return batch_x, batch_y

# Create data generators with augmentation
datagen = CustomDataGenerator(
    rescale=1./255,
    validation_split=0.9,
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # Use validation subset as test data
)

# Convert generator to tf.data.Dataset
test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, *image_size, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
)

# Define focal loss
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred)
        fl = - alpha_t * tf.keras.backend.pow((tf.ones_like(y_true) - p_t), gamma) * tf.keras.backend.log(p_t)
        return tf.keras.backend.mean(fl)
    return focal_loss_fixed

# Load the model
model = tf.keras.models.load_model('abnormal_detection_model.h5', custom_objects={'focal_loss_fixed': focal_loss()})

# Evaluate the model
results = model.evaluate(test_dataset, steps=test_generator.samples // batch_size)
print(f"Test Loss: {results[0]}")
print(f"Test Accuracy: {results[1]}")