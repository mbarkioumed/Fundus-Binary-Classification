import os
import cv2
import numpy as np
import tensorflow as tf

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
BatchNormalization = tf.keras.layers.BatchNormalization
Adam = tf.keras.optimizers.Adam
EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
VGG16 = tf.keras.applications.VGG16

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
    validation_split=0.2,
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Convert generators to tf.data.Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, *image_size, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
).repeat()

validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, *image_size, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
).repeat()

# Calculate steps per epoch
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

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

# Build the model using VGG16 with transfer learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*image_size, 3))
base_model.trainable = False  # Freeze the base model

model = Sequential([
    tf.keras.layers.Conv2D(3, (3, 3), padding='same', input_shape=(*image_size, 1)),  # Convert grayscale to 3 channels
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=Adam(), loss=focal_loss(), metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0003)

# Train the model
model.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs=20,
    validation_data=validation_dataset,
    callbacks=[early_stopping, reduce_lr]
)

# Save the model
model.save('abnormal_detection_model.h5')