import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

VGG16 = tf.keras.applications.VGG16
Model = tf.keras.models.Model
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
Adam = tf.keras.optimizers.Adam

# Apply CLAHE to an image
def apply_clahe(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply CLAHE to the grayscale image
    clahe_img = clahe.apply(gray)
    # Convert back to BGR format to maintain shape
    clahe_img_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    return clahe_img_bgr

# Load and preprocess images
def load_and_preprocess_images(folder, label, image_size=(512, 512)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.tif')):  # Adjust if using other image formats
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # Apply CLAHE
                img = apply_clahe(img)
                img = cv2.resize(img, image_size)  # Resize to match model input size
                img = img / 255.0  # Normalize pixel values
                images.append(img)
                labels.append(label)
            else:
                print("Warning could not load image hehehehehehe" + img_path)
    return np.array(images), np.array(labels)

# Paths
test_folder_sick = 'data/abnormal'
test_folder_normal = 'data/normal'

# Load data
X_sick, y_sick = load_and_preprocess_images(test_folder_sick, label=1)
X_normal, y_normal = load_and_preprocess_images(test_folder_normal, label=0)

# Combine sick and normal data
X = np.concatenate((X_sick, X_normal), axis=0)
y = np.concatenate((y_sick, y_normal), axis=0)

# Compute class weights to handle class imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)
class_weights = dict(enumerate(class_weights))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation for both classes
datagen = ImageDataGenerator(
    rotation_range=90,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the data generator to the training data
datagen.fit(X_train)

# Define the model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(512, 512, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with augmented data
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          epochs=10,
          validation_data=(X_val, y_val),
          class_weight=class_weights)

# Save the model
model.save('clahe_augmented_eye_disease_classifier.h5')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {test_accuracy}")
print(f"Validation Loss: {test_loss}")
