# Import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import random
from PIL import Image
import pathlib
import os

# Specify the directory where the data is located
data_dir = pathlib.Path('/Users/ali/Documents/Apidon/OpenCV/simpsons_dataset').with_suffix('')
# Initialize a counter for directories
folder_count = 0

# Loop through each item in data_dir
for item in data_dir.iterdir():
    # Check if the item is a directory
    if item.is_dir():
        # Increment the folder_count by 1
        folder_count += 1
print(folder_count)

# Calculate the total number of images in the dataset
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# Choose a random image from a specific category to display
category = "bart_simpson"
images = list(data_dir.glob(f'{category}/*.jpg')) 
random_image = random.choice(images)  # Randomly select an image
image = Image.open(str(random_image))
image.show()

#Create Dataset
batch_size = 32
img_height = 180
img_width = 180

# Preparing training dataset with a split of 80% and validation dataset with a split of 20%
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Retrieving class names automatically from data directory structure
class_names = train_ds.class_names
print(class_names)

# Displaying a sample of 9 images and their labels from the training dataset
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax=plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

# Previewing the shape of an image batch and its corresponding labels batch
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Configure the dataset for performance 

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Standardize the data
normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# A basic Keras model
  # Create the model
num_classes = len(class_names)
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

# Data augmentation
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# Visualizing how augmentation will look on some training images
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")
plt.show()

# Redefining the model to include dropout for regularization to prevent overfitting
model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs")
])

#Compile and train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary() # Displaying a summary of the model architecture

# Training the model for 12 epochs
epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Plotting training & validation accuracy/loss to assess model performance over time
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Saving the trained model for future use
#model.save('/Users/ali/Documents/Apidon/apidon-classify/model.h5', overwrite=True, save_format='h5')
model_save_path = os.path.join(os.getcwd(), "model.h5")
model.save(model_save_path, overwrite=True, save_format='h5')
