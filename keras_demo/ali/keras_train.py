import os
import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt

# Image size
image_size = (90, 90)

# Batch size
batch_size = 32

# Path to the Intel dataset folder
data_dir = "/Users/ali/Documents/Apidon/Keras/archive/seg_train/seg_train"

# Create training and validation datasets
train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True,
    subset="training",
    validation_split=0.2,
    seed=123,
)
data_dir2 = "/Users/ali/Documents/Apidon/Keras/archive/seg_test/seg_test"

val_ds = keras.utils.image_dataset_from_directory(
    data_dir2,
    image_size=image_size,
    batch_size=batch_size,
    shuffle=True,
    subset="validation",
    validation_split=0.2,
    seed=123,
)

# Access class_names from the original dataset
class_names = train_ds.class_names

# Visualize training and validation datasets
fig, axes = plt.subplots(1, 2, figsize=(10, 10))

# Assuming you have six classes in your dataset
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

for ds, ax in zip([train_ds, val_ds], axes):
    for images, labels in ds.take(1):
        ax.imshow(images[0].numpy().astype("uint8"))

        label_str = class_names[labels[0].numpy()]
        ax.set_title(label_str)

# Data normalization
normalization_layer = layers.Rescaling(1.0 / 255)

# Build the model
model = keras.Sequential(
    [
        normalization_layer,
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(image_size[0], image_size[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(len(class_names), activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Build the model
model.build((batch_size, image_size[0], image_size[1], 3))

# Model summary
model.summary()

# Train the model
epochs = 10

history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
)

# Visualize training results
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Save the model
model.save("model.h5")
