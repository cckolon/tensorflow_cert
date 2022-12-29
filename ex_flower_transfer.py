import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow.keras import layers

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

(training_set, validation_set), dataset_info = tfds.load('tf_flowers',
                                                         split=['train[:70%]', 'train[70%:]'],
                                                         with_info=True,
                                                         as_supervised=True)

num_classes = dataset_info.features['label'].num_classes
num_training_examples = dataset_info.splits['train[:70%]'].num_examples
num_validation_examples = dataset_info.splits['train[70%:]'].num_examples

print('Total Number of Classes: {}'.format(num_classes))
print('Total Number of Training Images: {}'.format(num_training_examples))
print('Total Number of Validation Images: {} \n'.format(num_validation_examples))

batch_size = 32
img_shape = 224


def format_image(image, label):
    image = tf.image.resize(image, (img_shape, img_shape))/255.0
    return image, label


train_batches = training_set.shuffle(num_training_examples).map(format_image).batch(batch_size).prefetch(1)

val_batches = validation_set.shuffle(num_validation_examples).map(format_image).batch(batch_size).prefetch(1)

url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor = hub.KerasLayer(url, input_shape=(img_shape, img_shape, 3))
feature_extractor.trainable = False

model = tf.keras.Sequential([
    feature_extractor,
    layers.Dense(num_classes)])

epochs = 6

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(train_batches,
                    epochs=epochs,
                    validation_data=val_batches)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), acc, label='Training Accuracy')
plt.plot(range(epochs), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.subplot(1, 2, 2)
plt.plot(range(epochs), loss, label='Training Loss')
plt.plot(range(epochs), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.

image_batch, label_batch = next(iter(train_batches.take(1)))
image_batch = image_batch.numpy()
label_batch = label_batch.numpy()

result_batch = model.predict(image_batch)

predicted_class_names = dataset_info.features['label']

print(predicted_class_names)


