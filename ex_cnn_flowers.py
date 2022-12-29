import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

base_dir = "C:\\Users\\Carl Kolon\\.keras\\datasets\\flower_photos"
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

total_train = 0
total_val=0
for i in classes:
    total_train += len(os.listdir(os.path.join(train_dir, i)))
    total_val += len(os.listdir(os.path.join(val_dir, i)))

batch_size = 100
img_shape = 150

image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                            horizontal_flip=True,
                                                            rotation_range=45,
                                                            zoom_range=.5,
                                                            fill_mode="nearest",
                                                            width_shift_range=.15,
                                                            height_shift_range=.15)

train_data_gen = image_gen.flow_from_directory(batch_size=batch_size,
                                               directory=train_dir,
                                               shuffle=True,
                                               target_size=(img_shape, img_shape),
                                               class_mode='sparse')

image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=val_dir,
                                                 shuffle=False,
                                                 target_size=(img_shape, img_shape),
                                                 class_mode='sparse')


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


# augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plot_images(augmented_images)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_shape, img_shape, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

epochs = 40
history = model.fit(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train/float(batch_size))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val/float(batch_size)))
)

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

