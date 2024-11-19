#!/usr/bin/env python
# coding: utf-8

# # Data Loading and splitting it into 2 directories test and train

# In[1]:


pip install tensorflow numpy matplotlib


# In[2]:


pip install opencv-python


# In[8]:


import os
import shutil
import random

def split_dataset(dataset_dir, train_dir, test_dir, split_ratio=0.7):
    # Ensure the directories exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    # Iterate over each class folder in the dataset
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        
        if os.path.isdir(class_dir):
            images = os.listdir(class_dir)
            random.shuffle(images)  # Shuffle images
            
            train_size = int(len(images) * split_ratio)
            
            train_images = images[:train_size]
            test_images = images[train_size:]
            
            # Create class folders in train and test directories
            train_class_dir = os.path.join(train_dir, class_name)
            test_class_dir = os.path.join(test_dir, class_name)
            
            if not os.path.exists(train_class_dir):
                os.makedirs(train_class_dir)
            if not os.path.exists(test_class_dir):
                os.makedirs(test_class_dir)
            
            # Move files to train and test folders
            for image in train_images:
                shutil.move(os.path.join(class_dir, image), os.path.join(train_class_dir, image))
            
            for image in test_images:
                shutil.move(os.path.join(class_dir, image), os.path.join(test_class_dir, image))

# Paths to your dataset and target directories
dataset_dir = 'D:\\Main Project\\Padded_imgs'  # Replace with the path to your dataset
train_dir = 'D:\\Main Project\\Train'
test_dir = 'D:\\Main Project\\Test'

# Split the dataset
split_dataset(dataset_dir, train_dir, test_dir, split_ratio=0.7)


# In[9]:


import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Preprocessing and loading data
def preprocess_and_load_data(train_dir, test_dir, img_size=(128, 128)):
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        color_mode='grayscale',  # Use 'rgb' if your images are colored
        batch_size=32,
        class_mode='categorical'
    )
    
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical'
    )
    
    return train_generator, test_generator

train_generator, test_generator = preprocess_and_load_data(train_dir, test_dir)

# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Load class names from the train generator
num_classes = len(train_generator.class_indices)
input_shape = (128, 128, 1)  # Adjust if images are RGB

model = create_cnn_model(input_shape, num_classes)
model.summary()







# In[10]:


# Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=test_generator
)


# In[11]:


# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")


# In[12]:


# Plot training history
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()


# In[13]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.show()


# In[14]:


# Save the model
model.save('target_detection_model.h5')

