# Import necessary libraries
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Path to the dataset
dataset_dir = 'dataset 1+2\\plant disease datset\\train'  # Adjust the path

# Image Data Generator for augmenting images and splitting dataset
datagen = ImageDataGenerator(rescale=1./255)

# Load dataset and manually split it into train, validation, and test sets
all_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),  
    batch_size=8,
    class_mode='categorical',
    shuffle=True
)

# Split the data into train, validation, and test sets
train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

# Build the CNN model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3rd Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the layers
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Add dropout for regularization

# Output Layer
model.add(Dense(22, activation='softmax'))  # 22 classes for 22 types of plant diseases

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and saving the best model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    train_data,
    epochs=50,
    validation_data=val_data,
    callbacks=[early_stopping, model_checkpoint]
)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_data)

# Save the final model
model.save('final_plant_disease_model.h5')

print(f"Test accuracy: {test_acc}")
