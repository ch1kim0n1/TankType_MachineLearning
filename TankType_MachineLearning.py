import keras.utils as image
from keras.utils import load_img, img_to_array
import os
import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.optimizers import Adam

# Define the number of classes
num_classes = 5

# Load the pre-trained VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False)

# Add a new top layer for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Define the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with a low learning rate to avoid overfitting
adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Load images and labels from a folder
X_train = []
y_train = []

folder = r'C:\Users\skyla\Desktop\GitHub Code\TankType_MachineLearning\Example Images'
for i, filename in enumerate(os.listdir(folder)):
    img_path = os.path.join(folder, filename)
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    X_train.append(x)
    if "Light" in filename:
        y_train.append(0)
    elif "Middle" in filename:
        y_train.append(1)
    elif "Heavy" in filename:
        y_train.append(2)
    elif "Anti" in filename:
        y_train.append(3)
    elif "Artillery" in filename:
        y_train.append(4)

X_train = np.vstack(X_train)
y_train = np.array(y_train)

# Convert the labels into one-hot encoded format
y_train = to_categorical(y_train, num_classes)

# Train the model on your data using data augmentation
data_gen = image.ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False
)

# Train the model on your data
#EPOCHS is number of tries model can train (initially 10)
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict classes for new images
# Can switch image by changing name of the picture from the folder Test Images
#REAL_1.jpg as one of the examples
#RU_LI_1.png
img_path = r'C:\Users\skyla\Desktop\GitHub Code\TankType_MachineLearning\Test Images\RU_HEA_1.png'
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)


while True:
    # Print the class with the highest probability
    class_index = np.argmax(preds[0])
    if class_index == 0:
        print("Light Tank")
    elif class_index == 1:
        print("Middle Tank")
    elif class_index == 2:
        print("Heavy Tank")
    elif class_index == 3:
        print("Anti Tank")
    elif class_index == 4:
        print("Artillery")
    user_input = input("Is the prediction correct? (y/n)")

    if user_input == 'y':
        break
    elif user_input == 'n':
        # Update the path to a new image
        img_path = input("Enter the path to a new image:")
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
    else:
        print("Invalid input. Please enter 'y' or 'n'.")