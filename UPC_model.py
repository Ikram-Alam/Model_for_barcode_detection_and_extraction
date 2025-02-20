import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Load images and labels (UPCs)
def load_data(image_folder, label_file):
    images = []
    labels = []
    
    # Assuming images are named after UPC number
    for file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 32))  # Resize to fit input dimensions
        images.append(image)
        
        # Assuming labels are stored in a text file with corresponding filenames
        with open(label_file, 'r') as f:
            label = f.readline().strip()  # Read corresponding label
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Load data
images, labels = load_data('upc_images_folder', 'upc_labels.txt')

# Normalize images
images = images / 255.0

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)




import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the CNN model
def create_model():
    model = Sequential()
    
    # 1st Conv Layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 128, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # 2nd Conv Layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten the output
    model.add(Flatten())
    
    # Dense Layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    # Output Layer (Assuming 12 digits in UPC)
    model.add(Dense(12, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Initialize model
model = create_model()

# Check model summary
model.summary()







from tensorflow.keras.utils import to_categorical

# One-hot encode labels
y_train_one_hot = to_categorical([list(map(int, y)) for y in y_train])
y_test_one_hot = to_categorical([list(map(int, y)) for y in y_test])

# Reshape images for model
X_train = X_train.reshape(-1, 32, 128, 1)
X_test = X_test.reshape(-1, 32, 128, 1)

# Train the model
history = model.fit(X_train, y_train_one_hot, epochs=20, batch_size=32, validation_data=(X_test, y_test_one_hot))








# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test_one_hot)
print(f"Test accuracy: {test_acc}")





# Save the model
model.save('upc_detection_model.h5')

# Load the model
loaded_model = tf.keras.models.load_model('upc_detection_model.h5')



# Predict UPC from new image
def predict_upc(image_path, model):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 32)) / 255.0
    image = image.reshape(1, 32, 128, 1)
    
    prediction = model.predict(image)
    upc_number = ''.join([str(np.argmax(p)) for p in prediction])
    return upc_number

# Predict UPC
upc_prediction = predict_upc('new_upc_image.png', loaded_model)
print(f"Predicted UPC: {upc_prediction}")
