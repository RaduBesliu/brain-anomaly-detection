# Name: Besliu Radu-Stefan
# Group: 243
# File: CNN.py
import numpy as np
from keras import layers, models, callbacks
from PIL import Image
from sklearn import metrics
from keras.preprocessing.image import ImageDataGenerator

# Store the training labels
train_labels = []
with open('/kaggle/input/unibuc-brain-ad/data/train_labels.txt', "r") as f:
    # remove the first line of the file, because it contains the column names ( id, class )
    f.readline()
    # load the labels from the file, split the by "," and save only the second column (the labels)
    train_labels = np.loadtxt(f, delimiter=",", usecols=(1,))

# Store the validation labels
validation_labels = []
with open('/kaggle/input/unibuc-brain-ad/data/validation_labels.txt', "r") as f:
    # remove the first line of the file, because it contains the column names ( id, class )
    f.readline()
    # load the labels from the file, split the by "," and save only the second column (the labels)
    validation_labels = np.loadtxt(f, delimiter=",", usecols=(1,))

# Store the training images
train_images = []
for i in range(1, 15001):
    # images are in the format: "000001.png", "000002.png", ..., "015000.png"
    # that's why we need to use zfill(6) to add zeros to the left of the number until it has 6 digits
    image = Image.open('/kaggle/input/unibuc-brain-ad/data/data/' + str(i).zfill(6) + '.png')
    # convert the image to grayscale
    grayscale_image = image.convert('L')
    # resize the image to 188x188 ( for model 6 )
    # for the other models, the image is resized to 128x128, or it is not resized at all
    # use resample=Image.BICUBIC to use bicubic interpolation
    resized_image = grayscale_image.resize((188, 188), resample=Image.BICUBIC)
    # convert the image to a numpy array
    numpy_image = np.array(resized_image)
    # add the image to the list of training images
    train_images.append(numpy_image)

# Store the validation images
validation_images = []
for i in range(15001, 17001):
    # images are in the format: "015001.png", "015002.png", ..., "017000.png"
    # that's why we need to use zfill(6) to add zeros to the left of the number until it has 6 digits
    image = Image.open('/kaggle/input/unibuc-brain-ad/data/data/' + str(i).zfill(6) + '.png')
    # convert the image to grayscale
    grayscale_image = image.convert('L')
    # resize the image to 188x188 ( for model 6 )
    # for the other models, the image is resized to 128x128, or it is not resized at all
    resized_image = grayscale_image.resize((188, 188), resample=Image.BICUBIC)
    # convert the image to a numpy array
    numpy_image = np.array(resized_image)
    # add the image to the list of validation images
    validation_images.append(numpy_image)

# Store the test images
test_images = []
for i in range(17001, 22150):
    # images are in the format: "017001.png", "017002.png", ..., "022149.png"
    # that's why we need to use zfill(6) to add zeros to the left of the number until it has 6 digits
    image = Image.open('/kaggle/input/unibuc-brain-ad/data/data/' + str(i).zfill(6) + '.png')
    # convert the image to grayscale
    grayscale_image = image.convert('L')
    # resize the image to 188x188 ( for model 6 )
    # for the other models, the image is resized to 128x128, or it is not resized at all
    resized_image = grayscale_image.resize((188, 188), resample=Image.BICUBIC)
    # convert the image to a numpy array
    numpy_image = np.array(resized_image)
    # add the image to the list of test images
    test_images.append(numpy_image)

# Convert the images to numpy arrays
train_images = np.array(train_images)
validation_images = np.array(validation_images)
test_images = np.array(test_images)

# Normalize the images
# We divide by 255 because the pixel values are in the range [0, 255] and we want them in the range [0, 1]
# ( normalization for each variant of the model, except for the first one )
train_images = train_images / 255
validation_images = validation_images / 255
test_images = test_images / 255

# Expand the dimensions of the images to fit the input layer of the CNN
# We need to add a new dimension to the end of the array, because the input layer expects a 4D array
train_images = np.expand_dims(train_images, axis=-1)
validation_images = np.expand_dims(validation_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)


# CNN v1
# model = models.Sequential()
#
# # Input layer
# model.add(layers.Input(shape=(224, 224, 3)))
#
# # Convolutional layers
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D())
#
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D())
#
# model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D())
#
# model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D())
#
# model.add(layers.Conv2D(512, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D())
#
# model.add(layers.Conv2D(512, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D())
#
# # Flatten layer
# model.add(layers.Flatten())
#
# # Fully connected layers
# model.add(layers.Dense(1024, activation='relu'))
# model.add(layers.Dense(1024, activation='relu'))
#
# # Output layer
# model.add(layers.Dense(1, activation='sigmoid'))
#
# # Early stopping callback to prevent overfitting and save time
# early_stopping = callbacks.EarlyStopping(monitor='loss', patience=3)
#
# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
#
# # Train the model
# model.fit(train_images, train_labels, epochs=100, batch_size=64, callbacks=[early_stopping], validation_data=(validation_images, validation_labels))


# CNN v2
# model = models.Sequential()
#
# # Input layer
# model.add(layers.Input(shape=(128, 128, 1)))
#
# # Convolutional layers
# model.add(layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.25))
#
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.25))
#
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.25))
#
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.25))
#
#
# # Flatten layer
# model.add(layers.Flatten())
#
# # Fully connected layers
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(512, activation='relu'))
#
# # Output layer
# model.add(layers.Dense(1, activation='sigmoid'))
#
# early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
# model.fit(train_images, train_labels, epochs=100, batch_size=64, callbacks=[early_stopping], validation_data=(validation_images, validation_labels))


# CNN v3
# model = models.Sequential()
#
# # Input layer
# model.add(layers.Input(shape=(128, 128, 1)))
#
# # Convolutional layers
# model.add(layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(32, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.25))
#
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.25))
#
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.25))
#
# model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.25))
#
#
# # Flatten layer
# model.add(layers.Flatten())
#
# # Fully connected layers
# model.add(layers.Dense(256, activation='relu'))
#
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dropout(0.4))
#
# # Output layer
# model.add(layers.Dense(1, activation='sigmoid'))
#
# early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
# model.fit(image_data_generator.flow(train_images, train_labels, batch_size=32), epochs=100, callbacks=[early_stopping], validation_data=(validation_images, validation_labels))


# CNN v4
# image_data_generator = ImageDataGenerator(
#     rotation_range=15,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )
#
# # Fit the generator to the data
# image_data_generator.fit(train_images)
#
# model = models.Sequential()
#
# # Input layer
# model.add(layers.Input(shape=(128, 128, 1)))
#
# # Convolutional layers
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.25))
#
# model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.25))
#
# model.add(layers.Conv2D(512, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(512, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.25))
#
# model.add(layers.Conv2D(1024, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(1024, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.25))
#
#
# # Flatten layer
# model.add(layers.Flatten())
#
# # Fully connected layers
# model.add(layers.Dense(512, activation='relu'))
#
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dropout(0.4))
#
# # Output layer
# model.add(layers.Dense(1, activation='sigmoid'))
#
# early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
# class_weight = {0: 1.0, 1: 2.5}
# model.fit(
#     image_data_generator.flow(train_images, train_labels, batch_size=32),
#     epochs=150,
#     callbacks=[early_stopping],
#     class_weight=class_weight,
#     validation_data=(validation_images, validation_labels)
# )


# CNN v5
# image_data_generator = ImageDataGenerator(
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True
# )
#
# # Fit the generator to the data
# image_data_generator.fit(train_images)
#
# model = models.Sequential()
#
# # Input layer
# model.add(layers.Input(shape=(128, 128, 1)))
#
# # Convolutional layers
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.25))
#
# model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(256, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.25))
#
# model.add(layers.Conv2D(512, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(512, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.25))
#
# model.add(layers.Conv2D(1024, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(1024, (3, 3), activation='relu'))
# model.add(layers.BatchNormalization())
# model.add(layers.MaxPooling2D())
# model.add(layers.Dropout(0.25))
#
#
# # Flatten layer
# model.add(layers.Flatten())
#
# # Fully connected layers
# model.add(layers.Dense(512, activation='relu'))
#
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dropout(0.4))
#
# # Output layer
# model.add(layers.Dense(1, activation='sigmoid'))
#
# early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
# class_weight = {0: 1.0, 1: 2.5}
# model.fit(
#     image_data_generator.flow(train_images, train_labels, batch_size=32),
#     epochs=150,
#     callbacks=[early_stopping],
#     class_weight=class_weight,
#     validation_data=(validation_images, validation_labels)
# )


# CNN v6 ( comments added only for this model, the rest of the models are the same )
# Image augmentation
image_data_generator = ImageDataGenerator(
    rotation_range=12,  # rotate the image up to +/-12 degrees
    width_shift_range=0.2,  # shift the image up to +/-20% of the width
    height_shift_range=0.2,  # shift the image up to +/-20% of the height
    zoom_range=0.12,  # zoom the image up to +/-12%
    horizontal_flip=True  # flip the image horizontally
)

# Fit the generator to the data
image_data_generator.fit(train_images)

# Create the model
model = models.Sequential()

# Add the input layer, in which we specify the shape of the input images
model.add(layers.Input(shape=(188, 188, 1)))  # 1 channel because the images are grayscale

# Add the convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # 32 filters, each of size 3x3
model.add(layers.BatchNormalization())  # normalize the output of the previous layer

model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # 32 filters, each of size 3x3
model.add(layers.BatchNormalization())  # normalize the output of the previous layer
model.add(layers.MaxPooling2D())  # reduce the size of the output of the previous layer by 2
model.add(layers.Dropout(0.25))  # randomly drop 25% of the output of the previous layer

model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # 64 filters, each of size 3x3
model.add(layers.BatchNormalization())  # normalize the output of the previous layer

model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # 64 filters, each of size 3x3
model.add(layers.BatchNormalization())  # normalize the output of the previous layer
model.add(layers.MaxPooling2D())  # reduce the size of the output of the previous layer by 2
model.add(layers.Dropout(0.25))  # randomly drop 25% of the output of the previous layer

model.add(layers.Conv2D(128, (3, 3), activation='relu'))  # 128 filters, each of size 3x3
model.add(layers.BatchNormalization())  # normalize the output of the previous layer

model.add(layers.Conv2D(128, (3, 3), activation='relu'))  # 128 filters, each of size 3x3
model.add(layers.BatchNormalization())  # normalize the output of the previous layer
model.add(layers.MaxPooling2D())  # reduce the size of the output of the previous layer by 2
model.add(layers.Dropout(0.25))  # randomly drop 25% of the output of the previous layer

model.add(layers.Conv2D(256, (3, 3), activation='relu'))  # 256 filters, each of size 3x3
model.add(layers.BatchNormalization())  # normalize the output of the previous layer

model.add(layers.Conv2D(256, (3, 3), activation='relu'))  # 256 filters, each of size 3x3
model.add(layers.BatchNormalization())  # normalize the output of the previous layer
model.add(layers.MaxPooling2D())  # reduce the size of the output of the previous layer by 2
model.add(layers.Dropout(0.25))  # randomly drop 25% of the output of the previous layer

model.add(layers.Conv2D(512, (3, 3), activation='relu'))  # 512 filters, each of size 3x3
model.add(layers.BatchNormalization())  # normalize the output of the previous layer

model.add(layers.Conv2D(512, (3, 3), activation='relu'))  # 512 filters, each of size 3x3
model.add(layers.BatchNormalization())  # normalize the output of the previous layer
model.add(layers.MaxPooling2D())  # reduce the size of the output of the previous layer by 2
model.add(layers.Dropout(0.25))  # randomly drop 25% of the output of the previous layer

# Add the flattening layer
model.add(layers.Flatten())  # flatten the output of the previous layer

# Add the dense layers
model.add(layers.Dense(512, activation='relu'))  # 512 neurons
model.add(layers.BatchNormalization())  # normalize the output of the previous layer

model.add(layers.Dense(512, activation='relu'))  # 512 neurons
model.add(layers.BatchNormalization())  # normalize the output of the previous layer
model.add(layers.Dropout(0.4))  # randomly drop 40% of the output of the previous layer

# Add the output layer
model.add(layers.Dense(1, activation='sigmoid'))  # 1 neuron, because we have 2 classes, and we use sigmoid activation for binary classification

# Add the early stopping callback
# We monitor the validation loss, and if it doesn't decrease for 20 epochs, we stop the training
# We also restore the best weights from the best epoch
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Compile the model using the Adam optimizer, binary cross-entropy loss, and accuracy metric
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])

# Train the model
model.fit(
    image_data_generator.flow(train_images, train_labels, batch_size=32),  # use the image data generator to generate batches of images
    epochs=150,  # train for 150 epochs
    callbacks=[early_stopping],  # use the early stopping callback
    validation_data=(validation_images, validation_labels)  # use the 2000 images from the validation set as validation data
)

# Predict the labels of the validation set
val_prediction = model.predict(validation_images)

# Print the classification report and confusion matrix
# Round the predictions to the nearest integer, because they are probabilities
print(metrics.classification_report(validation_labels, np.round(val_prediction)))
print(metrics.confusion_matrix(validation_labels, np.round(val_prediction)))

# Predict the labels of the test set
test_prediction = model.predict(test_images)

# Open a file and write the predictions
with open("submission.csv", "w") as f:
    # write the column names
    f.write("id,class" + "\n")
    # write the predictions
    for i in range(len(test_prediction)):  # for each prediction
        # add 17001 to the index, because the test set starts from 17001
        # round the prediction to the nearest integer, because it is a probability
        # convert the prediction to an integer, because the submission file requires integers
        # we take [0] because the result is a 1D array
        # we use str() to convert the number to string to be able to concatenate it
        f.write(str(i + 17001).zfill(6) + "," + str(np.round(test_prediction[i])[0].astype(int)) + "\n")


# CNN v1
#               precision    recall  f1-score   support
#
#          0.0       0.91      0.96      0.93      1724
#          1.0       0.63      0.39      0.48       276
#
#     accuracy                           0.88      2000
#    macro avg       0.77      0.68      0.71      2000
# weighted avg       0.87      0.88      0.87      2000
#
# [[1659   65]
#  [ 167  109]]


# CNN v2
#               precision    recall  f1-score   support
#
#          0.0       0.93      0.96      0.94      1724
#          1.0       0.68      0.52      0.59       276
#
#     accuracy                           0.90      2000
#    macro avg       0.80      0.74      0.77      2000
# weighted avg       0.89      0.90      0.89      2000
#
# [[1657   67]
#  [ 132  144]]


# CNN v3
#               precision    recall  f1-score   support
#
#          0.0       0.92      0.99      0.95      1724
#          1.0       0.83      0.46      0.59       276
#
#     accuracy                           0.91      2000
#    macro avg       0.88      0.72      0.77      2000
# weighted avg       0.91      0.91      0.90      2000
#
# [[1699   25]
#  [ 150  126]]


# CNN v4
#               precision    recall  f1-score   support
#
#          0.0       0.90      0.99      0.94      1724
#          1.0       0.80      0.31      0.45       276
#
#     accuracy                           0.89      2000
#    macro avg       0.85      0.65      0.69      2000
# weighted avg       0.89      0.89      0.87      2000
#
# [[1702   22]
#  [ 190   86]]


# CNN v5
#               precision    recall  f1-score   support
#
#          0.0       0.94      0.96      0.95      1724
#          1.0       0.70      0.61      0.65       276
#
#     accuracy                           0.91      2000
#    macro avg       0.82      0.78      0.80      2000
# weighted avg       0.91      0.91      0.91      2000
#
# [[1653   71],
# [ 200  76]]


# CNN v6
#              precision    recall  f1-score   support
#
#          0.0       0.94      0.99      0.96      1724
#          1.0       0.88      0.61      0.72       276
#
#     accuracy                           0.93      2000
#    macro avg       0.91      0.80      0.84      2000
# weighted avg       0.93      0.93      0.93      2000
#
# [[1702   22]
#  [ 109  167]]
