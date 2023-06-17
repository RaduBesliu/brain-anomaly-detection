# Name: Besliu Radu-Stefan
# Group: 243
# File: KNN.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report


# Store the training images in a numpy array and save it to a *.npy file
def read_and_save_train_images():
    tr_images = []
    # read png images from 1 to 15000 and append to tr_images list
    for i in range(1, 15001):
        # images are in the format: "000001.png", "000002.png", ..., "015000.png"
        # that's why we need to use zfill(6) to add zeros to the left of the number until it has 6 digits
        tr_images.append(plt.imread("data/data/" + str(i).zfill(6) + ".png"))

    # save tr_images list to a numpy array
    np.save("data_npy/train_images.npy", tr_images)


# Store the training labels in a numpy array and save it to a *.npy file
def read_and_save_train_labels():
    # open the file and read the labels
    with open("data/train_labels.txt", "r") as f:
        # remove the first line of the file, because it contains the column names ( id, class )
        f.readline()
        # load the labels from the file, split the by "," and save only the second column (the labels)
        np.save("data_npy/train_labels.npy", np.loadtxt(f, delimiter=",", usecols=(1,)))


# Store the validation images in a numpy array and save it to a *.npy file
def read_and_save_validation_images():
    v_images = []
    # read png images from 15001 to 17000 and append to v_images list
    for i in range(15001, 17001):
        # images are in the format: "015001.png", "015002.png", ..., "017000.png"
        # that's why we need to use zfill(6) to add zeros to the left of the number until it has 6 digits
        v_images.append(plt.imread("data/data/" + str(i).zfill(6) + ".png"))

    # save v_images list to a numpy array
    np.save("data_npy/validation_images.npy", v_images)


# Store the validation labels in a numpy array and save it to a *.npy file
def read_and_save_validation_labels():
    # open the file and read the labels
    with open("data/validation_labels.txt", "r") as f:
        # remove the first line of the file, because it contains the column names ( id, class )
        f.readline()
        # load the labels from the file, split the by "," and save only the second column (the labels)
        np.save("data_npy/validation_labels.npy", np.loadtxt(f, delimiter=",", usecols=(1,)))


# Store the test images in a numpy array and save it to a *.npy file
def read_and_save_test_images():
    te_images = []
    # read png images from 17001 to 22149 and append to te_images list
    for i in range(17001, 22150):
        # images are in the format: "017001.png", "017002.png", ..., "022149.png"
        te_images.append(plt.imread("data/data/" + str(i).zfill(6) + ".png"))

    # save te_images list to a numpy array
    np.save("data_npy/test_images.npy", te_images)


# Data read and save to improve performance ( only run once )
# read_save_data.read_and_save_train_images()
# read_save_data.read_and_save_train_labels()
# read_save_data.read_and_save_validation_images()
# read_save_data.read_and_save_validation_labels()
# read_save_data.read_and_save_test_images()


# Load data from numpy arrays
train_images = np.load("data_npy/train_images.npy")
train_labels = np.load("data_npy/train_labels.npy")
validation_images = np.load("data_npy/validation_images.npy")
validation_labels = np.load("data_npy/validation_labels.npy")
test_images = np.load("data_npy/test_images.npy")

# Data preprocessing
# Reshape the images to 1D array
# The images are flattened to a 1D array, improving the computation time and accuracy
train_images = train_images.reshape(train_images.shape[0], -1)
validation_images = validation_images.reshape(validation_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

# KNN Model
# knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
# knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")

# Fit the model
knn.fit(train_images, train_labels)

# Predict the validation labels
val_prediction = knn.predict(validation_images)

# Round the predictions
val_prediction = np.round(val_prediction)

# Confusion matrix and classification report
cm = confusion_matrix(validation_labels, val_prediction)
cr = classification_report(validation_labels, val_prediction)
print('Confusion Matrix')
print(cm)
print('Classification Report')
print(cr)

# Predict the test labels
test_prediction = knn.predict(test_images)

# Open a file and write the predictions
with open("submission.csv", "w") as f:
    # write the column names
    f.write("id,class" + "\n")
    # write the predictions
    for i in range(len(val_prediction)):
        # images are in the format: "017001.png", "017002.png", ..., "022149.png"
        # that's why we need to use zfill(6) to add zeros to the left of the number until it has 6 digits
        # we use int() to remove the decimal point
        # and str() to convert the number to string to be able to concatenate it
        f.write(str(i + 17001).zfill(6) + "," + str(int(val_prediction[i])) + "\n")


# n_neighbors = 5
# Confusion Matrix
# [[1694   30]
#  [ 251   25]]
# Classification Report
#               precision    recall  f1-score   support
#
#          0.0       0.87      0.98      0.92      1724
#          1.0       0.45      0.09      0.15       276
#
#     accuracy                           0.86      2000
#    macro avg       0.66      0.54      0.54      2000
# weighted avg       0.81      0.86      0.82      2000


# n_neighbors = 3
# Confusion Matrix
# [[1658   66]
#  [ 242   34]]
# Classification Report
#               precision    recall  f1-score   support
#
#          0.0       0.87      0.96      0.92      1724
#          1.0       0.34      0.12      0.18       276
#
#     accuracy                           0.85      2000
#    macro avg       0.61      0.54      0.55      2000
# weighted avg       0.80      0.85      0.81      2000


# n_neighbors = 1
# [[1591  133]
#  [ 211   65]]
# Classification Report
#               precision    recall  f1-score   support
#
#          0.0       0.88      0.92      0.90      1724
#          1.0       0.33      0.24      0.27       276
#
#     accuracy                           0.83      2000
#    macro avg       0.61      0.58      0.59      2000
# weighted avg       0.81      0.83      0.82      2000
