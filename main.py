import fnmatch
import warnings

import numpy as np  # basically an array
from PIL import Image  # manipulate images in python
import os  # directory control
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from model_helper import create_model, predict_with
from plot_helper import plot_sign_dist, plt_confusion_matrix, plot_model_history, plot_occurrences

warnings.filterwarnings("ignore", category=DeprecationWarning)


def resize_images(shuffle=False, img_size=30):
    data = []
    labels = []
    classes = 43

    for i in range(classes):
        img_dir = str(i).zfill(5)
        path = os.path.join(os.getcwd(), 'data', img_dir)
        images = fnmatch.filter(os.listdir(path), '*.ppm')
        if shuffle:
            np.random.shuffle(images)

        for j in images:
            try:
                img_path = path + '\\' + j
                image = Image.open(img_path)
                image = image.resize((img_size, img_size))
                image_rgb = np.array(image)
                data.append(image_rgb)
                labels.append(i)
            except:
                print("Error loading image")
    return data, labels, classes


improved = True
img_size = 32 if improved else 30

print("Resizing images...")
if improved:
    data, labels, classes = resize_images(shuffle=True, img_size=img_size)
else:
    data, labels, classes = resize_images()

print("Splitting data...")
# Converting lists into numpy arrays because its faster and takes lesser memory
X = np.array(data)
y = np.array(labels)
print(X.shape, y.shape)

plot_occurrences(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=68)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print("Running data analysis...")
plot_sign_dist(y_train, y_test)
print("Traffic sign class distribution graphs generated.")

# # Converting the labels into one hot encoding
y_train_enc = to_categorical(y_train, 43)
y_test_enc = to_categorical(y_test, 43)
print(X_train.shape, X_test.shape, y_train_enc.shape, y_test_enc.shape)

model_name = 'Traffic_signs_model_imp.h5' if improved else 'Traffic_signs_model.h5'
model, history = create_model(X_train, y_train_enc, X_test, y_test_enc, improved, model_name=model_name)
plot_model_history(history, improved)
y_pred = predict_with(model, img_size=img_size)
plt_confusion_matrix(model_name, X_test, y_test, improved)

print("Finished.")
quit()
