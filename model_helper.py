import pandas as pd  # reading and analyze csv
import numpy as np  # basically an array
from PIL import Image  # manipulate images in python
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, LeakyReLU, BatchNormalization
import warnings

from plot_helper import test_plot_model


def create_model(X_train, y_train, X_test, y_test, improved, model_name):
    if improved:
        return create_model_improved(X_train, y_train, X_test, y_test, model_name)
    else:
        return create_model_original(X_train, y_train, X_test, y_test, model_name)


def create_model_improved(X_train, y_train, X_test, y_test, model_name):
    print("Building improved model...")
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.5))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(BatchNormalization())  # 0.9555819477434679
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))  # 16-32-64-128 0.9614410134600159
    model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))  # 32x32 img resize 0.9704671417260491
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))
    # Final training of model

    print("Saving model...")
    model.save(model_name)
    print("Model saved.")
    print("Generating architecture...")
    file_path = "./output/model_architecture_imp.png"
    test_plot_model(model, file_path)

    return model, history


def create_model_original(X_train, y_train, X_test, y_test, model_name="Traffic_signs_model.h5"):
    print("Building original model...")
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))
    # Final training of model
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print("Saving model...")
    model.save(model_name)
    print("Model saved.")
    file_path = "./output/model_architecture.png"
    test_plot_model(model, file_path)
    return model, history


def predict_with(model, img_size=30):
    y_test = pd.read_csv('test.csv')
    labels = y_test["ClassId"].values
    imgs = y_test["Path"].values

    data = []

    for img in imgs:
        image = Image.open(img)
        image = image.resize((img_size, img_size))
        data.append(np.array(image))

    X_test = np.array(data)

    y_pred = model.predict_classes(X_test)
    # Accuracy with the img_png data
    print("accuracy score:", accuracy_score(labels, y_pred))
    return y_pred
