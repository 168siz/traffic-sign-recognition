import csv

import matplotlib.pyplot as plt  # data visualisation
import numpy as np

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

from classes import classes


def plot_model_history(history, improved):
    # plotting graphs for accuracy
    print("Plotting stats...")
    plt.figure(0)
    plt.plot(history.history["accuracy"], label="training accuracy")
    plt.plot(history.history["val_accuracy"], label="val accuracy")
    plt.title("Крива на обучението (подобрен модел)" if improved else "Крива на обучението (оригинален модел)")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    acc_path = "./output/accuracy_graph_imp.png" if improved else "./output/accuracy_graph.png"
    __save(acc_path)
    # plotting graphs for loss
    plt.figure(1)
    plt.plot(history.history["loss"], label="training loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.title("Крива на загубите (подобрен модел)" if improved else "Крива на загубите (оригинален модел)")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    loss_path = "./output/loss_graph_imp.png" if improved else "./output/loss_graph.png"
    __save(loss_path)


def plot_dist(y_train, n_classes, title=None, ax=None, label_x=None, label_y=None, **kwargs):
    """
    Plot the traffic sign class distribution
    """
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(11)
    ax.hist(y_train, np.arange(-0.5, n_classes + 1.5), stacked=True, **kwargs)
    ax.set_xlim(-0.5, n_classes - 0.5)
    if label_x:
        ax.set_xlabel(label_x, fontsize=14)
    if label_y:
        ax.set_ylabel(label_y, fontsize=14)
    if title:
        ax.set_title(title)


def plot_sign_dist(y_train, y_test):
    n_classes = np.unique(y_train).size
    fig, ax = plt.subplots(1, 2, figsize=(19, 10))
    plot_dist(y_train, n_classes, title="Разпределение на пътните знаци за обучение", ax=ax[0],
              label_x="Sign type",
              label_y="Sign count",
              color="blue")
    plot_dist(y_test, n_classes, title="Разпределение на пътните знаци за тест", ax=ax[1],
              label_x="Sign type",
              label_y="Sign count",
              color="red")
    __save("./output/class_distribution.png")


def plot_occurrences(y_train):
    n_classes = np.unique(y_train).size
    classes_list = [sum(y_train == c) for c in range(n_classes)]

    with open("./output/occurrences.csv", "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = ["sign_name", "occurrences"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        n = 1
        writer.writeheader()
        for i in classes_list:
            writer.writerow({"sign_name": classes.get(n), "occurrences": i})
            n += 1


def plt_confusion_matrix(file_name, X_test, y_test, improved):
    print("Loading model...")
    model = load_model(file_name)
    print("Model loaded.")
    y_pred = model.predict_classes(X_test)

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix_custom(cm, improved)
    print("Confusion matrix saved.")


def plot_confusion_matrix_custom(cm, improved):
    cm = [row / sum(row) for row in cm]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    plt.title("Матрица на объркванията (подобрен модел)" if improved else "Матрица на объркванията (оригинален модел)")
    plt.xlabel("Идентификатор на прогнозния клас")
    plt.ylabel("Идентификатор на истинския клас")
    conf_matrix_path = "./output/confusion_matrix_imp.png" if improved else "./output/confusion_matrix.png"
    __save(conf_matrix_path)


def test_plot_model(model, file_path):
    plot_model(model, to_file=file_path, show_shapes=True)


def __save(image_path):
    plt.savefig(image_path)
    plt.close()
    plt.cla()
