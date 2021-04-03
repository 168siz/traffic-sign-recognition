import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy

from keras.models import load_model
import warnings

from classes import classes


def classify(file_path):
    image = Image.open(file_path)
    sign_imp = "Подобрен модел: " + predict(image, 32, model_imp)
    sign_ori = "Оригинален модел: " + predict(image, 30, model_ori)
    value.configure(foreground='black', text=sign_imp)
    value.pack()
    value2.configure(foreground='black', text=sign_ori)
    value2.pack()


def predict(image, size, model):
    image = image.resize((size, size))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    print(image.shape)
    pred = model.predict_classes([image])[0]
    open_sign = classes[pred + 1]
    print(open_sign)
    return open_sign


def show_cb(file_path):
    classify_b = Button(window, text="Класифицирай изображението", command=lambda: classify(file_path), padx=20, pady=5)
    classify_b.configure(background='#757575', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(x=440, y=500)


def uploader():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded = uploaded.resize((128, 128), Image.ANTIALIAS)
        im = ImageTk.PhotoImage(uploaded)
        sign.configure(image=im)
        sign.image = im
        value.configure(text="")
        value2.configure(text="")
        show_cb(file_path)
    except:
        pass


print("Loading model...")
model_imp = load_model('Traffic_signs_model_imp.h5')
model_ori = load_model('Traffic_signs_model.h5')
warnings.filterwarnings("ignore", category=DeprecationWarning)
print("Model loaded.")

window = tk.Tk()
window.geometry('800x600')
window.title('Класификатор на пътни знаци')
window.configure(background='#E0E0E0')

heading = Label(window, text="КЛАСИФИКАТОР НА ПЪТНИ ЗНАЦИ", padx=220, pady=20, font=('Arial', 18, 'bold'))
heading.configure(background='#E0E0E0', foreground='black')
heading.grid()
heading.pack()

sign = Label(window, pady=20)
sign.configure(background='#E0E0E0')
sign.pack()

value = Label(window, font=('Arial', 13), pady=20)
value.configure(background='#E0E0E0')

value2 = Label(window, font=('Arial', 13))
value2.configure(background='#E0E0E0')

upload = Button(window, text="Избери изображение", command=uploader, padx=10, pady=5)
upload.configure(background='#757575', foreground='white', font=('arial', 10, 'bold'))
upload.pack()
upload.place(x=100, y=500)

window.mainloop()
