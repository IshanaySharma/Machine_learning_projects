import numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model

model = load_model('digit_rec.h5')

st.title('Digit Recognizer Streamlit App')

SIZE = 200  # Size of the box where users can draw images(numbers from 1 to 9)
mode = st.checkbox('Draw or Delete', True)
canvas_res = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode='freedraw',
    key='canvas')

if canvas_res.image_data is not None:
    img = cv2.resize(canvas_res.image_data.astype('uint8'),(28,28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('This image will be used as Model input')
    st.image(rescaled)

if st.button('Predict'):
    test_x = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    res = model.predict(test_x.reshape(1,28,28))
    st.write(f'Result is {np.argmax(res[0])}')




