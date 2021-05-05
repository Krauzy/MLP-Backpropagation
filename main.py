import streamlit as st
import pandas as pd
import numpy as np
from streamlit_drawable_canvas import st_canvas


def separator(n=1, side=False):
    for i in range(n):
        if not side:
            st.text(' ')
        else:
            st.sidebar.text(' ')


# MAIN CONFIG
st.header('Multilayer Perceptron - Backpropagation')
separator()
with st.beta_expander('MNIST'):
    st.file_uploader(label='Upload Image', type=['PNG', 'JPG', 'JPEG'])
    st.text('OR')
    st.markdown('#### Draw the number')
    canvas = st_canvas(
        fill_color='',
        stroke_width=5,
        stroke_color='#FFD100',
        background_color='#202020',
        background_image=None,
        update_streamlit=True,
        height=400,
        drawing_mode='freedraw'
    )
    st.button('RUN PREDICTION')

separator(n=2)
chart = pd.DataFrame(np.random.randn(20, 6), columns=['score', 'a', 'b', 'c', 'd', 'e'])
st.line_chart(chart['score'])
st.dataframe(chart)


# SIDEBAR CONFIG
st.sidebar.header('TRAIN CONFIGURATION')
st.sidebar.text(' ')

with st.sidebar.beta_expander('STOP MODE'):
    side1, side2 = st.beta_columns([1, 1])
    stop_mode = side1.selectbox(label='Mode', options=['Epoch', 'Error'])
    if stop_mode == 'Epoch':
        stop_mode_value = side2.text_input(label='Value', value='2000')
    else:
        stop_mode_value = side2.text_input(label='Value', value='0.0001')
separator(side=True)

with st.sidebar.beta_expander('LEARNING RATE'):
    rate = st.slider(label=' ', min_value=0.0000, max_value=1.000, step=0.01, value=0.30)
separator(side=True)

with st.sidebar.beta_expander('OUTPUT FUNCTION'):
    output_func = st.selectbox(label='', options=['Linear', 'Logistics', 'Hyperbolic Tangent'])
separator(side=True)

with st.sidebar.beta_expander('UPLOAD TRAIN'):
    file = st.file_uploader(label='', type=['csv', 'xlsx'])
st.sidebar.text(' ')
st.sidebar.text(' ')
st.sidebar.button(label='CONFIRM')
