import streamlit as st
from streamlit_drawable_canvas import st_canvas


# MAIN CONFIG
st.header('Multilayer Perceptron - Backpropagation')


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
st.sidebar.text(' ')

with st.sidebar.beta_expander('LEARNING RATE'):
    rate = st.slider(label=' ', min_value=0.0000, max_value=1.000, step=0.01, value=0.30)
st.sidebar.text(' ')

with st.sidebar.beta_expander('OUTPUT FUNCTION'):
    output_func = st.selectbox(label='', options=['Linear', 'Logistics', 'Hyperbolic Tangent'])
st.sidebar.text(' ')

with st.sidebar.beta_expander('UPLOAD TRAIN'):
    file = st.file_uploader(label='', type=['csv', 'xlsx'])
st.sidebar.text(' ')
st.sidebar.text(' ')
st.sidebar.button(label='CONFIRM')


