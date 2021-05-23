import streamlit as st
from src.rna import RNA
import pandas as pd
from copy import copy


def separator(n=1, side=False):
    for i in range(n):
        if not side:
            st.text(' ')
        else:
            st.sidebar.text(' ')


@st.cache
def train_test(path_train, path_test, learning, epochs=200, error=0.1, out='Linear'):
    rna = RNA(rate=float(learning), error_rate=float(error), output_mode=out, epoch=int(epochs))
    rna.train(path=path_train)
    errors = rna.errors
    score, df = rna.test(path=path_test)
    return score, df, errors


# SIDEBAR CONFIG
st.sidebar.header('TRAIN CONFIGURATION')
st.sidebar.text(' ')
rate = 0.3
# output_func = 'Linear'
epoch_value = 100
error_value = 0.5

epoch_value = st.sidebar.text_input(label='Epochs', value=str(epoch_value))
separator(side=True)

error_value = st.sidebar.text_input(label='Error Rate', value=str(error_value))
separator(side=True)

rate = st.sidebar.slider(label='Learning Rate', min_value=0.0000, max_value=1.000, step=0.01, value=rate)
separator(side=True)

output_func = st.sidebar.selectbox(label='Output Function', options=['Linear', 'Logistics', 'Hyperbolic Tangent'], index=2)
separator(side=True)


# MAIN CONFIG
st.header('Multilayer Perceptron - Backpropagation')
separator(n=2)
file_train = st.file_uploader(label='Upload Train', type=['csv'])
if file_train is not None:
    ex = copy(file_train)
    with st.beta_expander('View Train'):
        st.dataframe(pd.read_csv(ex))
separator()

tester = st.file_uploader(label='Upload Test', type=['CSV', 'JPG', 'PNG'])
if tester is not None:
    es = copy(tester)
    with st.beta_expander('View Test'):
        st.dataframe(pd.read_csv(es))
separator()

if st.button('RUN TEST'):
    score, df, errors = train_test(
         path_train=file_train,
         path_test=tester,
         learning=rate,
         epochs=epoch_value,
         error=error_value,
         out=output_func
    )
    st.line_chart(data=errors)
    st.text('SCORE: ' + str(score))
    st.dataframe(data=df)
