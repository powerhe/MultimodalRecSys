from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np


def predict_quality(model, df):

    predictions_data = predict_model(estimator = model, data = df)
    return predictions_data['Label'][0]

model = load_model('extra_tree_model')


st.title('Wine Quality Classifier Web App')
st.write('This is a web app to classify the quality of your wine based on\
         several features that you can see in the sidebar. Please adjust the\
         value of each feature. After that, click on the Predict button at the bottom to\
         see the prediction of the classifier.')
