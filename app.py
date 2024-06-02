import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Category to flower type
category_to_flower = {
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
}

# Set title for the webapp
st.title("Iris Flower Classifier")

# Description
st.markdown('A classifier model to identify the type of iris flower based on the paramters')

st.header("Plant features")
cols = st.columns(2)

with cols[0]:
    st.text("Sepal Features")
    sepal_length = st.slider("Sepal Length (cm)", 1.0, 8.0, 0.5)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.4, 0.5)

with cols[1]:
    st.text("Petal Features")
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 0.5)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.5)

# Leave an exmpty space
st.text('')
if st.button('Predict Flower'):
    # Load the model
    clf = joblib.load('./model.sav')
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = clf.predict_proba(features)
    st.write(prediction)

    pred_label = np.argmax(prediction, axis=-1)
    st.markdown(f':green[The flower is {category_to_flower[pred_label[0]].capitalize()}]')

    prediction_df = pd.DataFrame(
        {
            'classes': list(category_to_flower.values()),
            'probability': prediction[0]
        }
    )

    st.bar_chart(
        data = prediction_df,
        x = 'classes',
        y = 'probability',
        color = 'classes'
    )

    max_prob = max(prediction[0])
    st.markdown(f'#### Graph suggest that the classifier is :blue[{max_prob*100:.2f}%] sure of its decision.')