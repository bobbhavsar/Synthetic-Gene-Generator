import joblib
import pandas as pd
import streamlit as st
import numpy as np
models = joblib.load('models.pkl')
genes = joblib.load('genes.pkl')

st.title("Cancer Drug Response Prediction")
st.write("Enter tumor gene expression values to predict drug sensitivity.")

user_input = {}
for gene in genes:
    user_input[gene] = st.number_input(f"{gene} expression (log2 TPM)", min_value=0.0, max_value=20.0, step=0.1, value=7.0)

if st.button("Predict"):
    X = pd.DataFrame([user_input])[genes]

    results = []
    for drug, model in models.items():
        prob = model.predict_proba(X)[0][1]
        results.append((drug, prob))

        results.sort(key=lambda x: x[1], reverse=True)
    st.subheader("Top Predicted Treatments:")
    for drug, score in results[:10]:
        st.write(f"**{drug}** - Predicted Sensitivity: {score:.2%}")