import joblib
import pandas as pd
import streamlit as st
import numpy as np
import py3Dmol
import requests

models = joblib.load('models.pkl')
genes = joblib.load('genes.pkl')

st.markdown(
    """
    <style>
    /* Overall background */
    .stApp {
        background-color: #5a72fa;
        color: white;
    }

    /* Make text readable on dark background */
    .css-1cpxqw2, .css-1d391kg {
        color: white;
    }

    /* Customize button */
    div.stButton > button {
        background-color: #fc6625;  /* Bright red */
        color: white;
        border-radius: 10px;
        padding: 0.5em 1.5em;
        border: none;
    }

    div.stButton > button:hover {
        background-color: blue;
    }

    /* Customize selectbox (dropdown) border */
    .stExpander > div > div {
        border: 10px solid #000000;  
        border-radius: 8px;
        padding: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.title("Cancer Drug Response Prediction")
st.write("Enter tumor gene expression values to predict drug sensitivity.")

with st.expander('Gene Expression Values'):
    st.write("Please enter the gene expression values for the following genes:")
    st.write("Keep 0 if the gene is not expressed.")
    cols = st.columns(2)
    user_input = {}
    for i, gene in enumerate(genes):
        with cols[i % 2]:
            user_input[gene] = st.slider(f"{gene}", 0.0, 20.0, 0.0, 0.1)

if st.button("Predict"):
    X = pd.DataFrame([user_input])[genes]

    results = []
    for drug, (model, scaler) in models.items():
        prob = model.predict_proba(X)[0][1]
        results.append((drug, prob))

        results.sort(key=lambda x: x[1], reverse=True)
    st.write("Top Predicted Treatments:")
    for drug, score in results[:10]:
        st.write(f"{drug} - Predicted Sensitivity: {score:.2%}")

#FOLDINGGG
with st.expander("See Gene Folding Structure"):
    def show_alphafold_structure(uniprot_id):
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        r = requests.get(url)
        pdb_data = r.text

        st.subheader("3D Structure (AlphaFold Prediction)")
        viewer = py3Dmol.view(width=600, height=500)
        viewer.addModel(pdb_data, "pdb")
        viewer.setStyle({"cartoon": {"color": "spectrum"}})
        viewer.zoomTo()
        st.components.v1.html(viewer._make_html(), height=500)

    gene_to_uniprot = {
        "TP53": "P04637", "EGFR": "P00533", "BRCA1": "P38398", "BRCA2": "P51587", "KRAS": "P01116",
        "PIK3CA": "P42336", "PTEN": "P60484", "ALK": "Q9UM73", "BRAF": "P15056", "MYC": "P01106",
        "CDKN2A": "P42771", "RB1": "P06400", "ARID1A": "O14497", "CTNNB1": "P35222", "ATM": "Q13315",
        "IDH1": "O75874", "IDH2": "P48735", "SMAD4": "Q13485", "NRAS": "P01111", "VHL": "P40337",
        "APC": "P25054", "MDM2": "Q00987", "FGFR1": "P11362", "FGFR2": "P21802", "FGFR3": "P22607",
        "NF1": "P21359", "MET": "P08581", "ERBB2": "P04626", "ERBB3": "P21860", "CDH1": "P12830",
        "NTRK1": "P04629", "NTRK2": "Q16620", "NTRK3": "Q16288", "AKT1": "P31749", "AKT2": "P31751",
        "AKT3": "Q9Y243", "MTOR": "P42345", "PDGFRA": "P16234", "KIT": "P10721", "ROS1": "Q20401",
        "RET": "P07949", "CHEK2": "O96017", "MLH1": "P40692", "MSH2": "P43246", "MSH6": "P52701",
        "TSC1": "Q92574", "TSC2": "P49815", "CDK4": "P11802", "CDK6": "Q00534", "CCND1": "P24385",
        "NOTCH1": "P46531", "NOTCH2": "Q04721", "GNAS": "P84996", "SMARCA4": "P51532", "SMARCB1": "Q12824",
        "STK11": "Q15831", "EZH2": "Q15910", "EP300": "Q09472", "CREBBP": "Q92793", "FOXA1": "P55317",
        "GATA3": "P23771", "MAP2K1": "Q02750", "MAP2K2": "P36507", "HRAS": "P01112", "NFE2L2": "Q16236",
        "POLE": "Q07864", "FANCA": "O15360", "FANCD2": "Q9BXW9", "SUFU": "Q9UMX1", "TERT": "O14746",
        "WT1": "P19544", "ZFHX3": "Q15911", "CHD4": "Q14839", "FAT1": "Q14517", "HNF1A": "P20823",
        "KMT2A": "Q03164", "KMT2C": "Q8NEZ4", "KMT2D": "O14686", "AR": "P10275", "PDCD1": "Q15116",
        "CD274": "Q9NZQ7", "JAK1": "P23458", "JAK2": "O60674", "IL7R": "P16871", "SOCS1": "O15524",
        "STAT3": "P40763", "CXCR4": "P61073", "TNFAIP3": "P21580", "BCL2": "P10415", "BCL6": "P41182",
        "CD79B": "P40259", "TNFRSF14": "Q92956", "TRAF3": "Q13114", "NFKBIA": "P25963", "NFKB2": "Q00653",
        "REL": "Q04864", "FOXP1": "Q9H334", "IKZF1": "Q13422"
    }

    selected_gene = st.selectbox("Select a gene:", sorted(gene_to_uniprot.keys()))

if st.button("Show 3D Fold"):
    uniprot_id = gene_to_uniprot[selected_gene]
    show_alphafold_structure(uniprot_id)