import streamlit as st
import pickle
import pandas as pd
import shap
import base64
import streamlit.components.v1 as components
from predict import get_prediction, explain_model_prediction

from keras.models import load_model
nn = load_model('Model/DL_DMSP.h5')

open_file = open("Data/featuresf", "rb")
features = pickle.load(open_file)
open_file.close()


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)



st.set_page_config(page_title="Deep =>  DMSP Particle Precipitate Flux Prediction",
                   page_icon="🌐", layout="wide" )
                   
def main():
    
    st.markdown(
        """
        <style>
        .container {
            height: 400px;
            display: flex;
          
        }
        
        .logo-text {
            font-weight:500 !important;
            font-family:cambria;
            font-size:45px !important;
            color: #444444 !important;
            padding-top: 150px !important;
            padding-left: 75px !important;
            
        }
        .sub-text {
            font-weight:100 !important;
            font-size:12px !important;
            color: #AEB6BF !important;
            padding-bottom: 20px !important;
            padding-right: 20px !important;
            
        }
        .logo-img {
            float:right;
            
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
        <div class="container">
            <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open('Img/globe.gif', "rb").read()).decode()}">
            <p class="logo-text">DMSP Particle Precipitate Flux Prediction</p>
            
        </div>
        """,
        unsafe_allow_html=True
    )
        
    with st.form('prediction_form'):
    
        st.header("Enter the input for following info:")

        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
      
            inp = {}
            inp[features[0]] = st.number_input(features[0].replace('_',' '), value = 0.35000, format="%.5f")
            inp[features[1]] = st.number_input(features[1].replace('_',' '), value = 250.00000, format="%.5f", step=1.0)
            inp[features[2]] = st.number_input(features[2].replace('_',' '), value = 0.25000, format="%.5f")
            inp[features[3]] = st.number_input(features[3].replace('_',' '), value = 6.00, format="%.1f")
            inp[features[4]] = st.number_input(features[4].replace('_',' '), value = -0.25500, format="%.5f")
            
        with col2:    
            inp[features[5]] = st.number_input(features[5].replace('_',' '), value = 0.20000, format="%.5f")
            inp[features[6]] = st.number_input(features[6].replace('_',' '), value = 68.9, format="%.1f", step=1.0)
            inp[features[7]] = st.number_input(features[7].replace('_',' '), value = 68.9, format="%.1f", step=1.0)
            inp[features[8]] = st.number_input(features[8].replace('_',' '), value = 68.9, format="%.1f", step=1.0)
            inp[features[9]] = st.number_input(features[9].replace('_',' '), value = 1.20000, format="%.5f")
            
        with col3:
      
            inp[features[10]] = st.number_input(features[10].replace('_',' '), value = 68.9, format="%.1f", step=1.0)
            inp[features[11]] = st.number_input(features[11].replace('_',' '), value = 5.02000, format="%.5f")
            inp[features[12]] = st.number_input(features[12].replace('_',' '), value = 68.9, format="%.1f", step=1.0)
            inp[features[13]] = st.number_input(features[13].replace('_',' '), value = 5.75, format="%.2f")
            inp[features[14]] = st.number_input(features[14].replace('_',' '), value = 68.9, format="%.1f", step=1.0)
            
        with col4:    
            inp[features[15]] = st.number_input(features[15].replace('_',' '), value = 7.41, format="%.2f", step=0.1)
            inp[features[16]] = st.number_input(features[16].replace('_',' '), value = 68.9, format="%.1f", step=1.0)
            inp[features[17]] = st.number_input(features[17].replace('_',' '), value = 68.9, format="%.1f", step=1.0)
            inp[features[18]] = st.number_input(features[18].replace('_',' '), value = 68.9, format="%.1f", step=1.0)
            datef = st.date_input(features[19].replace('_',' '))
        

        submit = st.form_submit_button("Estimate Precipitate Flux")

    if submit:
        inp[features[19]]=datef.strftime('%y%m%d')
        df = pd.DataFrame.from_dict([inp])
        X, pred = get_prediction(data=df, model=nn)

        st.markdown("""<style> .big-font { font-family:sans-serif; color: #1D7AA7 ; font-size: 30px; } </style> """, unsafe_allow_html=True)
        st.markdown(f'<p class="big-font">Capability of electron particle precipitation from the magnetosphere to the ionosphere predicted is { "{:.3e}".format(pred) }.</p>', unsafe_allow_html=True)
     

        p, shap_values = explain_model_prediction(X,nn,features)
        
        st.subheader('Extent of factors affecting Precipitate Flux')
        st_shap(p)
    

if __name__ == '__main__':
    main()