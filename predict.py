import joblib
import shap
import pandas as pd
import tensorflow as tf

with open('Model/scalerf.pkl', 'rb') as f:
    scaler = joblib.load(f)

with open('Data/ex_data.pkl', 'rb') as g:
    ex_data = joblib.load(g)
#ex_data = pd.read_pickle('Data/ex_data.pkl')





def get_prediction(data,model):
    X = scaler.transform(data)
    return X, float(model.predict(X))


def explain_model_prediction(data,nn,features):
    # Calculate Shap values
    #data = pd.DataFrame(data=data, columns=features.tolist(),index=range(1))
    nne = shap.KernelExplainer(nn.predict,ex_data)
    #tf.convert_to_tensor(ex_data, dtype=tf.float32)
    shap_values = nne.shap_values(data)
    #p = shap.force_plot(nne.expected_value, shap_values[0], data)
    p = shap.force_plot(nne.expected_value, shap_values[0][:], data, feature_names =features.tolist(), plot_cmap="PkYg")
    return p, shap_values
