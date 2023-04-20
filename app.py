import streamlit as st
import numpy as np
import pandas as pd
import joblib
import catboost
import shap

#st.title("R")
st.header('Claim Details')
### import df and preprocess data
df = pd.read_csv('insurance_claims.csv')
df.replace('?', np.nan, inplace = True) # replacing '?' with 'nan' value
df['collision_type'] = df['collision_type'].fillna(df['collision_type'].mode()[0])
df['property_damage'] = df['property_damage'].fillna(df['property_damage'].mode()[0])
df['police_report_available'] = df['police_report_available'].fillna(df['police_report_available'].mode()[0])
# dropping columns which are not necessary for prediction

to_drop = ['age', 'total_claim_amount','policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date',
    'insured_hobbies','auto_make','auto_model','auto_year', '_c39']

df.drop(to_drop, inplace = True, axis = 1)

X = df.drop('fraud_reported', axis = 1)
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

default_input_df = X[1:2]
default_input_df['Details'] = 'Details'
default_input_df = default_input_df.set_index('Details')
default_input_df.index.name = None

### DISPLAY details of claim ###
st.dataframe(default_input_df)

#
input_file = pd.read_csv('input.csv')


### load model
from_file = catboost.CatBoostClassifier()


model = from_file.load_model('catboostmodel', format='cbm')

### do prediction


def label_maker(ar):
    if bool(ar[0]):
        return 'RESULT: Possibly Fraudulent'
    else:
        return 'RESULT: Possibly Non-Fraudulent'
    

ff = label_maker(model.predict(input_file))
st.subheader(ff)
# shap.summary_plot(shap_values, X
# 
st.write('Scaled Values as model input displayed for attribute effect understanding')
# st.dataframe(input_file)   



X = pd.read_csv('X.csv')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X) 
st.set_option('deprecation.showPyplotGlobalUse', False)

vals= np.abs(shap_values).mean(0)
feature_importance = pd.DataFrame(list(zip(X.columns,vals)),columns=['col_name','feature_importance_vals'])
feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
feature_importance.head(10)
feat_impt = feature_importance.head(5)
feat_impt['feature_importance_vals'] = feat_impt['feature_importance_vals'].apply(lambda x: round(x, 2))
feat_impt.index = feat_impt['col_name']
col_names = list(feat_impt['col_name'])

del feat_impt['col_name']
# X[col_names]
# feat_impt['case input values'] = X[list(feat_impt['col_name'])]
vert = input_file[col_names].transpose()
st.write()
additional = input_file[col_names]
feat_impt['INPUT VALUES'] = vert
st.write(feat_impt)


#summary_plot
shap.summary_plot(shap_values, X, plot_type='violin', max_display = 5)
st.pyplot()

# #dependance_plot
# shap.dependence_plot("LSTAT", shap_values, X)
# st.pyplot()

# ###########################
