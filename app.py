import streamlit as st
import numpy as np
import pandas as pd
import joblib
import catboost
import shap
import lime
from lime import lime_tabular
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from copy import deepcopy
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")
#en = 3
en = 5

############# HEADER ############# 

st.header('Claim Details')

############# DISPLAY details of claim #############
df = pd.read_csv('insurance_claims.csv')
default_input_df = df[en:en+1]
default_input_df['Details'] = 'Details'
default_input_df = default_input_df.set_index('Details')
default_input_df.index.name = None
default_input_df = default_input_df[default_input_df.columns[:-2]]
st.dataframe(default_input_df)


############# POSSIBILITY OF #############
df = pd.read_csv('insurance_claims.csv')
df.replace('?', np.nan, inplace = True) # replacing '?' with 'nan' value
df['collision_type'] = df['collision_type'].fillna(df['collision_type'].mode()[0])
df['property_damage'] = df['property_damage'].fillna(df['property_damage'].mode()[0])
df['police_report_available'] = df['police_report_available'].fillna(df['police_report_available'].mode()[0])
to_drop = ['age', 'total_claim_amount','policy_number','policy_bind_date','policy_state','insured_zip','incident_location','incident_date',
    'insured_hobbies','auto_make','auto_model','auto_year', '_c39']

df.drop(to_drop, inplace = True, axis = 1)
 
X = df.drop('fraud_reported', axis = 1)
y = df['fraud_reported']
y = y.map({'Y': 1, 'N': 0})   



# Split columns into numerical and categorical
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Standardize numerical columns
#scaler = StandardScaler()
X_processed = deepcopy(df)
#X_processed[num_cols] = scaler.fit_transform(df[num_cols]) 

#one-hot encode cat variables
X_processed = pd.get_dummies(X_processed,  prefix_sep = ' CAT: ', columns=cat_cols, drop_first = False)



X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size = 0.20, random_state=42,stratify=y)

from_file = catboost.CatBoostClassifier()


model = from_file.load_model('catboostmodel', format='cbm')

probabilities = model.predict_proba(X_processed[en:en+1])
probability_non_F = "{:.2f}".format(probabilities[0][0]*100)
probability_F = "{:.2f}".format(probabilities[0][1]*100)

pf = "Fraudulent: "+probability_F +'%'
pn = "Non-Fraudulent: "+probability_non_F + '%'

st.markdown('---')

st.markdown("<h1 style='text-align: center; color: Black;'>Possibility of</h1>", unsafe_allow_html=True)

            
col1, col2 = st.columns(2)           
            
col2.markdown("<h2 style='text-align: center; color: red;'>%s</h2>"%(pf), unsafe_allow_html=True)

col1.markdown("<h2 style='text-align: center; color: green;'>%s</h2>"%(pn), unsafe_allow_html=True)

##############################################################################

############# LIME ############# 
  

st.markdown('---')
st.markdown("<h1 style='text-align: center; color: Black;'>Current Prediction Explanation (LIME)</h1>", unsafe_allow_html=True)

col1, col2 = st.columns(2)           
            
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['Non-Fraudulent', 'Fraudulent'],
    mode='classification'
)


lime_exp = lime_explainer.explain_instance(
    data_row=X_processed.iloc[en],
    predict_fn=model.predict_proba
)

ll = lime_exp.as_list()
y_axis = [i[0] for i in ll]
y_axis.reverse()
x_axis = [i[1] for i in ll]
x_axis.reverse()

lime_table = pd.DataFrame(zip(y_axis, x_axis), columns = ['Variable','Value'])




def easy_cat(colname):
    if ' CAT: ' not in colname:
        return colname
    else:
        if '<= 1.00' in colname:
            mid = re.sub(r"(^|\W)\d+", "", colname)
            colname = re.sub(r"(>|=|<)", "", mid).strip()

            colname = colname +' [TRUE]'
        else:
            mid = re.sub(r"(^|\W)\d+", "", colname)
            colname = re.sub(r"(>|=|<)", "", mid).strip()
            colname = colname +' [FALSE]'
        return colname




lime_table['Variable'] = [easy_cat(acol) for acol in list(lime_table['Variable'])]
lime_table['positive'] = lime_table['Value'] > 0

def inv_trans_num(lime_table):
    for i, row in lime_table.iterrows():
        pass

plt.barh(lime_table['Variable'], lime_table['Value'], color=lime_table['positive'].map({True: 'r', False: 'g'}))
plt.title('Major Entry Contributing Factors')

plt.show()
col2.pyplot()

# st.write(y_axis)
y_axis.reverse()
y_axis= [ele.split(' CAT: ')[0] for ele in y_axis]
y_axis = [max(ele.split(' '), key=len) for ele in y_axis]
# st.write(y_axis)
set_list = []
for anele in y_axis:
    if anele not in set_list:
        set_list.append(anele)
# st.write(set_list)

table_shown_LIME = default_input_df[set_list].transpose()
col1.write(table_shown_LIME)

############# SHAP ############# 
  
from streamlit.components.v1 import html
import streamlit.components.v1 as components


st.markdown('---')
st.markdown("<h1 style='text-align: center; color: Black;'>Model Explanation (SHAP)</h1>", unsafe_allow_html=True)

import shap
shap.initjs()
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

shap_explainer = shap.TreeExplainer(model)
shap_values = shap_explainer.shap_values(X_processed[en:en+1])

st_shap(shap.force_plot(shap_explainer.expected_value, shap_values[:, :], X_processed[en:en+1]))


########################## ############# 

############# SCATTERPLOT ############# 
st.markdown('---')
st.markdown("<h1 style='text-align: center; color: Black;'>Principal Component Scatterplot</h1>", unsafe_allow_html=True)

desc = '<em>Principal component analysis, or PCA, is a statistical procedure that summarize the information content in large data tables by means of a smaller set of “summary indices” that can be more easily visualized and analyzed.<em>'

st.markdown("<p style='text-align: center; color: Black;'>%s</p>"%(desc), unsafe_allow_html=True)

col1, col2 , col3= st.columns(3)    

X_knn = X_processed[list(X_processed.columns)[:-2]]
y_knn = X_processed[list(X_processed.columns)[-2:]]


x = X_knn.values
x = StandardScaler().fit_transform(x)


pca_c = PCA(n_components=2, random_state= 123)
principalComponents_c = pca_c.fit_transform(x)

principal__Df = pd.DataFrame(data = principalComponents_c
             , columns = ['principal component 1', 'principal component 2'])
y_plot = y_knn['fraud_reported CAT: Y']
y_plot.replace(0, 'Non_Fraudulent',inplace=True)
y_plot.replace(1, 'Fraudulent',inplace=True)

plt.figure()
plt.figure(figsize=(5,5))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
#plt.title("Principal Component Scatterplot",fontsize=20)
targets = ['Non_Fraudulent', 'Fraudulent','Query']
colors = ['g', 'r',]
for target, color in zip(targets,colors):
    indicesToKeep = y_plot== target
    plt.scatter(principal__Df.loc[indicesToKeep, 'principal component 1']
               , principal__Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 15)
    
plt.scatter(principal__Df.loc[5, 'principal component 1']
            , principal__Df.loc[5, 'principal component 2'], c = 'purple', s = 50)

plt.legend(targets,prop={'size': 5})

col2.pyplot()

