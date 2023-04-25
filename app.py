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
from streamlit.components.v1 import html
import streamlit.components.v1 as components
import shap

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")
#en = 3
en = 5

############# HEADER ############# 

st.header('Claim Details')

############# DISPLAY details of claim #############
df = pd.read_csv('insurance_claims.csv')
default_input_df = df[en:en+1]
default_input_df['Details'] = 'Query Details'
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

pre_cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


bools = []
for acat in pre_cat_cols: 
    if set(df[acat])=={'YES','NO'}:
        bools.append(acat)

cat_cols = [a for a in pre_cat_cols if a not in bools]

for abool in bools:
    X[abool]=X[abool].replace('YES', 1)
    X[abool]=X[abool].replace('NO', 0)
    X.rename(columns={abool: abool + ' BOOLEAN'})

# One-hot encode categorical columns

X_processed = pd.get_dummies(X,  prefix_sep = ' CAT: ', columns=cat_cols, drop_first = False)



y_train = list(y[:en])+( list(y[en+1:]))
y_test = list(y[:en])+( list(y[en+1:]))
X_train = X_processed.loc[~X_processed.index.isin([en,])]
X_train = X_train
X_test = pd.DataFrame(X_processed.iloc[en,]).transpose()

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
col2.markdown("<h5 style='text-align: center; color: red;'>%s</h5>"%('Model Output Label: 1'), unsafe_allow_html=True)

col1.markdown("<h2 style='text-align: center; color: green;'>%s</h2>"%(pn), unsafe_allow_html=True)
col1.markdown("<h5 style='text-align: center; color: green;'>%s</h5>"%('Model Output Label: 0'), unsafe_allow_html=True)

##############################################################################

############# Single Observation Explanation ############# 
  

st.markdown('---')
st.markdown("<h1 style='text-align: left; color: Black;'>Local / Single Observation Explanation</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: left; color: Black;'>Local interpretability: explanation is valid for a single observation (i.e., a local explanation) and not for all possible observations (i.e., a global explanation).</h5>", unsafe_allow_html=True)


st.markdown("<h2 style='text-align: center; color: Black;'>LIME</h2>", unsafe_allow_html=True)

############# LIME (LOCAL)
col0, col1, col2 = st.columns([1,3,5])           
            
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['Non-Fraudulent', 'Fraudulent'],
    mode='classification'
)


lime_exp = lime_explainer.explain_instance(
    data_row=X_test.iloc[0,:],
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

with st.expander("**LIME explanation**"):
    st.write('''
        Local Interpretable Model-agnostic Explanations, also known as LIME. \n
        By assuming that that every complex model is linear on a local scale, it is a procedure that uses local models to explain the predictions made. \n
        LIME allows users to understand each components' influences on the prediction score by iteratively 
        removing its components and observe their respective effect on the original score (generated with full set of components). \n
        Observing the bar chart on the right, we can infer the direction and magnitude of impact each component has on the prediction score. \n
    ''')

shap_explainer = shap.TreeExplainer(model)
shap_values = shap_explainer.shap_values(X_test) #apply logistic sigmoid function


############# SHAP (LOCAL)
st.markdown("<h2 style='text-align: center; color: Black;'>SHAP Force Plot</h2>", unsafe_allow_html=True)
shap.initjs()
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
st_shap(shap.force_plot(shap_explainer.expected_value, shap_values[0 ,:], X_test, plot_cmap="RdYlGn"))

with st.expander("**SHAP Force Plot (local) explanation**"):
    st.write('''
        SHAP stands for “Shapley Additive Explanations," a concept derived from game theory and used to explain the output of machine learning models. \n
        In SHAP Force Plot, its score is denoted by f(x). By applying logistic sigmoid function 1/(1+exp(-f(x))) to the score f(x) of value in SHAP Force Plot, probability of Fraudulent %s is obtained. \n
        Higher scores lead the model to predict 1 and lower scores lead the model to predict 0. \n 
        The components of this query that were important to making the prediction for this observation are shown in red and blue, with red representing features that pushed the model score higher, and blue representing features that pushed the score lower. Features that had more of an impact on the score are located closer to the dividing boundary between red and blue, and the size of that impact is represented by the size of the bar.\n
    '''%(probability_F+'%'))
############# SHAP ############# 
#https://medium.com/dataman-in-ai/explain-your-model-with-the-shap-values-bc36aac4de3d
  





############# GLOBAL Explanation ############# 

############# SUMMARY

st.divider()
st.markdown("<h1 style='text-align: left; color: Black;'>Model Explanation</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: left; color: Black;'>Global interpretability: shows how much each component contributes, either positively or negatively, to the target variable.</h5>", unsafe_allow_html=True)





col1, col2 = st.columns([6,4])  

default_ix = list(X_train.columns).index('incident_severity CAT: Major Damage')
hist_col = col1.selectbox('Select Component to view Query Value and Database Distribution', options = X_train.columns, index=default_ix)


col2.markdown("<h2 style='text-align: center; color: Black;'>SHAP Summary Plot</h2>", unsafe_allow_html=True)
#---------- Hist GRAPH C2




col1.divider()



col1.markdown("<h4 style='text-align: left; color: orange;'>%s</h4>"%('''Query Value:'''), unsafe_allow_html=True)



content = list(X_test[hist_col])[0]
is_bool = False
if set(list(X_train[hist_col])) == {0.0, 1.0} or set(list(X_train[hist_col])) == {0, 1}: #if chosen cat is boolean
    is_bool = True
    content = bool(list(X_test[hist_col])[0])

col1.markdown("<h3 style='text-align: center; color: orange;'>%s</h3>"%('''%s'''%( content)), unsafe_allow_html=True)


col1.divider()
col1.markdown("<h3 style='text-align: center; color: grey;'>%s</h3>"%('''%s'''%( 'Database Distribution for')), unsafe_allow_html=True)


if hist_col not in num_cols:
    
    col1a, col1b = col1.columns([1,1])

    plt.subplot(2,3,1)

    p1 = X_train[[hist_col]]

    if is_bool:
        p1[hist_col] = p1[hist_col].replace(0.0, 'False')
        p1[hist_col] = p1[hist_col].replace(1.0, 'True')
        p1[hist_col] = p1[hist_col].replace(0, 'False')
        p1[hist_col] = p1[hist_col].replace(1, 'True')

    try:
        p1.hist(hist_col,color = 'grey')
    except:
        p1[hist_col].value_counts().plot(kind='barh', color = 'grey')        

    col1a.markdown("<h4 style='text-align: left; color: grey;'>%s</h4>"%(hist_col), unsafe_allow_html=True)

    col1a.pyplot()  


    iscat = False
    if ' CAT: ' in hist_col:
        iscat = True
    ori_name = easy_cat(hist_col)
    ori_name= ori_name.split(' CAT: ')[0] 
    ori_name = max(ori_name.split(' '), key=len) 
    plt.subplot(2,3,1)

    if iscat:
        try:
            X.hist(ori_name,color = 'grey')
        except:
            X[ori_name].value_counts().plot(kind='barh', color = 'grey')
        col1b.markdown("<h4 style='text-align: left; color: grey;'>%s</h4>"%(ori_name), unsafe_allow_html=True)

        col1b.pyplot()


else:
    col1a, col1b, col1c = col1.columns([1,2,1])
    plt.subplot(2,3,1)

    p1 = X_train[[hist_col]]

    if is_bool:
        p1[hist_col] = p1[hist_col].replace(0.0, 'False')
        p1[hist_col] = p1[hist_col].replace(1.0, 'True')
        p1[hist_col] = p1[hist_col].replace(0, 'False')
        p1[hist_col] = p1[hist_col].replace(1, 'True')

    try:
        p1.hist(hist_col,color = 'grey')
    except:
        p1[hist_col].value_counts().plot(kind='barh', color = 'grey')        

    col1b.markdown("<h4 style='text-align: left; color: grey;'>%s</h4>"%(hist_col), unsafe_allow_html=True)

    col1b.pyplot()      

#---------- GLOBAL GRAPH C3
shap_values = shap.TreeExplainer(model).shap_values(X_train)
shap.summary_plot(shap_values, X_train)
col2.pyplot()


with st.expander("**SHAP Summary Plot (global) explanation**"):
    st.write('''CatBoostClassifier, a tree-based model that takes care of subspaces (hence non-linearity), is used to predict Fraudulent transactions. Due to this nature, the horizontal color gradient for each component in the summary plot is not expected to be smooth.

- Feature importance: Variables are ranked in descending order. \n
- Impact: The horizontal location shows whether the effect of that value is associated with a higher or lower prediction. \n
- Original value: Color shows whether that component value is high (in red) or low (in blue) for that observation. \n
                ''')




# ############# SCATTERPLOT ############# 
# st.markdown('---')
# st.markdown("<h1 style='text-align: center; color: Black;'>Principal Component Scatterplot</h1>", unsafe_allow_html=True)

# desc = '<em>Principal component analysis, or PCA, is a statistical procedure that summarize the information content in large data tables by means of a smaller set of “summary indices” that can be more easily visualized and analyzed.<em>'

# st.markdown("<p style='text-align: center; color: Black;'>%s</p>"%(desc), unsafe_allow_html=True)

# col1, col2 , col3= st.columns(3)    

# X_knn = X_train[list(X_train.columns)]
# y_knn = pd.DataFrame(y_train)


# x = X_knn.values
# x = StandardScaler().fit_transform(x)


# pca_c = PCA(n_components=2, random_state= 123)
# principalComponents_c = pca_c.fit_transform(x)

# principal__Df = pd.DataFrame(data = principalComponents_c
#              , columns = ['principal component 1', 'principal component 2'])
# y_plot = y_knn[0]
# st.write(y_knn)
# y_plot.replace(0, 'Non_Fraudulent',inplace=True)
# y_plot.replace(1, 'Fraudulent',inplace=True)

# plt.figure()
# plt.figure(figsize=(5,5))
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=14)
# plt.xlabel('Principal Component - 1',fontsize=20)
# plt.ylabel('Principal Component - 2',fontsize=20)
# #plt.title("Principal Component Scatterplot",fontsize=20)
# targets = ['Non_Fraudulent', 'Fraudulent','Query']
# colors = ['g', 'r',]
# for target, color in zip(targets,colors):
#     indicesToKeep = y_plot== target
#     plt.scatter(principal__Df.loc[indicesToKeep, 'principal component 1']
#                , principal__Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 15)
    
# plt.scatter(principal__Df.loc[5, 'principal component 1']
#             , principal__Df.loc[5, 'principal component 2'], c = 'purple', s = 50)

# plt.legend(targets,prop={'size': 5})

# col2.pyplot()

##### Nearest Neighbour

st.divider()
st.markdown("<h1 style='text-align: left; color: Black;'>Most Similiar Database References</h1>", unsafe_allow_html=True)

cola, colb = st.columns([2,5])



neighbours =[136, 258, 837, 461, 814]
distances = [7.10340597, 7.30592397, 7.95593459, 7.96269252, 8.2022564 ]
fraud_neighs = [0, 1, 1, 1, 0]

X = df.drop('fraud_reported', axis = 1)
y = df['fraud_reported']
y = y.map({'Y': 1, 'N': 0})   
y_train = list(y[:en])+( list(y[en+1:]))
y_test = list(y[:en])+( list(y[en+1:]))
X_train = X.loc[~X.index.isin([en,])]
X_train = X_train
X_test = pd.DataFrame(X.iloc[en,]).transpose()
X_test['Details'] = 'Query Details'
X_test = X_test.set_index('Details')
X_test.index.name = None


neighs_table = X_train[X_train.index.isin(neighbours)].reindex(neighbours)
neighs_table.insert(0, 'Similarity to Query', [str(int(1/max(1,a)*100))+'%'  for a in distances])
neighs_table.insert(1, 'Fraudulent',fraud_neighs)
neighs_table.index = ['DB Index '+ str(a) for a in neighs_table.index ]
final_table = pd.concat([neighs_table[neighs_table.columns[2:]], X_test])
colb.write(final_table)


cola.write(neighs_table[neighs_table.columns[:2]])
