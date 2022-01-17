#!/usr/bin/env python
# coding: utf-8

# # Introduction
#
# - The process of selecting a subset of relevant features for use in model construction
# - Is different from dimensionality reduction. Both methods seek to reduce the number of attributes in the dataset, but a dimensionality reduction method do so by creating new combinations of attributes, where as feature selection methods include and exclude attributes present in the data without changing them.
# -  Used to identify and remove unneeded, irrelevant and redundant attributes from data that do not contribute to the accuracy of a predictive model or may in fact decrease the accuracy of the model.
#
# - **OBJECTIVES**: improving the prediction performance of the predictors, providing faster and more cost-effective predictors, and providing a better understanding of the underlying process that generated the data.

# ![alt text](Feature_Selection_Techniques.png)
#
# *https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/*

#------------------------------------------------
# FUNCTIONS


def select_columns(coeff, threshold):
    cols = []
    for i in range(len(coeff)):
        value = coeff.iloc[i,0]
        if (value > threshold):
            cols.append(coeff.index[i])

    return(cols)

#---------

# #### Information Gain (Numeric Input, Numeric Output)
def func_information_gain(X_num,y,threshold_IG=0.02):
    from sklearn.feature_selection import mutual_info_classif

    # Define the threshold
    threshold_IG = 0.02

    # Calculate the mutual information coefficients and convert them to a data frame
    coeff_IG =pd.DataFrame(mutual_info_classif(X_num, y).reshape(-1, 1),
                             columns=['Coefficient'], index=X_num.columns)

    print('Information Gain')
    print(coeff_IG)

    # Only keep columns whose information gain is higer than the threshold
    cols_IG = select_columns(coeff_IG, threshold_IG)

    print('\nColumns to remain in the DF: ', cols_IG)

    return(cols_IG)

#---------
# #### Pearson's Correlation (Linear Correlation) (Numeric Input, Numeric Output)
def func_peason_corr(df_num,threshold_Pe = 0.09):
    # Calculate the correlation matrix
    corr_mat =  df_num.corr(method = 'pearson')

    # Select only those values related to the target
    coeff_Pe = corr_mat[target].sort_values(ascending = False)[1:] #discard the first one since it is the target itself

    coeff_Pe = pd.DataFrame(coeff_Pe.values, columns=['Coefficient'], index = coeff_Pe.index )

    print('Pearsons Correlation')
    print(abs(coeff_Pe))

    # Only keep columns whose ABSOLUTE value of the correlation is higer than the threshold
    cols_Pe = select_columns(abs(coeff_Pe), threshold_Pe)

    print('\nColumns to remain in the DF: ', cols_Pe)

    return(cols_Pe)

#---------
# #### Spearman’s Correlation (Nonlinear Correlation)(Numeric Input, Numeric Output)
def func_sperman_corr(df_num,threshold_Sp = 0.09):
    # Calculate the correlation matrix
    corr_mat =  df_num.corr(method = 'spearman')

    # Select only those values related to the target
    coeff_Sp = corr_mat[target].sort_values(ascending = False)[1:] #discard the first one since it is the target itself

    coeff_Sp = pd.DataFrame(coeff_Sp.values, columns=['Coefficient'], index = coeff_Sp.index )

    print('Spearmans Correlation')
    print(abs(coeff_Sp))

    # Only keep columns whose ABSOLUTE value of the correlation is higer than the threshold
    cols_Sp = select_columns(abs(coeff_Sp), threshold_Sp)

    print('\nColumns to remain in the DF: ', cols_Sp)

    return(cols_Sp)

#---------
# #### Kendall’s Tau (Numeric Input, Categorical Output) OR (Categorical Input, Numeric Output)
def func_kendalls_tau(X,y,threshold_Ke = 0.13,threshold_Ke_pvalue = 0.05):
    from scipy import stats

    corr_tau = []
    p_value = []

    for i in range(len(X.columns)):
        tau, p_val = stats.kendalltau(X.iloc[:,i], y)
        corr_tau.append(abs(tau))
        p_value.append(p_val)

    corr_tau = pd.DataFrame(corr_tau, columns = ["Coefficient"], index = X.columns)
    p_value = pd.DataFrame(p_value, columns = ["p_value"], index = X.columns)

    print("Correlaton")
    print(corr_tau)
    print("\nP-values")
    print(p_value)

    # Only select the statistically significant measures
    corr_tau = corr_tau[(p_value["p_value"] < threshold_Ke_pvalue).values]

    print('\nStatistically significant Kendall’s Tau')
    print(corr_tau)

    # Only keep columns whose ABSOLUTE value of the correlation is higer than the threshold
    cols_Ke = select_columns(corr_tau, threshold_Ke)

    print('\nColumns to remain in the DF: ', cols_Ke)

    return(cols_Ke)


#---------
# #### Chi-Squared Test (Categorial (ANY) Input, Categorical (ANY) Output)
def func_chi2(X_cat,X_num,y,alpha=0.05):

    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_selection import chi2


    # Null hypothesis H0: target is independent from the features.
    # H0 is REJECTED if p_value <= alpha.

    cols_Chi = []

    # Before performig Chi-Square test we have to make sure data is label encoded.
    label_encoder = LabelEncoder()
    X_chi_cat = X_cat.copy()
    for col in X_chi_cat.columns:
        X_chi_cat[col] = label_encoder.fit_transform(X_chi_cat[col])

    # X_chi is composed of all features. categorical and numeric.
    X_chi = pd.concat([X_chi_cat, X_num], axis =1)

    #print(X_chi.head(2))

    # y can be categorical or numeric
    chi_values, p_values = chi2(X_chi,y)
    print("\n Chi Square values: ", chi_values)
    print("\n p-values: ", p_values)

    # test if target is DEPENDENT of the features. True when p_value <= alpha
    for i in range(len(p_values)):
        if (p_values[i] <= alpha):
            cols_Chi.append(X_chi.columns[i])

    print('\nColumns to remain in the DF: ', cols_Chi)

    return(cols_Chi)


#---------
# #### Recursive Feature Elimination (Numerical Input, Numerical Output)
# First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through any specific attribute or callable. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.

def func_rfe(X_num, y,classification_problem,frac_features_keep=0.7):

    from sklearn.feature_selection import RFE
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    n_features = int(frac_features_keep*len(X_num.columns))

    if (classification_problem==1):
        estimator = DecisionTreeClassifier(min_samples_leaf=4)
    else:
        estimator = DecisionTreeRegressor(min_samples_leaf=4)

    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    selector = selector.fit(X_num, y)

    cols_RFE = list(X_num.columns[selector.support_])

    print('Columns to remain in the DF: ', cols_RFE)

    return(cols_RFE)


#---------
# #### ElasticNet (Numerical Input, Numerical Output)

def func_elasticnet(X_num, y, threshold_El = 0):
    from sklearn.linear_model import ElasticNet

    model = ElasticNet()
    model.fit(X_num, y)

    feature_importances = pd.DataFrame(np.abs(model.coef_), columns = ['Coefficient'], index = X_num.columns)

    print('Feature Importances')
    print(feature_importances)

    cols_El = select_columns(feature_importances, threshold_El)

    print('\nColumns to remain in the DF: ', cols_El)

    return(cols_El)


#---------
# #### Decision Trees (Numerical Input, Numerical Output)

def func_decision_tree(classification_problem,X_num,y,threshold_Tr = 0.11):
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

    if (classification_problem==1):
        model = ExtraTreesClassifier(min_samples_leaf=4)
    else:
        model = ExtraTressRegressor(min_samples_leaf=4)

    model.fit(X_num, y)

    feature_importances = pd.DataFrame(model.feature_importances_, columns = ['Coefficient'], index = X_num.columns)

    print('Feature Importances')
    print(feature_importances)


    cols_Tr = select_columns(feature_importances, threshold_Tr)

    print('\nColumns to remain in the DF: ', cols_Tr)

    return(cols_Tr)


#---------
# #### Random Forest (Numerical Input, Numerical Output)

def func_random_forest(classification_problem,X_num,y,threshold_RF = 0.11):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if (classification_problem==1):
        model = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=200, n_jobs=-1)

    model.fit(X_num, y)

    feature_importances = pd.DataFrame(model.feature_importances_, columns = ['Coefficient'], index = X_num.columns)

    print('Feature Importances')
    print(feature_importances)


    cols_RF = select_columns(feature_importances, threshold_RF)

    print('\nColumns to remain in the DF: ', cols_RF)

    return(cols_RF)



# # Imports, Parameters, and Functions
import pandas as pd
import numpy as np
import seaborn as sns


# In[3]:


# Will do this just once to create the json file
#df1 = pd.read_csv('../Data/titanic_train.csv')
#df1.to_json(r'../Data/titanic_train.json')
#df2 = pd.read_csv('../Data/titanic_test.csv')
#df2.to_json(r'../Data/titanic_test.json')


# Classification (1) or Regression (0)
classification_problem = 1

# Define the name of the target column
target = 'Survived'


#Load the Data
#df = pd.read_csv('../Data/titanic_train.csv')
df = pd.read_json('../Data/titanic_train.json')

# Remove observations with missing values
df.dropna(inplace=True)

df.head(2)

# ## Split the features from the target

X = df.drop(target, axis = 1)
y = df[target]



X.info()

y.head()


# Create a categorical target just to simulate this kind of output
y_cat = y.apply(lambda x: 'N' if (x==0) else 'Y')
y_cat.head()



# ## Split Numerical and Categorical Inputs

X_num = X.select_dtypes(include=[np.number])
X_cat = X.select_dtypes(exclude=[np.number])

df_num = df.select_dtypes(include=[np.number])
df_cat = df.select_dtypes(exclude=[np.number])


X_num.info()


X_cat.info()



print(X_num.shape)
print(y.shape)


# # Feature Selection Algorithms: Filter, Wrapper and Intrinsic Methods
# https://machinelearningmastery.com/an-introduction-to-feature-selection/
# https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
# https://machinelearningmastery.com/feature-selection-machine-learning-python/


'''
#------------------------------------------------------------------
# ## FILTER:
#------------------------------------------------------------------
'''
# - Methods apply a statistical measure to assign a scoring to each feature. The features are ranked by the score and either selected to be kept or removed from the dataset.
# - Methods: Information Gain, Pearson’s Correlation, Spearman’s Correlation, Feature Importance, Kendall's Tau

# **Some univariate statistical measures that can be used for FILTER-based feature selection.**
# ![alt text](Filter_Based_Feature_Selection.png)
#
# *https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/*



# ### NUMERICAL Input, NUMERICAL Output
# #### Information Gain

# Define the threshold
threshold = 0.02
selected_cols = func_information_gain(X_num,y,threshold)


# #### Pearson's Correlation (Linear Correlation)

# Define the threshold
threshold = 0.09
selected_cols = func_peason_corr(df_num,threshold)


# #### Spearman’s Correlation (Nonlinear Correlation)


# Define the threshold
threshold = 0.09
selected_cols = func_sperman_corr(df_num,threshold)



# ### NUMERICAL Input, CATEGORICAL Output
# OR
# ### CATEGORICAL Input, NUMERICAL Output

# #### Kendall’s Tau
#
# - Kendall’s tau is a measure of the correspondence between two rankings. Values close to 1 indicate strong agreement, and values close to -1 indicate strong disagreement.

threshold = 0.13
alpha = 0.05 #significance level

# X_num, y_cat
selected_cols = func_kendalls_tau(X_num,y_cat,threshold,alpha)

# X_cat, y_num
selected_cols = func_kendalls_tau(X_cat,y,threshold,alpha)





# ### Categorial (ANY) Input, Categorical (ANY) Output

# #### Chi-Squared Test
# https://towardsdatascience.com/chi-square-test-for-feature-selection-in-machine-learning-206b1f0b8223

# In[64]:

alpha = 0.05 #significance level
selected_cols = func_chi2(X_cat,X_num,y,alpha)




'''
#------------------------------------------------------------------
# ## WRAPPER:
#------------------------------------------------------------------
'''

# - Consider the selection of a set of features as a search problem, where different combinations are prepared, evaluated and compared to other combinations. A predictive model is used to evaluate a combination of features and assign a score based on model accuracy.
# - Method: Recursive Feature Elimination.

# ### NUMERICAL  Input, NUMERICAL Output

# #### Recursive Feature Elimination

frac_features_keep=0.7
selected_cols = func_rfe(X_num, y,classification_problem,frac_features_keep)




'''
#------------------------------------------------------------------
# ## INTRINSIC - Feature Importances:
#------------------------------------------------------------------
'''

# - There are some machine learning algorithms that perform feature selection automatically as part of learning the model.
# - This includes algorithms such as penalized regression models like Lasso and decision trees, including ensembles of decision trees like random forest.
# - Method: Elastic Net, Decision Trees, Random Forest.

# ### NUMERICAL Input, NUMERICAL Output

# #### ElasticNet

threshold = 0

selected_cols = func_elasticnet(X_num, y,threshold)




# #### Decision Trees
threshold = 0.11

selected_cols = func_decision_tree(classification_problem,X_num,y,threshold)




# #### Random Forest
threshold = 0.11

selected_cols = func_random_forest(classification_problem,X_num,y,threshold)





# # Save the final data frame with the selected columns as a json file
# Still have to decide which algorithm to use here. Saving the last one.


df[selected_cols].to_json(r'../Data/titanic_train_feature_selection.json')
