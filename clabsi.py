#!/usr/bin/env python
# coding: utf-8

# ### Capstone Practicum Project course II
# 
# 
# 
# 
# 

# # CLABSI Research Project Analysis

# ### CRISP-DM

# CRISP-DM (Cross-Industry Standard Process for Data Mining) is a process model that serves as the base for a data science process.
# 
# CRISP-DM has six sequential phases:
# 1. Business Understanding Phase
# 1. Data Understanding Phase
# 1. Data Preparation Phase
# 1. Modeling Phase
# 1. Evaluation Phase
# 1. Deployment Phase

# #### Structure of CRISP-DM

# <img src='CRISP-DM.png'></img>

# # CLABSI Research Study of Papers 1, 2 and 3

#  

# 

# 

# # Research/Business Understanding

# In[ ]:





# In[ ]:





# # Data Understanding

# #### Importing packages

# In[24]:


import pandas as pd
import numpy as np
from numpy import NaN as NA
import numpy.random as random
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import histplot as hist
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest
from time import time
from datetime import datetime
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, roc_curve, precision_score, f1_score
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


# inline plot
get_ipython().run_line_magic('matplotlib', 'inline')

# seaborn style
sns.set_style('ticks')


# In[25]:


np.random.seed(8)
random_state = 8


# #### Importing/Loading the csv file/dataset

# In[26]:


# Loading the data file
df_clabsi = pd.read_csv('bzan6361_clabsi.csv') 


# In[27]:


# Displaying four records

pd.set_option('display.max_columns',None)
pd.set_option("max_colwidth", None)
df_clabsi.head(4)


# #### Quick overview of data

# In[28]:


# The shape of the dataset and the target variable with true instances

print(f'The dataset has {df_clabsi.shape[0]} rows and {df_clabsi.shape[1]} columns.')
print('='*40)
print('With the target variable HasCLABSI displaying')
print((df_clabsi.HasCLABSI==1).sum(), f'instances as TRUE out of {df_clabsi.shape[0]} instances.')


# In[29]:


# Initial review of the dataset

df_clabsi.info()


# In[30]:


# The dtypes of the dataset

pd.set_option('display.max_rows',None)
print('The dataset has following columns with respective dtypes:')
print()
print(df_clabsi.dtypes)


# In[31]:


# Missing values in the given dataset

print(f'There are {df_clabsi.isnull().sum().sum()} missing values in the dataset')
print('='*50)
print('With the following columns displaying number of missing values:')
print()
print(df_clabsi.isnull().sum())


# #### `Note`:
# 
# The given CLABSI dataset has 14236 rows and 278 columns with target variable 'HasCLABSI'. The dataset includes 19 variables as bool, 19 as float64, 208 as int64 and 32 as object variables datatypes. There are a total of 239605 missing values.

# # Data Preprocessing
# 
# Data preprocessing is being done to significantly understand and resolve the restrictions and challenges associated with the given CLABSI dataset and to help in analysing the opportunities in the EDA process and support the completion of project to attain the required business objective.  

# #### Replacing missing values with 0

# In[32]:


# Replacing missing values with '0'
df_clabsi = df_clabsi.fillna(0)


# #### Sanity check

# In[33]:


# Performing sanity check after replacing missing values with '0'
df_clabsi.isnull().sum().sum()


# ### `Note`:
# 
# The following columns had missing values in the range 9038 to 13604 before replacing these missing values with zero: 
# 
# 1. ClabsiDTS, 
# 1. DaysToCLABSI, 
# 1. MostRecentClabsiDTS, 
# 1. DiagnosisLast15, 
# 1. DiagnosisLast5, 
# 1. DiagnosisLast3, 
# 1. DiagnosisLast2, 
# 1. SurgeriesLast15, 
# 1. SurgeriesLast5, 
# 1. SurgeriesLast3, 
# 1. SurgeriesLast2 
# 
#  
# These huge number of missing values in the dataset will clearly lead to a loss of information, reduce the statistical power and adds ambiguity to the analysis process. Moreover, this lack of data may introduce selection bias, which could potentially invalidate the entire study leading to biased results.
# 
# Therefore, dropping the above mentioned columns would be significant in overcoming the restrictions and challenges associated with the given CLABSI dataset.

# ### Dropping the following columns from the dataset : 
# 
# ClabsiDTS, DaysToCLABSI, MostRecentClabsiDTS, DiagnosisLast15, DiagnosisLast5, DiagnosisLast3, DiagnosisLast2, SurgeriesLast15, SurgeriesLast5, SurgeriesLast3, SurgeriesLast2

# In[34]:


df_clabsi = df_clabsi.drop(columns=['ClabsiDTS','DaysToCLABSI','MostRecentClabsiDTS','DiagnosisLast15','DiagnosisLast5',
                                   'DiagnosisLast3','DiagnosisLast2','SurgeriesLast15','SurgeriesLast5','SurgeriesLast3',
                                   'SurgeriesLast2'])


# #### Transforming object columns of date to datetime columns

# In[35]:


# Transforming 'Date' and 'InsertionDate' from object to datetime

df_clabsi['Date'] = pd.to_datetime(df_clabsi['Date'])
df_clabsi['InsertionDate'] = pd.to_datetime(df_clabsi['InsertionDate'])
df_clabsi['LastAncDate'] = pd.to_datetime(df_clabsi['LastAncDate'])


# #### Transforming binary text (binary string bool) columns with labels "TRUE" and "FALSE" to a dummy columns of float 

# In[36]:


# transforming binary string bool columns to dummy columns of float

df_clabsi=df_clabsi.astype({'HasCLABSI':float,
                            'HasFutureEncounterCLABSI':float,
                            'HadPreviousCLABSI':float,
                            'HasRecentLDAFlowsheetRecords':float,
                            'DiagnosisLeukemiaLast30':float,
                            'DiagnosisLeukemiaLast15':float,
                            'DiagnosisLeukemiaLast5':float,
                            'DiagnosisLeukemiaLast3':float,
                            'DiagnosisLeukemiaLast2':float,
                            'DiagnosisTransplantLast30':float,
                            'DiagnosisTransplantLast15':float,
                            'DiagnosisTransplantLast5':float,
                            'DiagnosisTransplantLast3':float,
                            'DiagnosisTransplantLast2':float,
                            'DiagnosisSwellingLast30':float,
                            'DiagnosisSwellingLast15':float,
                            'DiagnosisSwellingLast5':float,
                            'DiagnosisSwellingLast3':float,
                            'DiagnosisSwellingLast2':float})


# #### Sanity check to verify the successful transform of columns

# In[37]:


# Performing sanity check for 'HasCLABSI' the target column 

display(df_clabsi.dtypes[['HasCLABSI']])
print('='*40)
print('Original values look like:')
display(df_clabsi[['HasCLABSI']].head(10))
print('-'*30)
print('Transformed values look like:')
display(df_clabsi[['HasCLABSI']].head(10))


# In[38]:


# Similar to above, performing sanity check on other variable/column 'HadPreviousCLABSI'

display(df_clabsi.dtypes[['HadPreviousCLABSI']])
print('='*40)
print('Original values look like:')
display(df_clabsi[['HadPreviousCLABSI']].head(10))
print('-'*30)
print('Transformed values look like:')
display(df_clabsi[['HadPreviousCLABSI']].head(10))


# # Exploratory Data Analysis (EDA)

# ### Descriptive Statistics of numeric variables and target variable 'HasCLABSI', excluding  'PatientKey' and 'EncounterKey' as they are unique identifiers (Metadata) 

# In[39]:


# Copying the dataset 

df_clabsi_descriptive = df_clabsi.copy()


# In[40]:


# Dropping the unique identifiers 'PatientKey' and 'EncounterKey'

df_clabsi_descriptive = df_clabsi_descriptive.drop(columns=['PatientKey','EncounterKey']) 


# In[41]:


# Displaying the descriptive statistics

df_clabsi_descriptive.describe().round(2).T[['mean', 'std', 'min', 'max']]


# The above descriptive statistics displaying the mean, standard deviation, minimum value and maximum value of all the numerical variables.

# ## Exploring Target Variable 'HasCLABSI'

# In[42]:


# Displaying the value counts of 'HasCLABSI'
print('The value counts of the target variable "HasCLABSI":')
print()
print(df_clabsi.HasCLABSI.value_counts())


# #### Plot displaying the value counts for target variable 'HasCLABSI'  

# In[43]:


plt.figure(figsize = (8, 5))
plot = sns.countplot(x = 'HasCLABSI', data = df_clabsi)
plot.set(title="Instances of target variable 'HasCLABSI'")


# The above plot shows the patients CLABSI infection  with 52 instances of TRUE and 14184 instances of FALSE.

# ### Displaying the correlation between input variables and the target 'HasCLABSI' variable

# In[44]:


df_clabsi.corr()['HasCLABSI'].sort_values(ascending=False)


# #### Observations based on above correlation:
# 
# - HasCLABSI, the target variable, has a correlation of 1 with itself (the highest possible value).
# 
# - The correlation between the variables 'HasCLABSI' and 'HasFutureEncounterCLABSI' is 0.209961 i.e., approximately 0.21
# 
# - There is no strong correlation between the target variable 'HasCLABSI' and remaining all other variables.
# 

# ## Exploring the Patients data through Visualization 

# #### AgeGroup variable 

# In[45]:


## To get the discrete labels and value counts for 'AgeGroup'
n_labels_agegroup = df_clabsi['AgeGroup'].nunique()
print(f'There are {n_labels_agegroup} discrete labels in "AgeGroup"') 
print()
print('The AgeGroup labels and their respective value counts:')
print()
print(df_clabsi.AgeGroup.value_counts())


# In[46]:


# Plot showing patients from different AgeGroup

plt.figure(figsize = (18, 8))
plot1 = sns.countplot(x = 'AgeGroup', data = df_clabsi, order = df_clabsi['AgeGroup'].value_counts().index)
plot1.set_xticklabels(plot1.get_xticklabels(), rotation = 60)
plot1.set_title("Patients from different age groups")
plot1.set_ylabel("Value Counts of patients")
plt.show()


# ### Observations based on the above plot:
#   
# - Plot indicates majority of the patients belong to the age group of 0-2 years.
# - Age group 2-4 years, 4-6 years and 8-10 years  are having closer to similar distribution of patient and other age groups show lesser number of patients.
# - The dataset has no patients above the age of 44 years. 

# #### LineCount

# In[47]:


## To get the discrete labels and value counts for 'LineCount' variable

n_labels_lineCount = df_clabsi['LineCount'].nunique()
print(f'There are {n_labels_lineCount} discrete labels in "LineCount"') 
print()
print('The LineCount labels and their respective value counts:')
print()
print(df_clabsi.LineCount.value_counts())


# In[48]:


# Pie chart displaying the LineCount discrete labels in percentages

plt.figure(figsize = (20, 8))
plot2 = df_clabsi['LineCount'].value_counts().plot(kind='pie' , autopct='%.2f')
plot2.set_title("Percentage of Patients with the number of Central lines i.e.,'LineCount' variable")
plot2.set_ylabel("LineCount")
plt.show()


# ### Observations based on above pie chart and discrete labels of 'LineCount' value counts:
# 
# - The LineCount variable has 5 labels 1, 2, 3, 4 and 5 respectively which represent the patient with the number of central lines.
# - From the above calculated value counts and the pie chart of 'LineCount' variable, Label '1' shows a value count of 12165 representing 85.45% of patients had one central line being inserted. 11.98% of patients had two central lines with value count of 1706. Similarly 1.86% of patients had 3 central lines with value count of 265, 0.55% of patients had 4 central lines with count 79 and finally very less patient count of 21 had 5 central lines accounting 0.15%. 

# #### PastCLABSIs

# In[49]:


## To get the discrete labels and value counts for 'PastCLABSIs' variable

n_labels_pastClabsi = df_clabsi['PastCLABSIs'].nunique()
print(f'There are {n_labels_pastClabsi} discrete labels in "PastCLABSIs"') 
print()
print('The PastCLABSIs labels and their respective value counts:')
print()
print(df_clabsi.PastCLABSIs.value_counts())


# In[50]:


# Pie chart displaying the PastCLABSIs discrete labels in percentages

plt.figure(figsize = (20, 8))
plot2 = df_clabsi['PastCLABSIs'].value_counts().plot(kind='pie' , autopct='%.2f')
plot2.set_title("Percentage of Patients with the number of past CLABSI infection i.e., 'PastCLABSIs' variable")
plot2.set_ylabel("PastCLABSIs")
plt.show()


# ### Observations based on above pie chart and discrete labels of 'PastCLABSIs' value counts:
# 
# - The 'PastCLABSIs' variable has 3 labels 0, 1 and 2 respectively which represent the patient with the number of past CLABSI infections.
# - From the above calculated value counts and the pie chart of 'PastCLABSIs' variable, Label '0' shows a value count of 13003 representing 91.34% of patients had no past CLASBI infection. However, 6.46% of patients had past CLABSI infection once with value count of 920 reprsenting label '1' and 2.20% i.e., 313 patients were infected twice in the past with CLABSI representing label '2'. 

# #### HasRecentLDAFlowsheetRecords

# In[51]:


## To get the discrete labels and value counts for 'HasRecentLDAFlowsheetRecords' variable

n_labels_records = df_clabsi['HasRecentLDAFlowsheetRecords'].nunique()
print(f'There are {n_labels_records} discrete labels in "HasRecentLDAFlowsheetRecords"') 
print()
print('The HasRecentLDAFlowsheetRecords labels and their respective value counts:')
print()
print(df_clabsi.HasRecentLDAFlowsheetRecords.value_counts())


# In[52]:


## Plot for 'HasRecentLDAFlowsheetRecords' variable

plt.figure(figsize = (10, 5)) 
plot1 = sns.countplot(x = 'HasRecentLDAFlowsheetRecords', data = df_clabsi, order = df_clabsi['HasRecentLDAFlowsheetRecords'].value_counts().index)
plot1.set_xticklabels(plot1.get_xticklabels(), rotation = 60)
plot1.set_title("Number of Patients having recent nursing documentation about the central line i.e., 'HasRecentLDAFlowsheetRecords' variable")
plot1.set_ylabel("Value Counts of records")
plt.show()


# ### Observations of 'HasRecentLDAFlowsheetRecords' variable plot:
# 
# - The plot shows 12764 patients having recent nursing documentation about the central line. However, 1472 patients don't have the documentation.

# #### UndergoingChemoTreatment

# In[53]:


## To get the discrete labels and value counts for 'UndergoingChemoTreatment' variable

n_labels_chemo = df_clabsi['UndergoingChemoTreatment'].nunique()
print(f'There are {n_labels_chemo} discrete labels in "UndergoingChemoTreatment"') 
print()
print('The UndergoingChemoTreatment labels and their respective value counts:')
print()
print(df_clabsi.UndergoingChemoTreatment.value_counts())


# In[54]:


## Plot for 'UndergoingChemoTreatment' variable

plt.figure(figsize = (10, 5)) 
plot1 = sns.countplot(x = 'UndergoingChemoTreatment', data = df_clabsi, order = df_clabsi['UndergoingChemoTreatment'].value_counts().index)
plot1.set_xticklabels(plot1.get_xticklabels(), rotation = 60)
plot1.set_title("Number of Patients undergoing chemotherapy")
plot1.set_ylabel("Value Counts")
plt.show()


# ### Observations of 'UndergoingChemoTreatment' variable plot:
# - Lesser number of patients underwent chemotherapy.

# #### SurgeryGroupsLast30

# In[55]:


## To get the discrete labels and value counts for 'SurgeryGroupsLast30' variable

n_labels_Surgery = df_clabsi['SurgeryGroupsLast30'].nunique()
print(f'There are {n_labels_Surgery} discrete labels in "SurgeryGroupsLast30"') 
print()
print('The SurgeryGroupsLast30 labels and their respective value counts:')
print()
print(df_clabsi.SurgeryGroupsLast30.value_counts())


# In[56]:


# Plot for 'SurgeryGroupsLast30' variable

plt.figure(figsize = (18, 8))
plot1 = sns.countplot(x = 'SurgeryGroupsLast30', data = df_clabsi, order = df_clabsi['SurgeryGroupsLast30'].value_counts().index)
plot1.set_xticklabels(plot1.get_xticklabels(), rotation = 60)
plot1.set_title("Surgery types grouped under categories")
plot1.set_ylabel("Counts")
plt.show()


# ### Observations of 'SurgeryGroupsLast30' variable plot:
# 
# - The variable 'SurgeryGroupsLast30' consists of 7 discrete labels namely 0, 3, 3 6, 6, 3 4, 3 4 6 and 4.
# - Around 7325 patients are without any surgery, 5224 patients are having label '3' category group of surgery, 995 patients are under 3 6 category and all other remaining patients fall under labels 6, 3 4, 3 4 6 and 4 respectively.

# #### DepartmentDSC

# In[57]:


## To get the discrete labels and value counts for 'DepartmentDSC' variable

n_labels_department = df_clabsi['DepartmentDSC'].nunique()
print(f'There are {n_labels_department} discrete labels in "DepartmentDSC"') 
print()
print('The DepartmentDSC labels and their respective value counts:')
print()
print(df_clabsi.DepartmentDSC.value_counts())


# In[58]:


# Plot for 'SurgeryGroupsLast30' variable

plt.figure(figsize = (18, 8))
plot1 = sns.countplot(x = 'DepartmentDSC', data = df_clabsi, order = df_clabsi['DepartmentDSC'].value_counts().index)
plot1.set_xticklabels(plot1.get_xticklabels(), rotation = 60)
plot1.set_title("Hospital department where the patient was currently admitted")
plot1.set_ylabel("Counts")
plt.show()


# ### Observations of 'DepartmentDSC' variable plot:
# 
# - According to the above plot we can see most of the patients were admitted in 'HEMATOLOGY/ONCOLOGY','NEWBORN CNTR LVL 3', 'CARDIOVASCULR ICU', 'GNRL MEDICINE' and 'BNE MRRW TRNSPLT UT' departments.
# - All the other remaining patients were admitted into respective departments which can be visualized in the above plot.

# ## Overall analysis of the CLABSI Patient data:
# 
# - Majority of the patients belong to the age group of 0-2 years.
# - Around 85.45% of patients had one central line being inserted.
# - 91.34% of patients had no past CLASBI infection. Whereas 6.46% of patients had past CLABSI infection once and 2.20% of patients were infected twice in the past with CLABSI. 
# - Majority of patients were having recent nursing documentation about the central line.
# - Only 310 patients were undergoing chemotherapy.
# - Around 51.45% of patients are without any surgeries.
# - Most of the patients were admitted in 'HEMATOLOGY/ONCOLOGY','NEWBORN CNTR LVL 3', 'CARDIOVASCULR ICU', 'GNRL MEDICINE' and 'BNE MRRW TRNSPLT UT' departments.

# ## Visualization of Collinearity 

# ### Creating visualization to detect collinear input variables in the dataset
# As the dataset is huge and with many similar/duplicate columns representing similar information. It becomes difficult to check multi-collinearity considering all the variables leading to outcome bias. Therefore, we are shortlisting columns/variables which we believe will add value in buiding the required models.

# In[59]:


# Creating a new dataset by shortlisting input variables to detect the collinearity

df_clabsi_heatmap = df_clabsi[['HospitalDay','LineCount','LineDay','HasCLABSI','PastCLABSIs','HadPreviousCLABSI',
           'LineDaysPort','LineDaysPICC','LineDaysVenousAccess','LineDaysSingleLumen','LineDaysDoubleLumen',
           'LineDaysTripleLumen','LineDaysQuadrupleLumen','LineDaysMultiLumen','LineDaysFemoral','LineDaysSubclavian',
           'LineDaysUpperArm','LineDaysJugular','LineDaysChest','LineDaysLeg','FlushedLast10','CapChangesLast10',
           'DressingChangesLast10','TubingChangesLast10','UndergoingChemoTreatment','CytarabineChemoLast30',
           'ChlorhexidineLast30','CountMedicationsLast30','MedicationsLast30','MedicationsInjectedLast30',
           'AlteplaseAdministeredLast30','HCLAdministeredLast30','MedsAcidSuppTherapyLast30','MedsBowelRegimenLast30',
           'MedsCentralTPNLast30','MedsFatEmulsionLast30','MedsH2RALast30','MedsNSAIDLast30','MedsOralCareLast30',
           'MedsPeripheralTPNLast30','MedsPPILast30','MedsPrematureLast30','MedsPropofolLast30','MedsSteroidsLast30',
           'ProceduresLast30','ICDCodesLast30','ICDCategoriesLast30','DiagnosisLast30','DiagnosisLeukemiaLast30',
           'DiagnosisTransplantLast30','DiagnosisClotLast30','DiagnosisCancerLast30','DiagnosisDisorderLast30',
           'DiagnosisPtSizeLast30','DiagnosisEventLast30','DiagnosisInfectionLast30','DiagnosisShuntCardiacLast30',
           'DiagnosisRenalLast30','SurgeryGroupsLast30','SurgerySubGroupsLast30','SurgeryCountLast30','SurgeriesLast30',
           'SurgeriesBloodLast30','SurgeriesCancerLast30','SurgeriesCongenitalHeartDiseaseLast30',
           'SurgeriesSingleVentricleLast30','SurgeriesICULast30','SurgeriesAbdominalLast30','SurgeriesIntestinalLast30',
           'SurgeriesPlasticLast30','SurgeriesDelayedSternalClosure','CHGNonCompliantDays','CHGBathsLast30',
           'SedatedLast30','ICUDaysLast30','PICUDaysLast30','NICUDaysLast30','CVICUDaysLast30']].copy()


# In[60]:


# Correlation
clabsi_correlationtest = df_clabsi_heatmap.corr()


# In[61]:


# Preparing tools for making a correlation heatmap
mask = np.triu(np.ones_like(clabsi_correlationtest, bool))
f, ax = plt.subplots(1,1, figsize=(20, 20))
cmap = sns.diverging_palette(220, 10, as_cmap=True)


# Plot heatmap
sns.heatmap(clabsi_correlationtest, cmap=cmap, vmin=-1, vmax=1, center=0, square=True, 
            linewidths=1.5, cbar_kws={'shrink': .1})


# In[62]:


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
import sys
np.set_printoptions(threshold=2000)


# In[63]:


# Variables with collinearity greater than 0.7

pd.set_option("display.max_rows", None, "display.max_columns", None)
corr_var = df_clabsi_heatmap.corr().stack().reset_index()
corr_var.columns = ['Variable1', 'Variable2', 'CORRELATION']
mask_dups = (corr_var[['Variable1', 'Variable2']].apply(frozenset, axis=1).duplicated()) | (corr_var['Variable1']==corr_var['Variable2']) 
corr_var = corr_var[~mask_dups]
corr_var = corr_var.round(5)
corr_var = corr_var[corr_var['CORRELATION'] > .70]
corr_var.sort_values('CORRELATION',ascending=False)


# 

# ### Performing a train-test split

# In[64]:


X = df_clabsi.drop(columns=['PatientKey','EncounterKey','HasCLABSI'])
y = df_clabsi['HasCLABSI']


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# In[66]:


corrmatrix = X_train.corr()


# In[67]:


# Displaying the correlation matrix
corrmatrix


# In[68]:


def correlation(df_clabsi, threshold):
    correlated_columns = set()
    correlation_matrix = df_clabsi.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i,j]) > threshold:
                column_names = correlation_matrix.columns[i]
                correlated_columns.add(column_names)
    return correlated_columns


# In[69]:


correlated_variables = correlation(X_train, 0.8)
correlated_variables


# In[70]:


X_train.drop(labels=correlated_variables, axis=1, inplace=True)
X_test.drop(labels=correlated_variables, axis=1, inplace=True)


# In[71]:


X_train.shape, X_test.shape


# In[ ]:





# #### Evaluate validity of split using hypothesis-testing
# ###### Equal proportion for cases y=1

# Stating null hypothesis H0 (Null Hypothesis): The proportion of patients with clabsi are same in both the sets or the proportion of the clabsi data train and test split are same in both the sets in terms of 'HasCLABSI' variable.
# 
# Hypothesis-testing procedure (z-test of equal proportion): with Alpha= 0.05 (Significance level or type-1 error)

# In[72]:


count_default_train = y_train.value_counts()[1]
count_default_test = y_test.value_counts()[1]
z_stat, p = proportions_ztest(count=[count_default_train, count_default_test], nobs=[y_train.shape[0], y_test.shape[0]])
print('z-stat = {:0.3f}, p = {:0.3f}'.format(z_stat, p))


# #### Result interpretation:
# The above results show z-stat = -0.727 and p = 0.467. Because p-value > alpha which is 0.05, we can conclude that H0 (Null hypothesis) is not rejected and the porportion of patients with clabsi is same in the two sets or the proportion of the clabsi data train and test split is same in the two sets in terms of the 'HasCLABSI' variable.
