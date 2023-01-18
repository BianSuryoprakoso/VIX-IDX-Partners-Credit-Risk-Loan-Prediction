#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('loan_data_2007_2014.csv')


# ### EDA

# In[3]:


df.info()


# Hasil pengamatan:
# - Data memiliki 466285 baris dan 75 kolom
# - Terdapat fitur atau kolom yang memiliki null values (kolom yang null values > 50% akan di drop)
# - Fitur atau kolom yang kebanyakan unique values akan dihapus / drop
# - Belum ada kolom target

# #### Check Duplicated Data

# In[129]:


df.duplicated().any()


# In[4]:


num_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numericals_df = df.select_dtypes(include = num_dtypes)

categoricals_df = df.select_dtypes(include = 'object')

nums = numericals_df.columns
cats = categoricals_df.columns

print('ini fitur type numerical:', nums)
print('')
print('ini fitur type numerical:', cats)


# In[11]:


df.sample(10)


# In[9]:


df[nums].describe().T


# Hasil pengamatan:
# - Fitur / kolom `Unnamed: 0`, `id`, `member_id` mempresentasikan unique values

# In[7]:


df[cats].describe()


# In[8]:


for col in cats:
    print(df[col].value_counts())
    print('')


# In[13]:


df['policy_code'].value_counts()


# Fitur `policy code` akan di drop nantinya karena memiliki satu value

# In[14]:


df['application_type'].value_counts()


# Fitur  `application_type` akan di drop nantinya karena memiliki satu value

# ### Create a target of the loan_status feature

# In[12]:


df['loan_status'].value_counts()


# ### Create the Target

# In[19]:


Status = []
for index, kolom in df.iterrows():
    if 'Current' in kolom['loan_status']:
        Status.append(1)
    elif 'Fully Paid' in kolom['loan_status']:
        Status.append(1)
    elif 'In Grace Period' in kolom['loan_status']:
        Status.append(1)
    else:
        Status.append(0)

df['Status_Borrower'] = Status


# Melabelkan jika loan status `Current`, `Fully Paid`, dan `In Grace Period` mempresentasikan 1 (Good Borrower).<br>
# Selain itu mempresentasikan 0 (Bad Borrower)

# In[20]:


df['Status_Borrower'].value_counts()


# Terdapat 414099 `Good Borrower` dan 52186 `Bad Borrower`

# ### Data Pre-Processing

# In[21]:


df.info()


# ### Feature Selection

# #### Ambil kolom yang diperlukan dan Drop kolom yang tidak diperlukan

# In[22]:


df_xixi = df[['loan_amnt', 'int_rate', 'installment', 'grade', 'annual_inc', 'issue_d', 'pymnt_plan', 'delinq_2yrs', 'mths_since_last_delinq',
              'open_acc', 'revol_bal', 'revol_util', 'total_pymnt', 'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 'collections_12_mths_ex_med',
              'acc_now_delinq', 'tot_cur_bal', 'total_rev_hi_lim', 'Status_Borrower']]
df_xixi.info()


# ### Menampilkan null values

# In[23]:


df_xixi.isna().sum()


# ### Imputasi Numerikal

# In[24]:


df_xixi['mths_since_last_delinq'].fillna(df_xixi['mths_since_last_delinq'].median(), inplace = True)
df_xixi['open_acc'].fillna(df_xixi['open_acc'].median(), inplace = True)
df_xixi['revol_util'].fillna(df_xixi['revol_util'].median(), inplace = True)
df_xixi['collections_12_mths_ex_med'].fillna(df_xixi['collections_12_mths_ex_med'].median(), inplace = True)
df_xixi['acc_now_delinq'].fillna(df_xixi['acc_now_delinq'].median(), inplace = True)
df_xixi['annual_inc'].fillna(df_xixi['annual_inc'].median(), inplace = True)
df_xixi['tot_cur_bal'].fillna(df_xixi['tot_cur_bal'].median(), inplace = True)
df_xixi['total_rev_hi_lim'].fillna(df_xixi['total_rev_hi_lim'].median(), inplace = True)
df_xixi['delinq_2yrs'].fillna(df_xixi['delinq_2yrs'].median(), inplace = True)


# In[25]:


df_xixi.isna().sum()


# #### Heatmap Corr

# In[26]:


plt.figure(figsize = (15,13))
sns.heatmap(df_xixi.corr(), annot = True, fmt = '.2f', cmap = 'Spectral')


# ### Feature Selection Part 2

# In[27]:


df_dropped = df_xixi.drop(columns = ['installment', 'total_pymnt', 'revol_bal', 'collection_recovery_fee', 'issue_d'])
plt.figure(figsize = (15,13))
sns.heatmap(df_dropped.corr(), annot = True, fmt = '.2f', cmap = 'Spectral')


# Drop fitur yang dianggap redundan, seperti `installment`, `total_pymnt`, `revol_bal`, `collection_recovery_fee`, `issue_d`

# In[28]:


df_dropped.info()


# In[29]:


df_dropped.describe()


# In[30]:


#inisiasi untuk handling outliers
num_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numericals_df = df_dropped.select_dtypes(include = num_dtypes)

categoricals_df = df_dropped.select_dtypes(include = 'object')

nums = numericals_df.columns
cats = categoricals_df.columns

print(nums)
print(cats)


# #### Handling Outliers

# In[31]:


from scipy import stats
#Using Z-Score
print('Jumlah baris sebelum memfilter outlier:', len(df_dropped))

filtered_entries = np.array([True] * len(df))

for col in nums:
    zscore = abs(stats.zscore(df_dropped[col])) # hitung absolute z-scorenya
    filtered_entries = (zscore < 3) & filtered_entries # keep yang kurang dari 3 absolute z-scorenya
    
df_dropped = df_dropped[filtered_entries] # filter, cuma ambil yang z-scorenya dibawah 3

print('Jumlah baris setelah memfilter outlier:', len(df_dropped))

#Using Z-score


# #### Scalling

# In[32]:


df_dropped.info()


# In[33]:


plt.figure(figsize = (15,13))
sns.heatmap(df_dropped.corr(), annot = True, fmt = '.2f', cmap = 'Spectral')


# Hasil Pengamatan:
# Setelah outliers dibuang, fitur `collections_12_mths_ex_med` dan `acc_now_delinq ` harus dihapus karena tidak ada korelasi

# In[34]:


df_clean = df_dropped.drop(columns = ['collections_12_mths_ex_med', 'acc_now_delinq'])
plt.figure(figsize = (15,13))
sns.heatmap(df_clean.corr(), annot = True, fmt = '.2f', cmap = 'Spectral')


# In[35]:


df_clean.describe()


# In[36]:


#Scalling
from sklearn.preprocessing import MinMaxScaler, StandardScaler
df_clean['loan_amnt_norm'] = MinMaxScaler().fit_transform(df_clean['loan_amnt'].values.reshape(len(df_clean), 1))
df_clean['int_rate_norm'] = MinMaxScaler().fit_transform(df_clean['int_rate'].values.reshape(len(df_clean), 1))
df_clean['annual_inc_std'] = StandardScaler().fit_transform(df_clean['annual_inc'].values.reshape(len(df_clean), 1))
df_clean['delinq_2yrs_norm'] = MinMaxScaler().fit_transform(df_clean['delinq_2yrs'].values.reshape(len(df_clean), 1))
df_clean['mths_since_last_delinq_norm'] = MinMaxScaler().fit_transform(df_clean['mths_since_last_delinq'].values.reshape(len(df_clean), 1))
df_clean['open_acc_norm'] = MinMaxScaler().fit_transform(df_clean['open_acc'].values.reshape(len(df_clean), 1))
df_clean['revol_util_norm'] = MinMaxScaler().fit_transform(df_clean['revol_util'].values.reshape(len(df_clean), 1))
df_clean['recoveries_std'] = StandardScaler().fit_transform(df_clean['recoveries'].values.reshape(len(df_clean), 1))
df_clean['last_pymnt_amnt_std'] = StandardScaler().fit_transform(df_clean['last_pymnt_amnt'].values.reshape(len(df_clean), 1))
df_clean['tot_cur_bal_std'] = StandardScaler().fit_transform(df_clean['tot_cur_bal'].values.reshape(len(df_clean), 1))
df_clean['total_rev_hi_lim_std'] = StandardScaler().fit_transform(df_clean['total_rev_hi_lim'].values.reshape(len(df_clean), 1))


# In[37]:


df_clean.info()


# In[38]:


#drop the original column that has not been scaling
df_cleaning = df_clean.drop(columns = ['loan_amnt', 'int_rate', 'annual_inc', 'delinq_2yrs', 'mths_since_last_delinq', 'open_acc',
                                       'revol_util', 'recoveries', 'last_pymnt_amnt', 'tot_cur_bal', 'total_rev_hi_lim'])
df_cleaning.info()


# In[39]:


plt.figure(figsize = (15,13))
sns.heatmap(df_cleaning.corr(), annot = True, fmt = '.2f', cmap = 'Spectral')


# #### Feature Encoding

# In[40]:


df_cleaning.info()


# In[41]:


#one hot encoding
for cat in ['grade', 'pymnt_plan']:
    onehots = pd.get_dummies(df_cleaning[cat], prefix=cat)
    df_cleaning = df_cleaning.join(onehots)


# In[42]:


#drop pymnt_plan, grade
df_cleaning = df_cleaning.drop(columns = ['pymnt_plan', 'grade'])
df_cleaning.info()


# In[43]:


plt.figure(figsize = (15,13))
sns.heatmap(df_cleaning.corr(), annot = True, fmt = '.2f', cmap = 'Spectral')


# #### Modelling

# In[44]:


X = df_cleaning.drop(labels=['Status_Borrower'],axis=1)
y = df_cleaning[['Status_Borrower']]


# In[45]:


#split test and train
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,stratify=y,random_state = 42)


# In[46]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def eval_classification(model, xtrain, ytrain, xtest, ytest):
    ypred = model.predict(xtest)
    print("Accuracy (Test Set): %.2f" % accuracy_score(ytest, ypred))
    print("Precision (Test Set): %.2f" % precision_score(ytest, ypred))
    print("Recall (Test Set): %.2f" % recall_score(ytest, ypred))
    print("F1-Score (Test Set): %.2f" % f1_score(ytest, ypred))
    
    y_pred_proba = model.predict_proba(xtest)
    print("AUC: %.2f" % roc_auc_score(ytest, y_pred_proba[:, 1]))


# <h2>Logisitic Reggression</h2>

# In[49]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# List Hyperparameters yang akan diuji
penalty = ['l2','l1','elasticnet','none']
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000] # Inverse of regularization strength; smaller values specify stronger regularization.
solver = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga','none']
class_weight = [{0: 1, 1: 1},
                {0: 1, 1: 2}, 
                {0: 1, 1: 3},
                {0: 1, 1: 4},
                'none']
hyperparameters = dict(penalty=penalty, C=C,class_weight=class_weight,solver=solver)

# Inisiasi model
logres = LogisticRegression(random_state=42) # Init Logres dengan Gridsearch, cross validation = 5
model = RandomizedSearchCV(logres, hyperparameters, cv=5, random_state=42, scoring='recall')

# Fitting Model & Evaluation
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
eval_classification(model, X_train, y_train, X_test, y_test)


# <b>Prediction Result in Dataset<b>

# In[50]:


y_pred = model.predict(X_test)
y_pred


# In[51]:


model.predict_proba(X_test)


# In[52]:


y_pred_train = model.predict(X_train)
y_pred_train


# <b>Evaluation<b>

# In[53]:


from sklearn.metrics import roc_auc_score #ini gapake predict_proba
roc_auc_score(y_test, y_pred)


# In[54]:


eval_classification(model, X_train, y_train, X_test, y_test) #ini auc nya udah pake predict proba


# In[55]:


print('Train score: ' + str(model.score(X_train, y_train))) #accuracy
print('Test score:' + str(model.score(X_test, y_test))) #accuracy


# <h2>KNN</h2>

# In[56]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Prediction & Evaluation
y_pred = knn.predict(X_test)
eval_classification(knn, X_train, y_train, X_test, y_test)


# In[57]:


print('Train score: ' + str(knn.score(X_train, y_train))) #accuracy
print('Test score:' + str(knn.score(X_test, y_test))) #accuracy


# <h2>Decision Tree</h2>

# In[58]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)

y_pred = dt.predict(X_test)
eval_classification(dt, X_train, y_train, X_test, y_test)


# In[59]:


print('Train score: ' + str(dt.score(X_train, y_train))) #accuracy
print('Test score:' + str(dt.score(X_test, y_test))) #accuracy


# <h2>Random Forest</h2>

# In[60]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)
eval_classification(rf, X_train, y_train, X_test, y_test)


# In[61]:


print('Train score: ' + str(rf.score(X_train, y_train))) #accuracy
print('Test score:' + str(rf.score(X_test, y_test))) #accuracy


# ### Insights

# In[67]:


ins = df.groupby(['loan_status']).agg({'id' : 'count'}).sort_values(['id'], ascending = False).reset_index()
ins.columns = ['loan_status', 'frequency']
ins['percentage %'] = round(ins['frequency']*100/sum(ins['frequency']),2)
ins


# In[128]:


sns.set_style('darkgrid')
plt.figure(figsize = (15, 9))

#set title
plt.title('Frequency of applicants based on Loan Status', fontsize = 16)

sns.barplot(x = 'frequency', y = 'loan_status', data = ins, color = 'cornflowerblue')

#set x y label
plt.xlabel('Frequency', fontsize = 13)
plt.ylabel('Loan Status', fontsize = 13)

plt.tight_layout()


# Berdasarkan visualisasi di atas `Loan Status Current` memiliki frequency 224226 yang sekitar 48% dan diikuti oleh `Loan Status Fully Paid` 184739 freq (39.62%) dan `Loan Status Charged Off` 42475 freq (9.11%)

# In[93]:


ins2 = df.groupby(['Status_Borrower']).agg({'id' : 'count'}).sort_values(['id'], ascending = False).reset_index()
ins2.columns = ['Status_Borrower', 'frequency']
ins2['percentage %'] = round(ins2['frequency'] * 100 / sum(ins['frequency']), 2)
ins2


# In[127]:


sns.set_style('whitegrid')
plt.figure(figsize = (10, 8))

#set title
plt.title('Frequency of applicants based on Status Borrower', fontsize = 16)

colr = cols = ['orange' if (x < max(ins2['Status_Borrower'])) else 'grey' for x in ins2['Status_Borrower']]
sns.barplot(x = 'Status_Borrower', y = 'frequency', data = ins2, palette = colr)

#set x y label
plt.xlabel('Status Borrower', fontsize = 13)
plt.ylabel('Frequency', fontsize = 13)

plt.tight_layout()

