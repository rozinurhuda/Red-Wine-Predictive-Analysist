#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Predictive Analysist

# ## Dataset Preparation

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


data_from_local = r'C:\Users\Lenovo\Documents\Machine Learning\Project\ML-009_Wine_quality_Prediction\dataset\winequality-red.csv'
df = pd.read_csv(data_from_local)
df


# ## Data Understanding

# * Memberikan informasi seperti jumlah data, kondisi data, dan informasi mengenai data yang digunakan
# 
# * Sumber data diambil dari tautan (link download):
# 
# https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

# ## NOTE

# Informasi mengenai kolom dataset:
# 
# 1. ***fixed acidity*** - Jumlah asam tartarat dalam g/L, meskipun fixed acidity umumnya dalam anggur termasuk asam malat, sitrat, dan suksinat juga.
# 2. ***volatile acidity*** - Jumlah asam asetat (cuka) dalam g/L. Volatile acidity yang lebih umum juga termasuk laktat, formik, butirat, dan propionat. Asam ini terkait dengan pembusuan anggur.
# 3. **citric acid** - Jumlah asam sitrat dalam g/L. Asam sitrat biasanya hadir dalam anggur tetapi dapat ditambahkan ke anggur untuk meningkatkan keasaman.
# 4. ***residual sugar*** - Biasanya jumlah gula alami dalam g/L yang tersisa dalam anggur setelah proses fermentasi selesai. Beberapa negara mengizinkan tambahan gula untuk ditambahkan, tetapi praktik ini tidak disukai oleh para kritikus.
# 5. ***chlorides*** - Jumlah sodium klorida dalam g/L.
# 6. ***free sulfur dioxide*** - Jumlah sulfit yang tersedia untuk bereaksi dalam mg/L. Sulfit (sulfur dioksida atau SO2) sering ditambahkan ke anggur sebagai pengawet, tetapi beberapa juga terjadi secara alami.
# 7. ***total sulfur dioxide*** - Jumlah total sulfit bebas dan sudah bereaksi (terikat) dalam mg/L.
# 8. ***density*** - Terukur dalam g/ml.
# 9. ***pH*** - Pengukuran keasaman anggur (pH lebih rendah lebih asam)
# 10. ***sulphates*** - Bentuk lain dari belerang alami (SO4) yang bergantung pada komposisi tanah tempat anggur ditanam.
# 11. ***alcohol*** - Persentase alkohol berdasarkan volume.
# 12. ***quality*** - nilai antara 0 sampai dengan 10.

# # EDA

# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


df.isnull().sum()


# In[8]:


from tabulate import tabulate

summary = df.describe().transpose()
markdown_result = tabulate(summary, headers='keys', tablefmt='github')
markdown_result = f"```markdown\n{markdown_result}\n```"
print(markdown_result)


# ### Jumlah Data Pada Kolom *quality*
# 
# Melihat banyaknya jumlah data yang terdapat pada fitur quality

# In[9]:


plt.figure(dpi=120)
ax = sns.countplot(data=df, x='quality')
ax.bar_label(ax.containers[0], fmt='%.1f')


# Ternyata dalam dataset terdapat range nilai quality dari 3 sampai dengan 8 saja.
# 
# Untuk melihat lebih dalam berikut ini akan dibuat tabel perbandingan antara kualitas dan fitur-fitur lainnya dalam dataset.

# In[10]:


plt.figure(dpi=120)  #for setting the plot size
plt.title('Quality vs pH')  #add title
ax = sns.barplot(x='quality',y='pH',data=df)
ax.bar_label(ax.containers[0], fmt='%.1f')


# In[11]:


plt.figure(dpi=120)  #for setting the plot size
plt.title('Quality vs Fixed acidity')  #add title
ax = sns.barplot(x='quality',y='fixed acidity',data=df)
ax.bar_label(ax.containers[0], fmt='%.1f')


# In[12]:


plt.figure(dpi=120)  #for setting the plot size
plt.title('Quality vs Volatile acidity')  #add title
ax = sns.barplot(x='quality',y='volatile acidity',data=df)
ax.bar_label(ax.containers[0], fmt='%.1f')


# In[13]:


plt.figure(dpi=120)  #for setting the plot size
plt.title('Quality vs Citric Acid')  #add title
ax = sns.barplot(x='quality',y='citric acid',data=df)
ax.bar_label(ax.containers[0], fmt='%.1f')


# In[14]:


plt.figure(dpi=120)  #for setting the plot size
plt.title('Quality vs Residual Sugar')  #add title
ax = sns.barplot(x='quality',y='residual sugar',data=df)
ax.bar_label(ax.containers[0], fmt='%.2f')


# In[15]:


plt.figure(dpi=120)  #for setting the plot size
plt.title('Quality vs Chlorides')  #add title
ax = sns.barplot(x='quality',y='chlorides',data=df)
ax.bar_label(ax.containers[0], fmt='%.2f')


# In[16]:


plt.figure(dpi=120)  #for setting the plot size
plt.title('Quality vs Free Sulfur Dioxide')  #add title
ax = sns.barplot(x='quality',y='free sulfur dioxide',data=df)
ax.bar_label(ax.containers[0], fmt='%.1f')


# In[17]:


plt.figure(dpi=120)  #for setting the plot size
plt.title('Quality vs Total Sulfur Dioxide')  #add title
ax = sns.barplot(x='quality',y='total sulfur dioxide',data=df)
ax.bar_label(ax.containers[0], fmt='%.1f')


# In[18]:


plt.figure(dpi=120)  #for setting the plot size
plt.title('Quality vs Density')  #add title
ax = sns.barplot(x='quality',y='density',data=df)
ax.bar_label(ax.containers[0], fmt='%.1f')


# In[19]:


plt.figure(dpi=120)  #for setting the plot size
plt.title('Quality vs Sulphates')  #add title
ax = sns.barplot(x='quality',y='sulphates',data=df)
ax.bar_label(ax.containers[0], fmt='%.1f')


# In[20]:


plt.figure(dpi=120)  #for setting the plot size
plt.title('Quality vs Alcohol')  #add title
ax = sns.barplot(x='quality',y='alcohol',data=df)
ax.bar_label(ax.containers[0], fmt='%.1f')


# In[21]:


plt.figure(figsize=(12,10))  #to set the plot size
sns.heatmap(df.corr(),annot=True,linewidth=0.5,cmap='coolwarm')


# Dari matriks korelasi di atas yang memiliki fitur korelasi positif adalah
# 
# 1. **fixed acidity** dan **citric acid**
# 
# 2. **free sulphur dioxide** dan **total sulphor dioxide**
# 
# 3. **fixed acidity** dan **density** 
# 
# 
# dan korelasi negatif meliputi
# 
# 1. **citric acid** dan **volatile acidity**
# 
# 2. **fixed acidity** dan **pH**
# 
# 3. **density** dan **alcohol** 

# In[22]:


def corr_graph(x,y):
    sns.regplot(x=df[f'{x}'], y=df[f'{y}'], color = '#0e87cc', lowess=True, scatter_kws={'edgecolor':'black', 'alpha':.6}, line_kws={"color":"red", "linewidth":"2"})


# In[23]:


# Negative Correlation B/W Fixed Acidity & pH
plt.figure(dpi=120)
corr_graph('citric acid','volatile acidity')
plt.title("Negative Correlation B/W Citric Acid & Volatile Acidity")


# In[24]:


# Negative Correlation B/W Fixed Acidity & pH
plt.figure(dpi=120)
corr_graph('fixed acidity','pH')
plt.title("Negative Correlation B/W Fixed Acidity & pH")


# In[25]:


# Positive Correlation B/W Fixed Acidity & pH
plt.figure(dpi=120)
corr_graph('fixed acidity','citric acid')
plt.title("Positive Correlation B/W fixed acidity & citric acid")


# In[26]:


# Positive Correlation B/W Fixed Acidity & Density
plt.figure(dpi=120)
corr_graph('fixed acidity','density')
plt.title("Positive Correlation B/W Fixed Acidity & Density")


# ## Modeling

# In[27]:


df.shape


# In[28]:


df['quality'] = df['quality'].apply(lambda x: 1 if x >= 6.5 else 0)
df


# In[29]:


plt.figure(figsize=(8, 4),dpi=100)

ax= sns.countplot(data=df, x='quality')
ax.bar_label(ax.containers[0], fmt='%.1f')
plt.xticks([0,1], ['Bad wine', 'Good wine'])


# In[30]:


X=df.drop('quality',axis=1).values
y=df['quality']


# In[31]:


from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy=1) # Numerical value
# rus = RandomUnderSampler(sampling_strategy="not minority") # String
X_res, y_res = rus.fit_resample(X, y)
ax = y_res.value_counts().plot.pie(autopct='%.2f')
_ = ax.set_title("Under-sampling")


# In[32]:


y_res.value_counts()


# In[33]:


from imblearn.over_sampling import RandomOverSampler
#ros = RandomOverSampler(sampling_strategy=1) # Float
ros = RandomOverSampler(sampling_strategy="not majority") # String
X_res, y_res = ros.fit_resample(X, y)
ax = y_res.value_counts().plot.pie(autopct='%.2f')
_ = ax.set_title("Over-sampling")


# In[34]:


y_res.value_counts()


# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size = 0.2, random_state = 42)


# In[36]:


print("Shape of X_train: ",X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)


# In[37]:


from lazypredict.Supervised import LazyClassifier

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
top_5_models = models.head(5)
top_5_models


# In[38]:


table = tabulate(top_5_models, headers="keys", tablefmt="github")

print(table)


# ### Extra Trees Classifier

# In[39]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report


# In[40]:


from sklearn.ensemble import ExtraTreesClassifier

classifier_xts = ExtraTreesClassifier(n_estimators=100, random_state=42)
classifier_xts.fit(X_train, y_train)


# In[72]:


y_pred_xts=classifier_xts.predict(X_test)
confusion_svc_xts=confusion_matrix(y_test,y_pred_xts)
plt.figure(figsize=(4,4))
sns.heatmap(confusion_svc_xts,fmt="d",annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
print(classification_report(y_test,y_pred_xts))
print(f"Accuracy Score: {accuracy_score(y_test, y_pred_xts)*100:.2f}%")


# In[42]:


accuracies_xts = cross_val_score(estimator = classifier_xts, X = X_train, y = y_train, cv = 10)
print("Accuracy ExtraTreesClassifier: {:.2f} %".format(accuracies_xts.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies_xts.std()*100))


# ## Decision Tree Classifier

# In[43]:


from sklearn.tree import DecisionTreeClassifier

classifier_dtc = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
classifier_dtc.fit(X_train, y_train)


# In[44]:


y_pred_dtc=classifier_dtc.predict(X_test)
confusion_svc_dtc=confusion_matrix(y_test,y_pred_dtc)
plt.figure(figsize=(4,4))
sns.heatmap(confusion_svc_dtc,fmt="d",annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
print(classification_report(y_test,y_pred_dtc))
print(f"Accuracy Score: {accuracy_score(y_test, y_pred_dtc)*100:.2f}%")


# In[45]:


from sklearn.model_selection import GridSearchCV

classifier_dtc = DecisionTreeClassifier()
parameters = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

classifier_dtc_grid_search = GridSearchCV(classifier_dtc, parameters, cv=5)
classifier_dtc_grid_search.fit(X_train, y_train)

best_params = classifier_dtc_grid_search.best_params_
best_model = classifier_dtc_grid_search.best_estimator_

y_pred_dtc=best_model.predict(X_test)
confusion_svc_dtc=confusion_matrix(y_test,y_pred_dtc)
plt.figure(figsize=(4,4))
sns.heatmap(confusion_svc_dtc,fmt="d",annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
print(classification_report(y_test,y_pred_dtc))
print(f"Accuracy Score: {accuracy_score(y_test, y_pred_dtc)*100:.2f}%")

accuracy = accuracy_score(y_test, y_pred_dtc)
print("Akurasi:", accuracy)
print("Parameter terbaik:", best_params)


# In[46]:


accuracies_dtc = cross_val_score(estimator = classifier_dtc_grid_search, X = X_train, y = y_train, cv = 10)
print("Accuracy RandomForestClassifier: {:.2f} %".format(accuracies_dtc.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies_dtc.std()*100))


# ## Random Forest Classifier

# In[47]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


# In[48]:


from sklearn.ensemble import RandomForestClassifier

classifier_rf = RandomForestClassifier(n_estimators= 100,criterion = 'entropy', random_state = 42)
classifier_rf.fit(X_train, y_train)


# In[49]:


y_pred_rf=classifier_rf.predict(X_test)
confusion_svc_rf=confusion_matrix(y_test,y_pred_rf)
plt.figure(figsize=(4,4))
sns.heatmap(confusion_svc_rf,fmt="d",annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
print(classification_report(y_test,y_pred_rf))
print(f"Accuracy Score: {accuracy_score(y_test, y_pred_rf)*100:.2f}%")


# In[50]:


accuracies_rf = cross_val_score(estimator = classifier_rf, X = X_train, y = y_train, cv = 10)
print("Accuracy RandomForestClassifier: {:.2f} %".format(accuracies_rf.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies_rf.std()*100))


# ## XGB Classifier

# In[51]:


import xgboost as xgb

classifier_xgb = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1)
classifier_xgb.fit(X_train, y_train)


# In[52]:


y_pred_xgb=classifier_xgb.predict(X_test)
confusion_svc_xgb=confusion_matrix(y_test,y_pred_xgb)
plt.figure(figsize=(4,4))
sns.heatmap(confusion_svc_xgb,fmt="d",annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
print(classification_report(y_test,y_pred_xgb))
print(f"Accuracy Score: {accuracy_score(y_test, y_pred_xgb)*100:.2f}%")


# In[53]:


accuracies_xgb = cross_val_score(estimator = classifier_xgb, X = X_train, y = y_train, cv = 10)
print("Accuracy XGBClassifier: {:.2f} %".format(accuracies_xgb.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies_xgb.std()*100))


# ## Bagging Classifier

# In[64]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Membangun BaggingClassifier dengan DecisionTreeClassifier
# sebagai pengklasifikasi dasar
base_classifier = DecisionTreeClassifier(max_depth=3) # Mengatur parameter max_depth
classifier_bc = BaggingClassifier(base_classifier, n_estimators=10, max_samples=0.8, max_features=0.8, bootstrap=True, random_state=42)
classifier_bc.fit(X_train, y_train)


# In[65]:


y_pred_bc=classifier_bc.predict(X_test)
confusion_svc_bc=confusion_matrix(y_test,y_pred_bc)
plt.figure(figsize=(4,4))
sns.heatmap(confusion_svc_bc,fmt="d",annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
print(classification_report(y_test,y_pred_bc))
print(f"Accuracy Score: {accuracy_score(y_test, y_pred_bc)*100:.2f}%")


# In[66]:


accuracies_bc = cross_val_score(estimator = classifier_bc, X = X_train, y = y_train, cv = 10)
print("Accuracy BaggingClassifier: {:.2f} %".format(accuracies_bc.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies_bc.std()*100))


# ## Final Report

# In[67]:


final_report = {'Nama Model': [], 'Akurasi': []}
final_report['Nama Model'].append('ExtraTreesClassifier')
xts = accuracies_xts.mean()*100
final_report['Akurasi'].append(xts)
final_report['Nama Model'].append('RandomForestClassifier')
rf = accuracies_rf.mean()*100
final_report['Akurasi'].append(rf)
final_report['Nama Model'].append('XGBClassifier')
xgb = accuracies_xgb.mean()*100
final_report['Akurasi'].append(xgb)
final_report['Nama Model'].append('DecisionTreeClassifier')
dtc = accuracies_dtc.mean()*100
final_report['Akurasi'].append(dtc)
final_report['Nama Model'].append('BaggingClassifier')
bc = accuracies_bc.mean()*100
final_report['Akurasi'].append(bc)
final_report


# In[68]:


pd.options.display.float_format = '{:.4f}'.format


# In[69]:


final_report = pd.DataFrame.from_dict(final_report)
table = tabulate(final_report, headers="keys", tablefmt="github")

print(table)


# In[ ]:




