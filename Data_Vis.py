#Libraries for Data Visualization

import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import numpy as np 
import seaborn as sns
import time

#Other Libraries

from sklearn.preprocessing import StandardScaler, RobustScaler #Scaling Time and Amount
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import norm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import collections


#Importing Data

input_data=pd.read_csv('creditcard.csv')
df=input_data.copy()
#print(df.head())

fraud_data=df.loc[df['Class']==1]
safe_data=df.loc[df['Class']==0]
### Make histogram for each attribute of fraudulent tarnsactions
# for i in fraud_data.columns:
#     plt.figure(i)
#     fraud_data[i].hist(bins=50)
# plt.show()

### Percentage of frauds
# percentage_frauds=fraud_data.shape[0]/df.shape[0]*100
# print('The percentage of fraudulent transactions is {:03.3f}%. This amounts to a total of fraudulent operations of {:d}.\n'.format(percentage_frauds,fraud_data.shape[0]) + 'The mean amount in fraudulent transactions is {:05.2f}. The mean of all transactions is {:04.2f}.'.format(fraud_data['Amount'].mean(),df['Amount'].mean()))

#print(df.describe()) 
#print(df.isna().sum().max()) #Check for null values

### Count plot for 
# colors = ["#0101DF", "#DF0101"]
# sns.countplot('Class',data=df,palette=colors)
# plt.title('Class distribution \n 0 = no fraud || 1 = fraud')
# plt.show()

### Check distribution of Time and Amount
# fig,ax=plt.subplots(1,2,figsize=(18,4))

# amount_val=df['Amount'].values
# time_val=df['Time'].values

# sns.distplot(amount_val,ax=ax[0])
# ax[0].set_title('Dist. of transaction amount')
# ax[0].set_xlim([min(amount_val),max(amount_val)])
# sns.distplot(time_val,ax=ax[1])
# ax[1].set_title('Dist. of transaction time')
# ax[1].set_xlim([min(time_val),max(time_val)])
# plt.show()

### Scaling Time and Amount
std_scaler=StandardScaler()
rb_scaler=RobustScaler()

df['scaled_Time']=rb_scaler.fit_transform(df['Time'].values.reshape(-1,1))
df['scaled_Amount']=rb_scaler.fit_transform(df['Amount'].values.reshape(-1,1))

df.drop(['Time','Amount'],axis=1,inplace=True)

df.insert(0,'Scaled_Amount',df['scaled_Amount'])
df.insert(1,'Scaled_Time',df['scaled_Time'])
del df['scaled_Amount'],df['scaled_Time']
#print(df.head())

### Check distribution of scaled Time and scaled Amount
# fig,ax=plt.subplots(1,2,figsize=(18,4))

# amount_val=df['Scaled_Amount'].values
# time_val=df['Scaled_Time'].values

# sns.distplot(amount_val,ax=ax[0])
# ax[0].set_title('Dist. of transaction scaled amount')
# ax[0].set_xlim([min(amount_val),max(amount_val)])
# sns.distplot(time_val,ax=ax[1])
# ax[1].set_title('Dist. of transaction scaled time')
# ax[1].set_xlim([min(time_val),max(time_val)])
# plt.show()



X=df.drop('Class',axis=1)
y=df['Class']

sss=StratifiedKFold(n_splits=5,random_state=None,shuffle=False)

for train_index, test_index in sss.split(X,y):
    #print('Train:', train_index, 'Test:', test_index)
    X_train, X_test=X.iloc[train_index],X.iloc[test_index]
    y_train, y_test=y.iloc[train_index], y.iloc[test_index]
    #print('Fraud percentage in train:', sum(y_train)/len(y_train)*100, 'Fraud percentage in test:', sum(y_test)/len(y_test)*100) #Check if it keeps the same percentage of frauds in each split

#print(y.value_counts())


### This is for the random undersampling!

#Shuffle the data!
df=df.sample(frac=1)
fraud_df=df.loc[df['Class']==1]
non_fraud_df=df.loc[df['Class']==0][:492] #Only the first 492 shuffled (i.e. random) safe transactions

#Join both shuffled df's and shuffle them!
balanced_df=pd.concat([fraud_df,non_fraud_df]).sample(frac=1)

#Balanced countplot!
# sns.countplot('Class',data=balanced_df)
# plt.title('Distribution of the Classes in the subsample dataset')
# plt.show()


### Correlation Matrices
# f,ax =plt.subplots(1,2,figsize=(18,6))

# df_corr=df.corr(method='pearson')
# balanced_corr=balanced_df.corr(method='pearson')

# sns.heatmap(df_corr,vmin=-1,vmax=1,center=0,cmap='coolwarm_r',annot_kws={'size':20},ax=ax[0])
# ax[0].set_title("Imbalanced Correlation Matrix \n (don't use for reference)")

# sns.heatmap(balanced_corr,vmin=-1,vmax=1,center=0,cmap='coolwarm_r',annot_kws={'size':20},ax=ax[1])
# ax[1].set_title("SubSample Correlation Matrix \n (use for reference)")
# plt.show()

###Boxplots to see the differences between the full population and the subsample

##Negative correlations

# f2, axes = plt.subplots(ncols=4, figsize=(20,4))

# sns.boxplot(x='Class',y='V17',data=balanced_df,ax=axes[0])
# axes[0].set_title('Class v V17 NC')

# sns.boxplot(x='Class',y='V14',data=balanced_df,ax=axes[1])
# axes[1].set_title('Class v V14 NC')

# sns.boxplot(x='Class',y='V12',data=balanced_df,ax=axes[2])
# axes[2].set_title('Class v V12 NC')

# sns.boxplot(x='Class',y='V10',data=balanced_df,ax=axes[3])
# axes[3].set_title('Class v V10 NC')


# ##Positive correlations

# f3, axes2 = plt.subplots(ncols=4, figsize=(20,4))

# sns.boxplot(x='Class',y='V19',data=balanced_df,ax=axes2[0])
# axes[0].set_title('Class v V19 PC')

# sns.boxplot(x='Class',y='V11',data=balanced_df,ax=axes2[1])
# axes[1].set_title('Class v V11 PC')

# sns.boxplot(x='Class',y='V4',data=balanced_df,ax=axes2[2])
# axes[2].set_title('Class v V4 PC')

# sns.boxplot(x='Class',y='V2',data=balanced_df,ax=axes2[3])
# axes[3].set_title('Class v V2 PC')

# plt.show()

###Visualizing distributions
## Negative correlations

# fig, ax =plt.subplots(2,4,figsize=(20,10))

# sns.distplot(balanced_df['V17'].loc[balanced_df['Class']==1].values,ax=ax[0][0],fit=norm)
# ax[0][0].set_title('V17 Distribution \n (Fraud Transactions)')

# sns.distplot(balanced_df['V14'].loc[balanced_df['Class']==1].values,ax=ax[0][1],fit=norm,color='red')
# ax[0][1].set_title('V14 Distribution \n (Fraud Transactions)')

# sns.distplot(balanced_df['V12'].loc[balanced_df['Class']==1].values,ax=ax[0][2],fit=norm,color='green')
# ax[0][2].set_title('V12 Distribution \n (Fraud Transactions)')

# sns.distplot(balanced_df['V10'].loc[balanced_df['Class']==1].values,ax=ax[0][3],fit=norm,color='orange')
# ax[0][3].set_title('V10 Distribution \n (Fraud Transactions)')

# ## Positive correlations

# sns.distplot(balanced_df['V19'].loc[balanced_df['Class']==1].values,ax=ax[1][0],fit=norm)
# ax[1][0].set_title('V19 Distribution \n (Fraud Transactions)')

# sns.distplot(balanced_df['V11'].loc[balanced_df['Class']==1].values,ax=ax[1][1],fit=norm,color='red')
# ax[1][1].set_title('V11 Distribution \n (Fraud Transactions)')

# sns.distplot(balanced_df['V4'].loc[balanced_df['Class']==1].values,ax=ax[1][2],fit=norm,color='green')
# ax[1][2].set_title('V4 Distribution \n (Fraud Transactions)')

# sns.distplot(balanced_df['V2'].loc[balanced_df['Class']==1].values,ax=ax[1][3],fit=norm,color='orange')
# ax[1][3].set_title('V2 Distribution \n (Fraud Transactions)')

# plt.show()


###Removing extreme outliers

## V14

V14_fraud_array=balanced_df['V14'].loc[balanced_df['Class']==1].values
q25,q75=np.percentile(V14_fraud_array,25),np.percentile(V14_fraud_array,75)
#print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
V14_iqr=q75-q25
#print('iqr: {}'.format(V14_iqr))

V14_cutoff=V14_iqr*1.5
V14_lower, V14_upper = q25 - V14_cutoff, q75 + V14_cutoff #np.mean(V14_fraud_array) - V14_cutoff/2, np.mean(V14_fraud_array) + V14_cutoff/2
# print('Cut Off: {}'.format(V14_cutoff))
# print('V14 Lower: {}'.format(V14_lower))
# print('V14 Upper: {}'.format(V14_upper))

outliers_V14=[x for x in V14_fraud_array if x < V14_lower or x > V14_upper]
#print(outliers_V14)

balanced_df_no_outliers=balanced_df.drop(balanced_df[(balanced_df['V14']< V14_lower) | (balanced_df['V14']>V14_upper)].index)
#print(balanced_df_no_outliers.shape)



## V12

V12_fraud_array=balanced_df['V12'].loc[balanced_df['Class']==1].values
q25,q75=np.percentile(V12_fraud_array,25),np.percentile(V12_fraud_array,75)
#print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
V12_iqr=q75-q25
#print('iqr: {}'.format(V12_iqr))

V12_cutoff=V12_iqr*1.5
V12_lower, V12_upper = q25 - V12_cutoff, q75 + V12_cutoff #np.mean(V12_fraud_array) - V12_cutoff/2, np.mean(V12_fraud_array) + V12_cutoff/2
# print('Cut Off: {}'.format(V12_cutoff))
# print('V12 Lower: {}'.format(V12_lower))
# print('V12 Upper: {}'.format(V12_upper))

outliers_V12=[x for x in V12_fraud_array if x < V12_lower or x > V12_upper]
#print(outliers_V14)

balanced_df_no_outliers=balanced_df_no_outliers.drop(balanced_df_no_outliers[(balanced_df_no_outliers['V12']< V12_lower) | (balanced_df_no_outliers['V12']>V12_upper)].index)
#print(balanced_df_no_outliers.shape)



## V10

V10_fraud_array=balanced_df['V10'].loc[balanced_df['Class']==1].values
q25,q75=np.percentile(V10_fraud_array,25),np.percentile(V10_fraud_array,75)
#print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
V10_iqr=q75-q25
#print('iqr: {}'.format(V10_iqr))

V10_cutoff=V10_iqr*1.5
V10_lower, V10_upper = q25 - V10_cutoff, q75 + V10_cutoff #np.mean(V10_fraud_array) - V10_cutoff/2, np.mean(V10_fraud_array) + V10_cutoff/2
# print('Cut Off: {}'.format(V10_cutoff))
# print('V12 Lower: {}'.format(V10_lower))
# print('V12 Upper: {}'.format(V10_upper))

outliers_V10=[x for x in V10_fraud_array if x < V10_lower or x > V10_upper]
#print(outliers_V14)

balanced_df_no_outliers=balanced_df_no_outliers.drop(balanced_df_no_outliers[(balanced_df_no_outliers['V10']< V10_lower) | (balanced_df_no_outliers['V10']>V10_upper)].index)
#print(balanced_df_no_outliers.shape)

### Check if it's more normalized

# f,ax=plt.subplots(1,3,figsize=(20,6))

# sns.distplot(balanced_df_no_outliers['V14'].loc[balanced_df_no_outliers['Class']==1].values,ax=ax[0],fit=norm,color='red')

# sns.distplot(balanced_df_no_outliers['V12'].loc[balanced_df_no_outliers['Class']==1].values,ax=ax[1],fit=norm,color='blue')

# sns.distplot(balanced_df_no_outliers['V10'].loc[balanced_df_no_outliers['Class']==1].values,ax=ax[2],fit=norm,color='green')

# plt.show()


### See clustering with different methods

balanced_X=balanced_df_no_outliers.drop('Class',axis=1)
balanced_y=balanced_df_no_outliers['Class']
##t-SNE

# t0=time.time()
# X_reduced_tsne=TSNE(n_components=2,random_state=42).fit_transform(balanced_X.values)
# t1=time.time()
# print('t-SNE took {:.2} seconds.'.format(t1-t0))

# ##PCA
# t0=time.time()
# X_reduced_PCA=PCA(n_components=2,random_state=42).fit_transform(balanced_X.values)
# t1=time.time()
# print('PCA took {:.2} seconds.'.format(t1-t0))

# ##Truncated SVD
# t0=time.time()
# X_reduced_TSVD=TruncatedSVD(n_components=2,algorithm='randomized',random_state=42).fit_transform(balanced_X.values)
# t1=time.time()
# print('Truncated SVD took {:.2} seconds.'.format(t1-t0))

##Showing the clusters for the three methods

# fig,ax=plt.subplots(1,3,figsize=(20,6))
# fig.suptitle('Clusters from various DR methods')

# blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
# red_patch = mpatches.Patch(color='#AF0000', label='Fraud')

# ##t-SNE scatterplot
# ax[0].scatter(X_reduced_tsne[:,0],X_reduced_tsne[:,1],c=(balanced_y==0),cmap='coolwarm',label='No Fraud',linewidths=2)
# ax[0].scatter(X_reduced_tsne[:,0],X_reduced_tsne[:,1],c=(balanced_y==1),cmap='coolwarm',label='Fraud',linewidths=2)
# ax[0].set_title('t-SNE')
# ax[0].grid(True)

# ax[0].legend(handles=[blue_patch,red_patch])

# ##PCA scatterplot
# ax[1].scatter(X_reduced_PCA[:,0], X_reduced_PCA[:,1], c=(balanced_y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
# ax[1].scatter(X_reduced_PCA[:,0], X_reduced_PCA[:,1], c=(balanced_y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
# ax[1].set_title('PCA', fontsize=14)

# ax[1].grid(True)

# ax[1].legend(handles=[blue_patch, red_patch])

# ##TSVD scatterplot
# ax[2].scatter(X_reduced_TSVD[:,0], X_reduced_TSVD[:,1], c=(balanced_y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
# ax[2].scatter(X_reduced_TSVD[:,0], X_reduced_TSVD[:,1], c=(balanced_y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
# ax[2].set_title('Truncated SVD', fontsize=14)

# ax[2].grid(True)

# ax[2].legend(handles=[blue_patch, red_patch])
# plt.show()

###### This also works, using .scatterplot instead of .scatter
# fig,ax=plt.subplots(1,3,figsize=(20,6))
# fig.suptitle('Clusters from various DR methods')

# sns.scatterplot(x=X_reduced_tsne[:,0],y=X_reduced_tsne[:,1],hue=balanced_y,palette='RdBu')
# plt.show()


### Splitting data into train and test sets
X_train, X_test, y_train, y_test =train_test_split(balanced_X,balanced_y, test_size=0.2,random_state=42)

X_train=X_train.values
X_test=X_test.values
y_train=y_train.values
y_test=y_test.values


### Dictionary of classifiers
classifiers={'Logistic_Regression':LogisticRegression(solver='liblinear'),'K_Nearest':KNeighborsClassifier(),'Support Vector Classifier':SVC(gamma='auto'),'Decision_Tree': DecisionTreeClassifier(),'Random_Forest':RandomForestClassifier(n_estimators=10)}

### Training scores for all the classifiers
for key,classifier in classifiers.items():
    classifier.fit(X_train,y_train)
    training_score=cross_val_score(classifier,X_train,y_train,cv=5)
    print(key, 'has a training score of: {:.4} %.'.format(training_score.mean()*100))

# LR=LogisticRegression().fit(X_train,y_train)
# print(LR.score(X_test,y_test))
# print(cross_val_score(LogisticRegression(),X_train,y_train,cv=5).mean())

### Choose the best parameters for the estimators -- hypertunning

## Logistic regression

log_reg_param={'penalty':['l1','l2'],'C':[0.001,0.01,0.1,1,10,100,1000]}

grid_log_reg=GridSearchCV(LogisticRegression(),log_reg_param)
grid_log_reg.fit(X_train,y_train)
log_reg=grid_log_reg.best_estimator_

## K_Nearest_neighbors

knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knears=GridSearchCV(KNeighborsClassifier(),knears_params)
grid_knears.fit(X_train,y_train)

knears_neighbors=grid_knears.best_estimator_

## Support Vector Classifier

scv_params={'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc=GridSearchCV(SVC(),scv_params)
grid_svc.fit(X_train,y_train)
svc=grid_svc.best_estimator_

## Decision tree

tree_params={"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), "min_samples_leaf": list(range(5,7,1))}

grid_tree=GridSearchCV(DecisionTreeClassifier(),tree_params)
grid_tree.fit(X_train,y_train)
tree_clf=grid_tree.best_estimator_

## Random Forest -- Nothing, I would have to check the parameters to hypertune

### Cross Validation scores for after the hypertuning

# log_reg_score=cross_val_score(log_reg,X_train,y_train)
# print('The cross-validation score for Logistic Regression is {:.4} %.'.format(log_reg_score.mean()*100) )

# knears_score=cross_val_score(knears_neighbors,X_train,y_train)
# print('The cross-validation score for K Nearest Neighbors is {:.4} %.'.format(knears_score.mean()*100) )

# svc_score=cross_val_score(svc,X_train,y_train)
# print('The cross-validation score for SVC is {:.4} %.'.format(svc_score.mean()*100) )

# tree_score=cross_val_score(tree_clf,X_train,y_train)
# print('The cross-validation score for Decision Tree is {:.4} %.'.format(tree_score.mean()*100) )


### Test scores

# log_reg_score=cross_val_score(log_reg,X_test,y_test)
# print('The test score for Logistic Regression is {:.4} %.'.format(log_reg_score.mean()*100) )

# knears_score=cross_val_score(knears_neighbors,X_test,y_test)
# print('The test score for K Nearest Neighbors is {:.4} %.'.format(knears_score.mean()*100) )

# svc_score=cross_val_score(svc,X_test,y_test)
# print('The test score for SVC is {:.4} %.'.format(svc_score.mean()*100) )

# tree_score=cross_val_score(tree_clf,X_test,y_test)
# print('The test score for Decision Tree is {:.4} %.'.format(tree_score.mean()*100) )

