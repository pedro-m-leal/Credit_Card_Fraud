#Libraries for Data Visualization

import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import numpy as np 
import seaborn as sns
import time

from sklearn.preprocessing import StandardScaler, RobustScaler #Scaling Time and Amount
from mpl_toolkits import mplot3d


#### Importing Data

input_data=pd.read_csv('creditcard.csv')
df=input_data.copy()

frauds=df.loc[df['Class']==1] ## 491 fraudulent transactions
legit=df.loc[df['Class']==0] ## 284315 legitimate transactions

#### Check missing data in every column
# for i in df.columns.values:
#     print('Column', i, 'has', df[i].isnull().sum(), 'missing values.')
 # No missing values


#### Data Visualization

## Countplot to see how imbalanced the dataset is
# sns.countplot(x='Class',data=df)
# plt.show()

#### Since the data is imbalanced, we will try to use an anomaly detection algorithm. We will assume the features are Gaussian for the first quick implementation

#### Dataset division
#### Training set - 60% of legitimate transactions
#### Cross Validation set - 20% of legitimate transactions + 50% of fraudulent transactions
#### Test set - 20% of legitimate transactions + 50% of fraudulent transactions

shuffled_frauds=frauds.sample(frac=1)
shuffled_legit=legit.sample(frac=1)
n_frauds=shuffled_frauds['Class'].size
n_legit=shuffled_legit['Class'].size
#print(n_legit, n_frauds)
n_test=round(n_legit*0.6)
train_set=shuffled_legit.iloc[[i for i in range(n_test)]]
shuffled_legit=shuffled_legit.drop(index=train_set.index.values) #delete already used data

n_legit=shuffled_legit['Class'].size
#print(n_legit)
n_cv=round(n_legit*0.5)
cv_legit=shuffled_legit.iloc[[i for i in range(n_cv)]]
cv_frauds=shuffled_frauds.iloc[[i for i in range(round(n_frauds/2))]]
cv_set=pd.concat([cv_legit, cv_frauds])
cv_set=cv_set.sample(frac=1) #shuffle the dataset

shuffled_legit=shuffled_legit.drop(index=cv_legit.index.values) #delete already used data
shuffled_frauds=shuffled_frauds.drop(index=cv_frauds.index.values) #delete already used data

n_legit=shuffled_legit['Class'].size
n_frauds=shuffled_frauds['Class'].size
#print(n_legit,n_frauds)


test_set=pd.concat([shuffled_legit, shuffled_frauds])
test_set=test_set.sample(frac=1)

shuffled_legit=shuffled_legit.drop(index=shuffled_legit.index.values) #delete already used data
shuffled_frauds=shuffled_frauds.drop(index=shuffled_frauds.index.values) #delete already used data

# n_legit=shuffled_legit['Class'].size
# n_frauds=shuffled_frauds['Class'].size
#print(n_legit,n_frauds)

## All the sets are now created

#### The nest step is to create the multivariate Gaussian distribution

rb_scaler=RobustScaler()

train_timeless=train_set.drop(['Time','Class'],axis=1)
train_timeless['scaled_Amount']=rb_scaler.fit_transform(train_timeless['Amount'].values.reshape(-1,1))
train_timeless=train_timeless.drop('Amount',axis=1)

means=train_timeless.mean().values.tolist()
cov_matrix=train_timeless.cov()
# print(cov_matrix)

# sns.heatmap(cov_matrix)
# plt.show()

def multVarGauss(X,mean=means,cov=cov_matrix):
    ## Determines the probability of X, given a multivariate Gaussian distribution with mean and cov

    if type(X)==list:
        X=np.asarray(X)
    else:
        X=np.asarray(X.values.tolist())
    mean=np.asarray(mean)
    cov=np.asarray(cov)
    dim=len(mean)
    exponent=-0.5*np.dot((X-mean),np.dot(np.linalg.inv(cov),(X-mean).T))
    # print(X)
    # print(mean)
    # print(exponent,dim)
    prefactor=np.sqrt((2*np.pi)**dim*np.linalg.det(cov))
    # print(prefactor)
    f=(1/prefactor)*np.exp(exponent)
    return f


### Testing the function on random point from the same distribution
# for i in range(20):
#     train_dist=np.random.multivariate_normal(means,cov_matrix)
#     print(multVarGauss(train_dist.tolist()))


#### Now that we have defined the training ('unlabeled') data to the multivariate Gaussian distribution, we proceed do evaluate the Gaussian in the CV set

cv_calc=cv_set.drop(['Class','Time','Amount'],axis=1)
cv_calc['scaled_Amount']=rb_scaler.fit_transform(cv_set['Amount'].values.reshape(-1,1))
cv_results=cv_set['Class']



probs=np.zeros(cv_results.size)

for i in range(cv_results.size):
    probs[i]=multVarGauss(cv_calc.iloc[i])

#print(probs[0:10])

def decision(probability,threshold):
    if probability<threshold:
        return 1
    else:
        return 0




#print(preds[0:10])

#### Now all we need to do is to calculate the number of false positives, false negatives, true positives and true negatives

epsilons=10**(-np.asarray([i for i in np.arange(5,250,0.5,dtype=float)][::-1]))

cv_results=cv_results.values.tolist()
confusion_list=np.zeros(4) # 0-> fn / 1-> tn / 2-> tp / 3-> fp
precisions=np.zeros(len(epsilons))
recalls=np.zeros(len(epsilons))
f1s=np.zeros(len(epsilons))
epsilon_count=0
for epsilon in epsilons:

    preds=np.zeros(len(cv_results))

    for i in range(len(cv_results)):
        preds[i]=decision(probs[i],epsilon)

    
    


    results=2*preds-cv_results+1
    results=[int(k) for k in results] #make all elements integers instead of floats

    for result in results:
        confusion_list[result]+=1

    ## Initial naive method. The one above is much better
    # tp_count=0
    # tn_count=0
    # fp_count=0
    # fn_count=0
    # miss=0
    # for i in results:
    #     if i==2:
    #         tp_count+=1
    #     elif i==3:
    #         fp_count+=1
    #     elif i==0:
    #         fn_count+=1
    #     elif i==1:
    #         tn_count+=1
    #     else:
    #         miss+=1

    precision=confusion_list[2]/(confusion_list[2]+confusion_list[3])
    recall=confusion_list[2]/(confusion_list[2]+confusion_list[0])
    f1Score=2*precision*recall/(precision+recall)
    precisions[epsilon_count]=precision
    recalls[epsilon_count]=recall
    f1s[epsilon_count]=f1Score
    epsilon_count+=1
    # print('Precision is:' , precision, '\n', 'Recall is:' , recall, '\n', 'F1 Score is: ', f1Score)

    #print('TP :', confusion_list[2], '\n', 'TN :', confusion_list[1], '\n', 'FP :', confusion_list[3], '\n', 'FN :', confusion_list[0])

# plt.figure(1)
# plt.plot(-np.log10(epsilons),precisions,'x')
# plt.figure(2)
# plt.plot(-np.log10(epsilons),recalls,'x')
# plt.figure(3)
# plt.plot(-np.log10(epsilons),f1s,'x')
# plt.figure(4)
# plt.plot(recalls,precisions,'x')
# plt.show()


best_epsilon=epsilons[f1s.tolist().index(max(f1s))]
best_f1=max(f1s)

print('best epsilon:' ,best_epsilon, 'best F1 Score:', best_f1)
    
#### Now that we found the best hyperparameter on the CV set, let's test it on the test set

test_calc=test_set.drop(['Time','Amount','Class'],axis=1)
test_calc['scaled_Amount']=rb_scaler.fit_transform(test_set['Amount'].values.reshape(-1,1))
test_results=test_set['Class']

test_probs=np.zeros(test_results.size)

for i in range(test_results.size):
    test_probs[i]=multVarGauss(test_calc.iloc[i])

test_preds=np.zeros(len(test_results))
test_results=test_results.values.tolist()

for i in range(len(test_results)):
    test_preds[i]=decision(test_probs[i],best_epsilon)

test_confusion_list=np.zeros(4) # 0-> fn / 1-> tn / 2-> tp / 3-> fp

test_results=2*test_preds-test_results+1
test_results=[int(k) for k in test_results] #make all elements integers instead of floats

for result in test_results:
    test_confusion_list[result]+=1

test_precision=test_confusion_list[2]/(test_confusion_list[2]+test_confusion_list[3])
test_recall=test_confusion_list[2]/(test_confusion_list[2]+test_confusion_list[0])
test_f1Score=2*test_precision*test_recall/(test_precision+test_recall)

print('TP :', test_confusion_list[2], '\n', 'TN :', test_confusion_list[1], '\n', 'FP :', test_confusion_list[3], '\n', 'FN :', test_confusion_list[0])

print('Precision is:' , test_precision, '\n', 'Recall is:' , test_recall, '\n', 'F1 Score is: ', test_f1Score)


#### These results are not very good... ######

    