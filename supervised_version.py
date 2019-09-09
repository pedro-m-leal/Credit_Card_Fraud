#Libraries for Data Visualization

import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import numpy as np 
import seaborn as sns
import time

from sklearn.preprocessing import StandardScaler, RobustScaler #Scaling Time and Amount
from mpl_toolkits import mplot3d
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, precision_recall_fscore_support, fbeta_score, classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier



### Visualization functions


def histograms(series):
    if isinstance(series,pd.Series):
        mn=int(min(series))
        mx=int(max(series))
        n_bins=len(range(mn,mx))+1
        ax=sns.distplot(series,n_bins,kde=False)
        ax.set_title('{}'.format(series.name))
        plt.show()
    elif isinstance(series,pd.DataFrame):
        cols=series.columns
        length=len(cols)
        if length%4==0:
            n_rows=4
            n_columns=length//4
        else:
            n_rows=4
            n_columns=length//4+1
        
        fig, axis=plt.subplots(nrows=n_rows,ncols=n_columns)
        for i in range(length):
            mn=int(min(series[cols[i]]))
            mx=int(max(series[cols[i]]))
            n_bins=len(range(mn,mx))+1
            sns.distplot(series[cols[i]],n_bins,kde=False,ax=axis.flatten()[i])
            axis.flatten()[i].set_title(cols[i])
        plt.show()
    else:
        raise ValueError('Please use pd.Series or pd.DataFrame as argument.')
def boxplots(series):
    if isinstance(series,pd.Series):
        ax=sns.boxplot(series)
        ax.set_title('{}'.format(series.name))
        plt.show()
    elif isinstance(series,pd.DataFrame):
        cols=series.columns
        length=len(cols)
        if length%2==0:
            n_rows=2
            n_columns=length//2
        else:
            n_rows=2
            n_columns=length//2+1

        fig, axis=plt.subplots(nrows=n_rows,ncols=n_columns)
        for i in range(length):
            sns.boxplot(series[cols[i]],ax=axis.flatten()[i])
            axis.flatten()[i].set_title(cols[i])
        plt.show()
    else:
        raise ValueError('Please use pd.Series or pd.DataFrame as argument.')


#### Importing Data

input_data=pd.read_csv('creditcard.csv')
df=input_data.copy()

# Let us first get the descriptive statistics on the dataset

print('-> Types of the variables:')
print(df.dtypes)

print('-> Descriptive statistics of the dataset')
print(df.drop('Time',axis=1).describe())


# Plot of the mean of each feature
plt.plot(range(df.drop(['Time','Amount','Class'],axis=1).describe().loc['mean'].shape[0]),df.drop(['Time','Amount','Class'],axis=1).describe().loc['mean'].tolist())
plt.show()

# Plot of the std of each feature
plt.plot(range(df.drop(['Time','Amount','Class'],axis=1).describe().loc['mean'].shape[0]),df.drop(['Time','Amount','Class'],axis=1).describe().loc['std'].tolist())
plt.show()


# Let us check for missing values

x=sum(df.isnull().sum())
print('-> Is any value missing from the database?', '\n')
print('-> No! \n') if x==0 else print('-> Yes...', '\n') 

if x!=0:
    print('-> These are the rows with missing values:','\n')
    mv_rows=df[df.isnull().any(axis=1)]
    print(mv_rows)
    df.drop(mv_rows.index.tolist(),inplace=True)
    print('\n-> There should be no missing values now. \n')
    print(df.isnull().sum(), '\n \n \n')

# Since there are two classes, let us see the amount of instances of each class

sns.countplot(df['Class'],palette='pastel')
plt.show()
print(df['Class'].value_counts().loc[1])
print('\n-> There are {} total frauds detected. That amounts to {:.3f} % of the transactions'.format(df['Class'].value_counts().loc[1],df['Class'].value_counts().loc[1]/sum(df['Class'].value_counts().tolist())*100))

# Let's now split the data

y=df['Class']
X=df.drop(['Class','Time'],axis=1)

sss=StratifiedShuffleSplit(n_splits=1,test_size=0.2)


for train_cv_index, test_index in sss.split(X,y):
    X_train_cv,X_test = X.loc[train_cv_index], X.loc[test_index]
    y_train_cv, y_test = y.loc[train_cv_index], y.loc[test_index]

print('\n-> There are {} total frauds detected in the training set. That amounts to {:.3f} % of the transactions'.format(y_train_cv.value_counts().loc[1],y_train_cv.value_counts().loc[1]/sum(y_train_cv.value_counts().tolist())*100))

print('\n-> There are {} total frauds detected in the test set. That amounts to {:.3f} % of the transactions'.format(y_test.value_counts().loc[1],y_test.value_counts().loc[1]/sum(y_test.value_counts().tolist())*100))

X_train_cv.reset_index(drop=True, inplace=True) #Resetting indices
y_train_cv.reset_index(drop=True, inplace=True) #Resetting indices



################################################################################################
#
#
## MODEL FITTING AND TESTING
#
#
################################################################################################

## This functions defines the best value for a certain parameter, for a given estimator

def best_model_params(model,param_list,undersample,*args,**kwargs):#,param,class_weight=None):
    """
    *args must contain the name of the parameter to evaluate

    **Kwargs is optional and may contain other arguments to pass to the model
    """
    param_d=dict()
    for arg in args:
        param_d[arg]=arg
        param=arg
    av_scores=[]
    for i in param_list:
        skf=StratifiedKFold(n_splits=5)
        scores=[]
        precisions=[]
        recalls=[]
        supports=[]
        #scores2=[]
        for train_index, cv_index in skf.split(X_train_cv,y_train_cv):
            X_train, X_cv = X_train_cv.loc[train_index], X_train_cv.loc[cv_index]
            y_train, y_cv = y_train_cv.loc[train_index], y_train_cv.loc[cv_index]

            #Undersampling
            if undersample!=None:
                X_train['y']=y_train
                X_train_0=X_train[X_train['y']==0]
                X_train_1=X_train[X_train['y']==1]
                X_train_0=X_train_0.sample(n=undersample*X_train_1['y'].size)
                X_train=pd.concat([X_train_0,X_train_1],ignore_index=True)
                X_train=X_train.sample(frac=1)
                y_train=X_train['y']
                X_train.drop(['y'],axis=1,inplace=True)


            #Scale the amount individually for each iteration of the split
            rb=RobustScaler()
            rb.fit(X_train['Amount'].values.reshape(-1,1))
            X_train['Amount']=rb.transform(X_train['Amount'].values.reshape(-1,1))
            X_cv['Amount']=rb.transform(X_cv['Amount'].values.reshape(-1,1))

            param_d[param]=i
            md=model(**param_d,**kwargs)
            md.fit(X_train,y_train)
            preds=md.predict(X_cv)
            score=precision_recall_fscore_support(y_cv,preds,beta=1.5)
            #score2=fbeta_score(y_cv,preds,beta=1.0)
            precision=score[0][1]
            recall=score[1][1]
            f1=score[2][1]
            support=score[3][1]
            scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)
            supports.append(support)
            #scores2.append(score2)
        av_score=np.asarray(scores).mean()
        av_precision=np.asarray(precisions).mean()
        av_recall=np.asarray(recalls).mean()
        av_support=np.asarray(supports).mean()
        #av_score2=np.asarray(scores2).mean()
        av_scores.append(av_score)
        print('\n-> The average scores for the parameter {} with value {} are: Fbeta Score {}, precision {}, recall {}, support {}.'.format(param,i,av_score,av_precision,av_recall,av_support))
    best_score=np.asarray(av_scores).max()
    best_param=param_list[np.argmax(np.asarray(av_scores))]
    print('\n\n-> The best value for parameter {} is {} with an Fbeta Score of {}.\n'.format(param,best_param,best_score))
    return best_param


#####################
#
# Logistic Regression
#
#####################


try:
    best_params_df=pd.read_csv('best_params_lg.csv')
except:
    print('\n\n-> A DataFrame with the best parameters did not exist. The optimization will run and create the file.')
    best_c=best_model_params(LogisticRegression,[0.01,0.1,1,10,100],None,'C',solver='liblinear',penalty='l2')

    best_weights=best_model_params(LogisticRegression,[{0:1,1:1},{0:1,1:5},{0:1,1:10},{0:1,1:100},{0:1,1:1000}],None,'class_weight',C=best_c,solver='liblinear',penalty='l2')

    best_c=best_model_params(LogisticRegression,[0.001,0.01,0.1,1,10,100], None,'C',solver='liblinear',penalty='l2',class_weight=best_weights)

    best_weights=best_model_params(LogisticRegression,[{0:1,1:1},{0:1,1:5},{0:1,1:10},{0:1,1:30},{0:1,1:60},{0:1,1:100}], None,'class_weight',C=best_c,solver='liblinear',penalty='l2')


    best_params_df=pd.DataFrame({'C':[best_c],'Class_Weights0':[best_weights[0]],'Class_Weights1':[best_weights[1]]})
    best_params_df.to_csv(r'best_params_lg.csv')


best_c=best_params_df['C'].tolist()[0]
best_weights={0:best_params_df['Class_Weights0'].tolist()[0],1:best_params_df['Class_Weights1'].tolist()[0]}

## Use the best model on the test data

# Before using the model on the test set, one must perform the scaling of the amount variable in the test set

rb=RobustScaler()
rb.fit(X_train_cv['Amount'].values.reshape(-1,1))
X_test['Amount']=rb.transform(X_test['Amount'].values.reshape(-1,1))

# Now the model

lg=LogisticRegression(C=best_c,class_weight=best_weights,solver='liblinear',penalty='l2')
lg.fit(X_train_cv,y_train_cv)
probs=lg.predict_proba(X_test)
predictions=lg.predict(X_test)
print(classification_report(y_test,predictions))

## Plot the learning curve for the best model


train_sizes, train_scores, test_scores = learning_curve(lg, X_train_cv, y_train_cv, cv=5, n_jobs=1,scoring='f1')
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title('Learning Curves')
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.show()

## Plot the roc curve for the best model

fpr, tpr, thresholds = roc_curve(y_test,probs[:,1],)

plt.figure()
plt.title('ROC curve')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid()
plt.plot(fpr,tpr,color="r")
plt.show()

## Selecting the best threshold for the best model chosen above


def best_threshold(thresholds,probabilities,true_labels):
    pos_probs=probabilities[:,1]
    scores=[]
    for thresh in thresholds:
        y_preds=[1 if i>thresh else 0 for i in pos_probs]
        print('\n============================')
        print('-> Threshold = {}'.format(thresh))
        print('============================\n')
        print(classification_report(true_labels,y_preds))
        score=fbeta_score(true_labels,y_preds,beta=1.5)
        scores.append(score)
    bestScore=max(scores)
    bestThre=thresholds[np.argmax(np.asarray(scores))]
    print('-> The best value for the threshold is {} with an F Beta Score value of {:.3f}.'.format(bestThre,bestScore))
    return bestThre

if 'Threshold' not in best_params_df.columns:
    
    thresholds=[i*0.1+0.1 for i in range(9)]
    bestThr=best_threshold(thresholds,probs,y_test)
    best_params_df['Threshold']=[bestThr]
    best_params_df.to_csv(r'best_params_lg.csv')

bestThr=best_params_df['Threshold'].tolist()[0]
print('The best threshold is {:.1f}.'.format(bestThr))

sns.heatmap(confusion_matrix(y_test,predictions),annot=True)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for the test set')
plt.show()



"""
#####################
#
# SVM
#
#####################


try:
    best_params_df=pd.read_csv('best_params_svm.csv')
except:
    print('\n\n-> A DataFrame with the best parameters did not exist. The optimization will run and create the file.')

    best_c=best_model_params(LinearSVC,[0.01,0.1,1,10,100],None,'C')

    best_weights=best_model_params(LinearSVC,[{0:1,1:1},{0:1,1:5},{0:1,1:10},{0:1,1:100},{0:1,1:1000}],None,'class_weight',C=best_c)

    best_c=best_model_params(LinearSVC,[0.001,0.01,0.1,1,10,100], None,'C',penalty='l2',class_weight=best_weights)

    best_weights=best_model_params(LinearSVC,[{0:1,1:5},{0:1,1:10},{0:1,1:30},{0:1,1:60}], None,'class_weight',C=best_c)


    best_params_df=pd.DataFrame({'C':[best_c],'Class_Weights0':[best_weights[0]],'Class_Weights1':[best_weights[1]]})
    best_params_df.to_csv(r'best_params_svm.csv')


best_c=best_params_df['C'].tolist()[0]
best_weights={0:best_params_df['Class_Weights0'].tolist()[0],1:best_params_df['Class_Weights1'].tolist()[0]}

## Use the best model on the test data

svm=LinearSVC(C=best_c,class_weight=best_weights, max_iter=10000)
svm.fit(X_train_cv,y_train_cv)
predictions=svm.predict(X_test)
print(classification_report(y_test,predictions))

## Plot the learning curve for the best model


train_sizes, train_scores, test_scores = learning_curve(lg, X_train_cv, y_train_cv, cv=5, n_jobs=1,scoring='f1')
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title('Learning Curves')
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.show()

## Plot the roc curve for the best model

fpr, tpr, thresholds = roc_curve(y_test,probs[:,1],)

plt.figure()
plt.title('ROC curve')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid()
plt.plot(fpr,tpr,color="r")
plt.show()


"""

#####################
#
# AdaBoost
#
#####################


try:
    best_params_df=pd.read_csv('best_params_ada.csv')
except:
    print('\n\n-> A DataFrame with the best parameters did not exist. The optimization will run and create the file.')

    best_n=best_model_params(AdaBoostClassifier,[10,20,30,40,50,60,70,80,90,100],100,'n_estimators',base_estimator=DecisionTreeClassifier(max_depth=2))

    best_lr=best_model_params(AdaBoostClassifier,[0.01,0.05,0.1,0.2,0.4,0.6,0.8, 1, 10],100,'learning_rate',n_estimators=best_n,base_estimator=DecisionTreeClassifier(max_depth=2))

    best_n=best_model_params(AdaBoostClassifier,[10,20,30,40,50,60,70,80,90,100], 100,'n_estimators',learning_rate=best_lr,base_estimator=DecisionTreeClassifier(max_depth=2))

    best_lr=best_model_params(AdaBoostClassifier,[0.01,0.05,0.1,0.2,0.4,0.6,0.8, 1, 10], 100,'learning_rate',n_estimators=best_n,base_estimator=DecisionTreeClassifier(max_depth=2))


    best_params_df=pd.DataFrame({'n_estimators':[best_n],'Learning_Rate':[best_lr]})
    best_params_df.to_csv(r'best_params_ada.csv')


best_n=best_params_df['n_estimators'].tolist()[0]
best_lr=best_params_df['Learning_Rate'].tolist()[0]

## Use the best model on the test data. Now the 'Amount' in the test data has already been scaled

ada=AdaBoostClassifier(n_estimators=best_n,learning_rate=best_lr)
ada.fit(X_train_cv,y_train_cv)
predictions=ada.predict(X_test)
print(classification_report(y_test,predictions))

## Plot the learning curve for the best model


train_sizes, train_scores, test_scores = learning_curve(lg, X_train_cv, y_train_cv, cv=5, n_jobs=1,scoring='f1')
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title('Learning Curves')
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.grid()
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.legend(loc="best")
plt.show()

## Plot the roc curve for the best model

fpr, tpr, thresholds = roc_curve(y_test,probs[:,1],)

plt.figure()
plt.title('ROC curve')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.grid()
plt.plot(fpr,tpr,color="r")
plt.show()

sns.heatmap(confusion_matrix(y_test,predictions),annot=True)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for the test set')
plt.show()




# # Let us now make histograms and boxplots for each feature

# histograms(df.drop(['Time','Amount','Class'],axis=1)) # There are clearly outliers that we need to treat
# boxplots(df.drop(['Time','Amount','Class'],axis=1))