#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>
# 

# In this notebook we try to practice all the classification algorithms that we have learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Let's first load required libraries:
# 

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset
# 

# This dataset is about past loans. The **Loan_train.csv** data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# | -------------- | ------------------------------------------------------------------------------------- |
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |
# 

# Let's download the dataset
# 

# In[2]:


get_ipython().system('wget -O loan_train.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv')


# ### Load Data From CSV File
# 

# In[3]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[4]:


df.shape


# ### Convert to date time object
# 

# In[5]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 

# Let’s see how many of each class is in our data set
# 

# In[6]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection
# 

# Let's plot some columns to underestand data better:
# 

# In[7]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[8]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[9]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction
# 

# ### Let's look at the day of the week people get the loan
# 

# In[10]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week don't pay it off, so let's use Feature binarization to set a threshold value less than day 4
# 

# In[11]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values
# 

# Let's look at gender:
# 

# In[12]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Let's convert male to 0 and female to 1:
# 

# In[13]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding
# 
# #### How about education?
# 

# In[14]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Features before One Hot Encoding
# 

# In[15]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame
# 

# In[16]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature Selection
# 

# Let's define feature sets, X:
# 

# In[17]:


X = Feature
X[0:5]


# What are our lables?
# 

# In[18]:


y = df['loan_status'].values
y[0:5]


# ## Normalize Data
# 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split)
# 

# In[19]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# # Classification
# 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# 
# *   K Nearest Neighbor(KNN)
# *   Decision Tree
# *   Support Vector Machine
# *   Logistic Regression
# 
# \__ Notice:\__
# 
# *   You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# *   You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# *   You should include the code of the algorithm in the following cells.
# 

# # K Nearest Neighbor(KNN)
# 
# Notice: You should find the best k to build the model with the best accuracy.\
# **warning:** You should not use the **loan_test.csv** for finding the best k, however, you can split your train_loan.csv into train and test to find the best **k**.
# 

# In[28]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)


# In[29]:


from sklearn import metrics
import matplotlib.pyplot as plt

Ks=20
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks), mean_acc - 1 * std_acc, mean_acc +1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks), mean_acc - 3 * std_acc, mean_acc +3 * std_acc, alpha=0.10, color="red")
plt.legend(('Accuracy', '+/- 1sxtd', '+/- 3xstd'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show


# In[27]:


print("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)

# K=7
K7=7
neigh7 = KNeighborsClassifier(n_neighbors = K7).fit(X_train, y_train)
yhat_7 = neigh7.predict(X_test)
print ("Train set accuracy with K=7:", metrics.accuracy_score(y_train, neigh7.predict(X_train)))
print ("Test set accuracy with K=7:", metrics.accuracy_score(y_test, yhat_7))


# # Decision Tree
# 

# In[30]:


from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split (X, y, test_size=0.3, random_state=3)
print('Shape of X Test Set{}'. format(X_testset.shape),'&', 'Shape of Y test set{}'.format(y_testset.shape))


# In[31]:


from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
LoanTree = DecisionTreeClassifier(criterion="entropy", max_depth=4)
LoanTree.fit(X_trainset, y_trainset)
predTree = LoanTree.predict(X_testset)
print (predTree[0:10])


# In[32]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's accuracy:", metrics.accuracy_score(y_testset, predTree))
tree.plot_tree(LoanTree)
plt.show()


# # Support Vector Machine
# 

# In[33]:


from sklearn import svm
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
yhat[0:5]


# In[34]:


# Using RBM
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score
print('The Accuracy of the SVM model using RBM is:', f1_score (y_test, yhat, average='weighted'))
print('The Jaccard score of the SVM model using RBM is:', jaccard_score(y_test, yhat, pos_label='PAIDOFF'))

# Using Linear Kernel
clf_1=svm.SVC(kernel='linear')
clf_1.fit(X_train, y_train)
yhat_1 = clf_1.predict(X_test)
print("The Accuracy of the SVM model using Linear is: %.4f" % f1_score (y_test, yhat_1, average='weighted'))
print('The Jaccard score of the SVM model using Linear is:', jaccard_score(y_test, yhat_1, pos_label='PAIDOFF'))


# In[36]:


from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Compute confusion matrix
matrix = confusion_matrix(y_test, yhat, labels=['PAIDOFF', 'COLLECTION'])
np.set_printoptions(precision = 2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(matrix, classes = ['PAIDOFF', 'COLLECTION'], normalize= False,  title='Confusion matrix')


# # Logistic Regression
# 

# In[85]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder


# In[86]:


def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_e = le.transform(y_train)
    y_test_e = le.transform(y_test)
    return y_train_e, y_test_e
 

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

y_train_e, y_test_e = prepare_targets(y_train, y_test)


# In[87]:


LR = LogisticRegression(C = 0.01, solver = 'sag').fit(X_train, y_train_enc)

yhat = LR.predict(X_test)

yhat_prob = LR.predict_proba(X_test)
yhat_prob

# print metrics
print("LR's Accuracy: ", round(metrics.accuracy_score(y_test_enc, yhat), 3))


# In[80]:


from sklearn.metrics import jaccard_score
jaccard_score(y_test, yhat_LR, pos_label='PAIDOFF')


# In[88]:


round(metrics.accuracy_score(y_test, yhat_LR), 3)


# In[89]:


matrix = confusion_matrix(y_test_enc, yhat, labels = [1,0])
np.set_printoptions(precision = 2)

plt.figure()
plot_confusion_matrix(matrix, classes=['PAIDOFF','COLLECTION'],normalize= False,  title = 'Confusion matrix')


# # Model Evaluation using Test set
# 

# In[73]:


from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder


# First, download and load the test set:
# 

# In[42]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation
# 

# In[61]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[62]:


test_df['due_date'] = pd.to_datetime(df['due_date'])
test_df['effective_date'] = pd.to_datetime(df['effective_date'])
test_df.head()
test_df['loan_status'].value_counts()                        


# In[63]:


test_df['Gender'] = test_df['Gender'].replace(to_replace=['male','female'], value = [0,1])
test_df.head()


# In[64]:


test_df['effective_date'] = pd.to_datetime(test_df['effective_date']).dt.dayofweek
test_df['weekend']=test_df['effective_date'].apply(lambda x: 1 if (x>3) else 0)
df.head()


# In[65]:


Feature_test = test_df[['Principal', 'terms', 'age', 'weekend', 'Gender']]
Feature_test = pd.concat([Feature_test, pd.get_dummies(df['education'])], axis=1)
Feature_test = Feature_test.rename(columns = {'Bechalor' : 'Bachelor'})
Feature_test.drop(['Master or Above'], axis = 1,inplace=True)
Feature_test.dtypes


# In[68]:


Feature_test = Feature_test.dropna(how = 'any', axis = 0) 
Feature_test = Feature_test.astype('int')
Feature_test.shape



# In[90]:


X_test = Feature_test
y_test = test_df['loan_status'].values

KNClassifier = neigh.predict(X_test)
DTree = LoanTree.predict(X_test)
SVMClassifier = clf.predict(X_test)
LRClassifier = LR.predict(X_test)


# In[91]:


from tabulate import tabulate

d = [ 
      ["KNN", 
       jaccard_score(y_test, KNClassifier, pos_label = 'PAIDOFF'),
       f1_score(y_test, KNClassifier, average = 'weighted'),
       "NA"],
      ["Decision Tree",
       jaccard_score(y_test, DTree, pos_label = 'PAIDOFF'),
       f1_score(y_test, DTree, average = 'weighted'),
       "NA"],
      ["SVM",
       jaccard_score(y_test, SVMClassifier, pos_label = 'PAIDOFF'),
       f1_score(y_test, SVMClassifier, average = 'weighted'),
       "NA"],
      ["Logistic Regression",
       jaccard_score(LabelEncoder().fit(y_test).transform(y_test), LRClassifier, pos_label = 0),
       f1_score(LabelEncoder().fit(y_test).transform(y_test), LRClassifier, average = 'weighted'),
       log_loss(LabelEncoder().fit(y_test).transform(y_test), LR.predict_proba(X_test))]
    ]

print(tabulate(d, headers = ["Algorithm", "Jaccard", "F1-score", "LogLoss"]))


# In[ ]:





# # Report
# 
# You should be able to report the accuracy of the built model using different evaluation metrics:
# 

# | Algorithm          | Jaccard | F1-score | LogLoss |
# | ------------------ | ------- | -------- | ------- |
# | KNN                | ?       | ?        | NA      |
# | Decision Tree      | ?       | ?        | NA      |
# | SVM                | ?       | ?        | NA      |
# | LogisticRegression | ?       | ?        | ?       |
# 

# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# ## Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By    | Change Description                                                             |
# | ----------------- | ------- | ------------- | ------------------------------------------------------------------------------ |
# | 2020-10-27        | 2.1     | Lakshmi Holla | Made changes in import statement due to updates in version of  sklearn library |
# | 2020-08-27        | 2.0     | Malika Singla | Added lab to GitLab                                                            |
# 
# <hr>
# 
# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# 
# <p>
# 
