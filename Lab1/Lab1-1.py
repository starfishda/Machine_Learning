import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import seaborn as sn
import matplotlib.pyplot as plt

#Calculation kfold's accuracy
def cal(model):
    cv_scores = cross_val_score(model, X, y, cv=10)
    print(cv_scores)
    print('cv_scores mean:{}'.format(np.mean(cv_scores)))
    print()

Dscore = None
Dmodel = None

#Pick Decision Tree's max model
def update(model):
    global Dmodel, Dscore
    cv_scores = cross_val_score(model, X, y, cv=10)
    n_score = np.mean(cv_scores)
    if (Dscore == None):
        Dscore = n_score
        Dmodel = model
    elif(n_score > Dscore):
        Dscore = n_score
        Dmodel = model

Cscore = None
Cmodel = None

#Pick Logistic Regression's max model
def update2(model):
    global Cmodel, Cscore
    cv_scores = cross_val_score(model, X, y, cv=10)
    n_score = np.mean(cv_scores)
    if (Cscore == None):
        Cscore = n_score
        Cmodel = model
    elif(n_score > Cscore):
        Cscore = n_score
        Cmodel = model

Sscore = None
Smodel = None

#Pick SVM's max model
def update3(model):
    global Smodel, Sscore
    cv_scores = cross_val_score(model, X, y, cv=10)
    n_score = np.mean(cv_scores)
    if (Sscore == None):
        Sscore = n_score
        Smodel = model
    elif(n_score > Sscore):
        Sscore = n_score
        Smodel = model

#data read
heart = pd.read_csv("heart.csv")

#classifier attribute to target
X = heart.drop(['target'],axis = 1)
y = heart['target'].values

#scaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

#DecstionTree's all case
for case in['gini', 'entropy']:
    print("DecisionTree = " , case)
    model = DecisionTreeClassifier(criterion = case, max_depth=3, random_state=0)
    cal(model)
    update(model)

#Logistic Regression's all case
for case in['liblinear', 'lbfgs', 'sag']:
    for max in [50, 100,200]:
        print("Logistic = " , "solver :  " , case , " / max_iter : " , max)
        model = LogisticRegression(solver = case, max_iter = max)
        cal(model)
        update2(model)

#SVM's all case
for case in[0.1, 1.0, 10.0]:
    for kernel in['linear', 'poly', 'rbf', 'sigmoid']:
        for gam in[10, 100]:
            print("SVM = " , "C : " , case , " / gamma : " , gam , " / Kernel : " , kernel)
            model = SVC(C = case, gamma = gam, kernel = kernel)
            cal(model)
            update3(model)

#Training max model
Dmodel = Dmodel.fit(X, y)
Cmodel = Cmodel.fit(X, y)
Smodel = Smodel.fit(X, y)

#Decision Tree's confusion matrix
confusion_matrix = pd.crosstab(y, Dmodel.predict(X), rownames=['Actual'], colnames=['Predicted'], margins= True)
sn.heatmap(confusion_matrix, annot=True, fmt = 'd')
plt.show()

#Logistic Regression's confusion matrix
confusion_matrix = pd.crosstab(y, Cmodel.predict(X), rownames=['Actual'], colnames=['Predicted'], margins= True)
sn.heatmap(confusion_matrix, annot=True, fmt = 'd')
plt.show()

#SVN's confusion matrix
confusion_matrix = pd.crosstab(y, Smodel.predict(X), rownames=['Actual'], colnames=['Predicted'], margins= True)
sn.heatmap(confusion_matrix, annot=True, fmt = 'd')
plt.show()

#Box Plot
x = ['Decision', 'Logistic', 'SVM']
y = [Dscore, Cscore, Sscore]
df = pd.DataFrame(dict(x=x, y=y))
sn.factorplot("x","y", data=df,kind="bar",size=6,aspect=2,legend_out=False)
plt.ylim(0.7,1)
plt.show()