import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time

#%% DATA PROCESSING
start = time.time()
#Loading data from GitHub 
url='https://raw.githubusercontent.com/pradhanhitesh/Computational-Neuroscience-Class-Project/main/oasis_longitudinal.csv'
load_data=pd.read_csv(url, encoding='unicode_escape')

#Dropping unwanted columns
load_data=load_data.drop(['Hand','MRI ID','Visit','Subject ID'],axis=1)

#Data Evaluation - Correlational Map
cor = load_data.corr()
plt.figure(figsize=(9,9))
sns.heatmap(cor, xticklabels=cor.columns.values,yticklabels=cor.columns.values, annot=True)

#Sex vs CDR
sns.set(rc={'figure.figsize':(5,5)})
sns.barplot(load_data['M/F'],load_data['CDR'])

#Age density over Group
facet= sns.FacetGrid(load_data,hue="Group", aspect=3)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, load_data['Age'].max()))
facet.add_legend()
plt.xlim(50,100)


#Enconding classes into binary
load_data['M/F']=load_data['M/F'].replace(['F','M'],[0,1])                      #Female=0; Male=1
load_data['Group']=load_data['Group'].replace(['Converted'],['Demented'])       
load_data['Group']=load_data['Group'].replace(['Demented','Nondemented'],[1,0]) #Demented=1; Nondemented=0

#Filling the missing data with median
load_data['SES'].fillna(load_data['SES'].median(),inplace=True)
load_data['MMSE'].fillna(load_data['MMSE'].median(),inplace=True)

#Separating target and features
target=load_data['Group']
features=load_data.drop(columns=['Group'],axis=1)
features_list=list(features.columns)

#Splitting the target and features
train_features,test_features,train_labels,test_labels = train_test_split(features,target,stratify=target, random_state=43)
#print('The shape of feature training set is:',train_features.shape)
#print('The shape of labels training set is:',train_labels.shape)
#print('The shape of feature testing set is:',test_features.shape)
#print('The shape of labels testing set is:',test_labels.shape)

#Randomforest is not sensitive to scaling and hence the features were not 
#normalised using MinMaxScaller
#%%Making the model
model=RandomForestClassifier()
param_rf={
    "n_estimators":range(2,15,2),
    "max_features":range(1,9),
    "max_depth":range(1,10)
    }

#Using RandomizedSearchCV
random_rf=RandomizedSearchCV(estimator=model, param_distributions=param_rf, n_iter = 50 , scoring='accuracy', cv = 5, verbose=2, n_jobs=-1,random_state=42)
random_rf.fit(train_features,train_labels)
print('The best parameteres for the model are:',random_rf.best_params_)
print("----------------------\n")
model=random_rf.best_estimator_
crossval = cross_val_score(model,train_features,train_labels,cv=5,scoring='accuracy')
scores = np.mean(crossval)
scores=round(scores*100,2)
print('Train acc:',scores)
print("----------------------\n")
#model.fit(train_features,train_labels)
test_pred=model.predict(test_features)
print('Test acc:',accuracy_score(test_labels,test_pred)*100)

end = time.time()
total_time = end - start

#%%Model Evaluation - Confusion Matrix
test_pred=model.predict(test_features)
cm = confusion_matrix(test_labels, test_pred)
clr = classification_report(test_labels, test_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, vmin=0, fmt='g', cbar=False, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
print("Classification Report:\n----------------------\n", clr)
print("Time:"+ str(total_time))
#%%Model Evaluation - Learning Curve

training_size,training_scores,testing_scores=learning_curve(model,features,target,cv=5,shuffle=True,scoring='accuracy',train_sizes=np.linspace(0.01,1,50))
train_mean=np.mean(training_scores, axis=1)
train_std=np.std(training_scores,axis=1)
test_mean=np.mean(testing_scores,axis=1)
test_std=np.std(testing_scores,axis=1)

plt.figure(figsize=(10,6))
plt.plot(training_size,train_mean,'--',label='Training Score')
plt.plot(training_size,test_mean,label='Cross-Validation Score')
plt.fill_between(training_size,train_mean-train_std,train_mean+train_std,color='#DDDDDD')
plt.fill_between(training_size,test_mean-test_std,test_mean+test_std,color='#DDDDDD')
plt.title('Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

#%%Important Features
print('Feature importance for,',model,' as follows:')
importances = list(model.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features_list, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];