#classifying the features using svc

import sklearn
import numpy as np
import scipy.io as sio
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score


train_anno=sio.loadmat('train-anno')
labels=train_anno['trait_annotation']
features=train_anno['face_landmark']
print(features.shape)
print(labels.shape)
features = minmax_scale(features,feature_range=(0, 1), axis=0)

threshold=np.mean(labels,axis=0)

#label thresholding
for i in range(14):
    labels[labels<threshold[i]]=-1
    labels[labels>threshold[i]]=1
labels.astype(int)
print(labels)

#dataset splitting
features_train, features_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, shuffle=False)


gamma=2**np.linspace(-17,5,12)
cost=2**np.linspace(-5,13,10)
best_params_train=[]
accuracy_train=[]
accuracy_test=[]
precision_train=[]
precision_test=[]
#training for each feature
for i in range(14):
    y=y_train[:,i]
    y=y.tolist()
    y_true=y_test[:,i]
    y_true=y_true.tolist()
    parameters = [{'kernel':['rbf'], 'C':cost, 'gamma':gamma}]
    clf=GridSearchCV(SVC(), parameters, cv=10, scoring= 'accuracy', return_train_score=True, n_jobs=-1)
    clf.fit(features_train,y)
    best_params_train.append(clf.best_params_)
    accuracy_train.append(clf.best_score_)
    y_pred=clf.predict(features_test)
    accuracy=accuracy_score(y_true,y_pred)
    accuracy_test.append(accuracy)
    prec=precision_score(y_true, y_pred)
    precision_test.append(prec)
    print(clf.best_params_)
    print(clf.best_score_)
    with open('p1_1_accuracy_train.pickle', 'wb') as handle:
        pickle.dump(accuracy_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('p1_1_accuracy_test.pickle', 'wb') as handle:
        pickle.dump(accuracy_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('p1_1_accuracy_parameters.pickle', 'wb') as handle:
        pickle.dump(best_params_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    



#visualisation
plt.plot(accuracy_test ,'ro-')
plt.plot(accuracy_train, 'bo-')
plt.xlabel('features')
plt.ylabel('accuracy')
plt.show()

plt.plot(precision_test ,'ro-')
plt.plot(precision_train, 'bo-')
plt.xlabel('features')
plt.ylabel('precision')
plt.show()



