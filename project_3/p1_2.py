#classification with hog features using svr


import sklearn
import numpy as np
import scipy.io as sio
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import cv2
import os


train_anno=sio.loadmat('train-anno')
labels=train_anno['trait_annotation']
features=train_anno['face_landmark']
print(features.shape)
print(labels.shape)
print(labels)


#reading the politician images and sending to hog
img_dir='img'
i=0
final_features_concat=[]
for file_name in sorted(os.listdir(img_dir)):
    img = cv2.imread(os.path.join(img_dir, file_name))
    print(file_name)  
    hog_features, hog_image = hog(img, orientations=32, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
    features_concat=np.concatenate((features[i,:], hog_features), axis=0)
    final_features_concat.append(features_concat)
    print(final_features_concat.shape)
    i+=1
final_features_concat=np.array(final_features_concat)

#scaling and splitting data
final_features_concat= minmax_scale(final_features_concat,feature_range=(0, 1), axis=0)   
features_train, features_test, y_train, y_test = train_test_split(final_features_concat, labels, test_size=0.20,shuffle=False)

#training 14 svms using SVR
gamma=np.linspace(2**-17,2**5,10)
cost=np.linspace(2**-5,2**13,10)
epsilon=np.linspace(2**-9,2**5,10)
threshold=np.mean(y_train, axis=0)
print(threshold)
best_params_train=[]
accuracy_train=[] 
accuracy_test=[]
precision_train=[]
precision_test=[]
neg_mse=[]
for i in range(14):
    y=y_train[:,i]
    y=y.tolist()
    y_true=y_test[:,i]
    y_true=y_true.tolist()
    parameters = [{'kernel':['rbf'], 'C':cost, 'gamma':gamma, 'epsilon':epsilon}]
    clf=GridSearchCV(SVR(), parameters, cv=5, scoring= 'neg_mean_squared_error', return_train_score=True, n_jobs=-1,verbose=True)
    clf.fit(features_train,y)
    #calculating training accuracy and precision
    train_y_pred=clf.predict(features_train)
    train_y_pred=(train_y_pred>threshold[i])*2-1
      
    #threshold y
    y=(y>threshold[i])*2-1
    train_accuracy=accuracy_score(y,train_y_pred)
    accuracy_train.append(train_accuracy)
    train_precision=precision_score(y,train_y_pred)
    precision_train.append(train_precision)

    #calculating testing accuracy and precision   
    test_y_pred=clf.predict(features_test)
    print((test_y_pred>threshold[i]))
    test_y_pred=(test_y_pred>threshold[i])*2-1
    print(test_y_pred)
   
    #threshold y-test
    y_true=(y_true>threshold[i])*2-1
    test_accuracy=accuracy_score(y_true,test_y_pred)
    accuracy_test.append(test_accuracy)
    print(test_accuracy)
    test_precision=precision_score(y_true,test_y_pred)
    precision_test.append(test_precision)



    best_params_train.append(clf.best_params_)
    neg_mse.append(clf.best_score_)  #neg mse
    
    print(clf.best_params_)
    print(clf.best_score_)
    
    with open('p1_2_accuracy_train.pickle', 'wb') as handle:
        pickle.dump(accuracy_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('p1_2_accuracy_test.pickle', 'wb') as handle:
        pickle.dump(accuracy_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('p1_2_parameters.pickle', 'wb') as handle:
        pickle.dump(best_params_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('p1_2_clf_model.pickle', 'wb') as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#visualisation  
plt.figure(1) 
plt.plot(accuracy_test,'ro-')
plt.plot(accuracy_train,'bo-')
plt.xlabel('features')
plt.ylabel('accuracy')
plt.show()

plt.figure(2) 
plt.plot(precision_test,'ro-')
plt.plot(precision_train,'bo-')
plt.xlabel('features')
plt.ylabel('precision')
plt.show()