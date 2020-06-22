import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
#from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import GridSearchCV
from scipy import interp
import os

x_train=np.loadtxt('D:/SJTU Lessons/X_train.txt',delimiter=' ')
x_test=np.loadtxt('D:/SJTU Lessons/X_test.txt',delimiter=' ')
# 将标签二值化
Y_train=np.loadtxt('D:/SJTU Lessons/Y_train.txt',delimiter=' ')
Y_train = label_binarize(Y_train, classes=[0, 1, 2, 3])
Y_test=np.loadtxt('D:/SJTU Lessons/Y_test.txt',delimiter=' ')
Y_test = label_binarize(Y_test, classes=[0, 1, 2, 3])
# 设置种类
n_classes=4

# 训练模型并预测
#seed
random_state = np.random.RandomState(0)
#dim
n_samples=3532
n_features = 641

# Learn to predict each class against the other
#'linear’, ‘poly’, ‘rbf'
parameters = {'estimator__kernel':('linear', 'rbf','poly'), 'estimator__C':[0.1, 1, 10],
              'estimator__degree':[2,3,4], 'estimator__gamma':[0.001,0.01,0.1]}
svr = OneVsRestClassifier(svm.SVC(probability=True,random_state=random_state))
clf = GridSearchCV(svr, parameters)
clf.fit(x_train, Y_train)

print("The best parameters are %s with a score of %0.2f"
      % (clf.best_params_, clf.best_score_))



#classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, random_state=random_state))
#y_score = classifier.fit(x_train, Y_train).decision_function(x_test)




