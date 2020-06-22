import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
#from sklearn.multiclass import OneVsOneClassifier
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
classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True, C=0.1, degree=2,
                                         gamma=0.001, random_state=random_state))
y_score = classifier.fit(x_train, Y_train).decision_function(x_test)

np.savetxt('svmpred-g.001.txt',y_score)
