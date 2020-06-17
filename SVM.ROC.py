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
classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(x_train, Y_train).decision_function(x_test)

# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
