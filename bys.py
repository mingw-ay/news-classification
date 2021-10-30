from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_classif


# 一、读取得到训练集，测试集，验证机文本，并得到其分类标签列表

# 1.进行文件读取
base_dir = 'dataset'
X_train_dir = os.path.join(base_dir, 'train_contents.txt')
X_test_dir = os.path.join(base_dir, 'test_contents.txt')
X_val_dir = os.path.join(base_dir, 'val_contents.txt')
y_train_dir = os.path.join(base_dir, 'train_label.txt')
y_test_dir = os.path.join(base_dir, 'test_label.txt')
y_val_dir = os.path.join(base_dir, 'val_label.txt')

X_train = open(X_train_dir, encoding='utf-8').read().split('\n')
X_test = open(X_test_dir, encoding='utf-8').read().split('\n')
X_val = open(X_val_dir, encoding='utf-8').read().split('\n')
y_train = np.array(open(y_train_dir).read().split('\n')).astype(np.int32)
y_test = np.array(open(y_test_dir).read().split('\n')).astype(np.int32)
y_val = np.array(open(y_val_dir).read().split('\n')).astype(np.int32)


print('训练集样本数：', len(X_train), '测试机样本数：', len(X_test), '验证集样本数：', len(X_val))


# 二、使用TfidVectorizer()进行向量化

vec = TfidfVectorizer(ngram_range=(1, 2))
# ngram_range=(1,2)先将新闻切分为一维/二维元组再进行向量化
# 减少语句本来不同但切分出的单词相同带来的误差
X_train_tran = vec.fit_transform(X_train)
print('X_train vectorized')
X_test_tran = vec.transform(X_test)
print('X_test vectorized')
X_val_tran = vec.transform(X_val)
print('X_val vectorized')
print(X_train_tran.shape, X_test_tran.shape, X_val_tran.shape)


# 词袋模型向量化后会产生过多特征，用方差分析进行特征选择
# 选择出与目标相对变量最相关的20000个特征
# print(f_classif(X_train_tran, y_train))


# 将数组转化为float32的numpy数组
X_train_tran = X_train_tran.astype(np.float32)
X_test_tran = X_test_tran.astype(np.float32)
X_val_tran = X_val_tran.astype(np.float32)

selector = SelectKBest(f_classif, k=min(25000, X_train_tran.shape[1]))
selector.fit(X_train_tran, y_train)
X_train_tran = selector.transform(X_train_tran)
X_test_tran = selector.transform(X_test_tran)
X_val_tran = selector.transform(X_val_tran)
print(X_train_tran.shape, X_test_tran.shape, X_val_tran.shape)


# 三、使用朴素贝叶斯进行预测分析

# 朴素贝叶斯
clf = MultinomialNB(alpha=0.01)
clf.fit(X_train_tran, y_train)
y_hat = clf.predict(X_test_tran).tolist()
print(clf.predict(X_val_tran).tolist())
# print(classification_report(y_test, y_hat))
