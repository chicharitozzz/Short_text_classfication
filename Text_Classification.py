# @Time : 2018/7/27 16:15
# @Author : Chicharito_Ron
# @File : Text_Classification.py
# @Software: PyCharm Community Edition

import numpy as np
import json
import redis
import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA


def read_data():
    """读取数据并分词"""
    X, Y = [], []
    with open('./static/dataset.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            label, con = line.strip().split('\t')
            X.append(' '.join(jieba.cut(con, cut_all=False)))
            Y.append(label)

    return X, Y


def get_stop_words():
    """停用词列表"""
    stop_words = []
    with open('./static/停用词.txt') as f:
        lines = f.readlines()

    for line in lines:
        stop_words.append(line.strip())

    return stop_words


class Textclassify:
    def __init__(self, X, Y, stop_words):
        self.X = X
        self.Y = Y
        self.s_words = stop_words

    def cross_validation(self, X_vec):
        """交叉验证，90%数据用于训练，10%数据用于测试"""
        data = list(zip(X_vec, self.Y))
        np.random.shuffle(data)  # 数据乱序

        train_data = data[:4000]
        test_data = data[4000:]

        X_train, Y_train = zip(*train_data)
        X_test, Y_test = zip(*test_data)

        return X_train, Y_train, X_test, Y_test

    def proprocessing(self):
        """数据预处理：向量化；计算TF-IDF权重"""
        vectorizer = CountVectorizer(stop_words=self.s_words, min_df=1e-3)
        vec_train = vectorizer.fit_transform(self.X)
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vec_train)
        tfidf = tfidf.toarray()
        return tfidf

    @staticmethod
    def lower_dimension(tfidf):
        """PCA降维"""
        pca = PCA(n_components=1000, copy=False)
        X_lowd = pca.fit_transform(tfidf)

        return X_lowd

    @staticmethod
    def classifiers():
        """分类器"""
        nb = MultinomialNB(fit_prior=False)  # 朴素贝叶斯
        lr = LogisticRegression(solver='liblinear') # 逻辑回归
        dt = DecisionTreeClassifier(criterion='gini') # 决策树
        rfc = RandomForestClassifier(n_estimators=200, random_state=1000) # 随机森林
        svm = SVC(kernel='linear', tol=1e-2, max_iter=100) # SVM
        return [nb, lr, dt, rfc, svm]

    def classify(self):
        """分类"""
        clfs = self.classifiers()
        length = len(clfs)
        acc_li = [[] for _ in range(length)]

        tfidf = self.proprocessing()
        # lowd_X = self.lower_dimension(tfidf)

        for i in range(20):
            X_train, Y_train, X_test, Y_test = self.cross_validation(tfidf)
            for i in range(length):
                clfs[i].fit(X_train, Y_train)
                acc = clfs[i].score(X_test, Y_test)
                print(acc)
                acc_li[i].append(acc)

        with open('./static/res.json', 'w', encoding='utf-8') as f:
            json.dump(acc_li, f, indent=2)

        return acc_li


if __name__ == '__main__':
    X, Y = read_data()
    stop_words = get_stop_words()
    tc = Textclassify(X, Y, stop_words)
    acc_li = tc.classify()
    print(acc_li)
