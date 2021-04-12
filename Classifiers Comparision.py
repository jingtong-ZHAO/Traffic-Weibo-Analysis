import pandas as pd
import jieba
import re

import sklearn
from sklearn import *
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

random.seed(100)

# load data
file = 'data_for_label_test.xlsx'
data = pd.read_excel(file, sheet_name=0, usecols=[3, 4])
text = pd.read_excel(file, sheet_name=0, usecols=[3])
label = pd.read_excel(file, sheet_name=0, usecols=[4])
text = text.values.tolist()
label = label.values.tolist()

# import Chinese stopwords list
def stopwords_list():
    stop_words = [line.strip() for line in open('stopwords.txt', encoding='UTF-8').readlines()]
    return stop_words

# pre-processing text data
def preprocessingX(raw_text):
    stopwords = stopwords_list()
    # 1. Remove the urls
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)" \
            r"))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text1 = re.sub(regex, '', raw_text)

    # 2. Remove @ patterns
    text2 = re.sub('@[\u4e00-\u9fa5a-zA-Z0-9_-]{4,30}', '', text1)
    text_without_username = re.sub('\/\/@.*?:', '', text2)

    # 3. Tokenize the weibos using jieba and remove stopwords
    text3 = jieba.lcut(text_without_username.strip())
    result_split = [word for word in text3 if word not in stopwords]  # result is a list

    # 4. remove blank elements and digits
    text4 = [word for word in result_split if word != ' ']
    result_cleaned = list(filter(lambda x: not str(x).isdigit(), text4))

    # 5. transform list to string for further analysis
    result = ' '.join(str(word) for word in result_cleaned)
    return result

# pre-processing label data
def preprocessingY(raw_label):
    labels = []
    for k in range(len(raw_label)):
        labels.append(' '.join(str(num) for num in raw_label[k]))
        labels[k] = int(labels[k])
    labels = np.array(labels)   # labels is an array now
    return labels

# tf-idf vectorizer to count word weight
def vectorize(corpus):
    tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.6).fit(corpus)
    vectors = tfidf_model.transform(corpus)
    return vectors


# SVM classifier model
def SVM_poly_classifier(trainingX, trainingY, testingX, testingY):
    degree = [2]
    print("SVM-poly Classifier: ")
    for i in range(len(degree)):
        clf_SVM = SVC(kernel='poly', C=5, degree=degree[i], gamma='scale')
        clf_SVM.fit(trainingX, trainingY)
        predYtest = clf_SVM.predict(testingX)
        f1 = f1_score(testingY, predYtest, average='macro')
        print("For degree =", degree[i], ",F1 score on testing set is:", f1)
        print(metrics.precision_score(testingY, predYtest, average='macro'))
        print(metrics.recall_score(testingY, predYtest, average='macro'))

def SVM_linear_classifier(trainingX, trainingY, testingX, testingY):
    Cs = [1]
    print("SVM-linear Classifier: ")
    for i in range(len(Cs)):
        clf_SVM = SVC(kernel='linear', C=Cs[i],  gamma='scale')
        clf_SVM.fit(trainingX, trainingY)
        predYtest = clf_SVM.predict(testingX)
        f1 = f1_score(testingY, predYtest, average='macro')
        print("For C =", Cs[i], ",F1 score on testing set is:", f1)
        print(metrics.precision_score(testingY, predYtest, average='macro'))
        print(metrics.recall_score(testingY, predYtest, average='macro'))

def SVM_rbf_classifier(trainingX, trainingY, testingX, testingY):

    clf_SVM = SVC(C=5, gamma=0.2)
    clf_SVM.fit(trainingX, trainingY)
    predYtest = clf_SVM.predict(testingX)
    f1 = (f1_score(testingY, predYtest, average='macro'))
    return f1


def MultinomialNB(trainingX, trainingY, testingX, testingY):
    alphas = [0.4]
    print("Multinomial Naive Bayes Classifier: ")
    for i in range(len(alphas)):
        clf = sklearn.naive_bayes.MultinomialNB(alpha = alphas[i], fit_prior= True, class_prior= None)
        clf.fit(trainingX, trainingY)
        predYtest = clf.predict(testingX)
        f1 = f1_score(testingY, predYtest, average='macro')
        print("For alpha =", alphas[i], ",F1 score on testing set is:", f1)
        print(metrics.precision_score(testingY, predYtest, average='macro'))
        print(metrics.recall_score(testingY, predYtest, average='macro'))

def LR(trainingX, trainingY, testingX, testingY):
    Cs = [10, 100]
    print("Logistic Regression Classifier: ")
    for i in range(len(Cs)):
        clf = linear_model.LogisticRegression(C=Cs[i])
        clf.fit(trainingX, trainingY)
        predYtest = clf.predict(testingX)
        f1 = f1_score(testingY, predYtest, average='macro')
        print("For C =", Cs[i], ",F1 score on testing set is:", f1)
        print(metrics.precision_score(testingY, predYtest, average='macro'))
        print(metrics.recall_score(testingY, predYtest, average='macro'))

def RandomForest(trainingX, trainingY, testingX, testingY):
    Cs = [500, 1000]
    print("Random Forest Classifier: ")
    for i in range(len(Cs)):
        clf = RandomForestClassifier(n_estimators=Cs[i])
        clf.fit(trainingX, trainingY)
        predYtest = clf.predict(testingX)
        f1 = f1_score(testingY, predYtest, average='macro')
        print("For n_estimators =", Cs[i], ",F1 score on testing set is:", f1)
        print(metrics.precision_score(testingY, predYtest, average='macro'))
        print(metrics.recall_score(testingY, predYtest, average='macro'))

def dimension_reduction(vectors):
    SVD = TruncatedSVD(n_components=2)
    return SVD.fit_transform(vectors)

def clustering(X, Y):
    X = dimension_reduction(X)
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1, verbose=False)
    kmeans.fit(X)
    predY = kmeans.predict(X)
    plt.subplot(1, 2, 1)
    plt.title("Clustering")
    plt.scatter(X[:, 0], X[:, 1], c=predY, edgecolor=None)
    plt.subplot(1, 2, 2)
    plt.title("Actual distribution")
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolor=None)
    plt.show()


# main function
if __name__ == "__main__":
    results = []
    print("After pre-processing the text set: ")
    for i in range(len(text)):
        results.append(preprocessingX(raw_text=str(text[i])))
    X = vectorize(corpus=results)
    Y = preprocessingY(label)
    print(X.shape)

    components = [20, 50, 100, 150, 200, 300, 400, 500, 600, 800, 1100, 1200]
    f1=[]
    for i in range(len(components)):
        kpca = decomposition.KernelPCA(kernel='cosine', n_components=components[i])
        Xn = kpca.fit_transform(X)
        trainX, testX, trainY, testY = \
            model_selection.train_test_split(Xn, Y,
                                             train_size=0.80, test_size=0.20, random_state=4487)
        f1.append(SVM_rbf_classifier(trainX, trainY, testX, testY))
        print(f1[i])
    plt.xlabel('components')
    plt.ylabel('f1 score')
    plt.plot(components, f1)

    plt.show()
    print(max(f1))
    print(f1.index(max(f1)))





