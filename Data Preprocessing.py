import pandas as pd
import jieba
import re
from sklearn import *
from numpy import *
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
random.seed(100)
from sklearn.svm import SVC

# load data
file = 'data_for_label_final_5000.xlsx'
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

# tf-idf vectorizer to count word frequency
def vectorize(corpus):
    vectorizer = CountVectorizer(min_df=1, max_df=1.0, token_pattern='\\b\\w+\\b')
    vectorizer.fit(corpus)
    bag_of_words = vectorizer.get_feature_names()
    print(bag_of_words)
    vectors = vectorizer.transform(corpus)
    return vectors

# Naive Bayes classifier model
def classifier_model(trainingX, trainingY, testingX, testingY):
    bmodel = naive_bayes.BernoulliNB(alpha=0.1)
    bmodel.fit(trainingX, trainingY)
    predY = bmodel.predict(testingX)
    accuracy = metrics.accuracy_score(testingY, predY)
    print("Accuracy rate:", accuracy)

def SVM_classifier(trainingX, trainingY, testingX, testingY):
    clf_SVM = SVC(C=5, gamma=0.2)
    clf_SVM.fit(trainingX, trainingY)
    predYtest = clf_SVM.predict(testingX)
    return clf_SVM

# main function
if __name__ == "__main__":
    results = []
    print("After pre-processing the text set: ")
    for i in range(len(text)):
        results.append(preprocessingX(raw_text=str(text[i])))
    X = vectorize(corpus=results)
    Y = preprocessingY(label)
    trainX, testX, trainY, testY = \
        model_selection.train_test_split(X, Y,
            train_size=0.80, test_size=0.20, random_state=4487)
    SVM_classifier(trainX, trainY, testX, testY)






