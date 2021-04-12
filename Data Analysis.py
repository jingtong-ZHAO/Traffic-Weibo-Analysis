import pandas as pd
import jieba
import re
from sklearn import *
from numpy import *
import numpy as np
from xlwt import *
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

random.seed(100)
import datetime
start = datetime.datetime.now()
# load training data
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

def preprocessingXtest(raw_text):
    traffic_related = 0
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
    # Only select traffic-related data
    for word in result_split:
        if word in traffic_word_set:
            traffic_related = 1


    # 4. remove blank elements and digits
    text4 = [word for word in result_split if word != ' ']
    result_cleaned = list(filter(lambda x: not str(x).isdigit(), text4))

    # 5. transform list to string for further analysis
    result = ' '.join(str(word) for word in result_cleaned)

    if traffic_related == 1:
        outputX.append(raw_text)
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
def SVM_classifier(trainingX, trainingY, testingX, testingY):
    clf_SVM = SVC(C=5, gamma=0.2)
    clf_SVM.fit(trainingX, trainingY)
    predYtest = clf_SVM.predict(testingX)
    f1 = f1_score(testingY, predYtest, average='macro')
    print("The F1 score on testing set is: ", f1)
    return clf_SVM

def dimension_reduction(vectors):
    svd = TruncatedSVD(n_components=400)
    return svd.fit_transform(vectors)

def clusteringKmeans(X, Y):
    X = dimension_reduction(X)
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1, verbose=False)
    kmeans.fit(X)
    predY = kmeans.predict(X)
    plt.title("Clustering")
    plt.scatter(X[:, 0], X[:, 1], c=predY, edgecolor=None)
    plt.show()
    scatterLegend(X, predY, 0, 1)

def scatterLegend(data, labels, x, y):
    type1 = []
    type2 = []
    type3 = []
    for i in range(len(labels)):
        if labels[i] == 1:
            type1.append(np.array(data[i]))
        elif labels[i] == 2:
            type2.append(np.array(data[i]))
        else:
            type3.append(np.array(data[i]))
    type1 = np.array(type1)
    type2 = np.array(type2)
    type3 = np.array(type3)
    print(type3.shape)
    g1 = plt.scatter(type1[:, x], type1[:, y], c='red')
    g2 = plt.scatter(type3[:, x], type3[:, y], c='yellow')
    g3 = plt.scatter(type2[:, x], type2[:, y], c='blue')
    plt.legend(handles=[g1, g2, g3], labels=['label 0', 'label 1', 'label 2'])
    plt.show()




# main function
if __name__ == "__main__":
    traffic_word_set = {'堵', '拥堵', '阻塞', '塞车', '拥挤', '车祸', '剐蹭', '事故', '撞', '追尾', '相撞', '路况', '路段',
                        '路线', '封道', '封路', '绕行', '畅通', '立交', '高架', '快速路', '大桥', '隧道', '驾驶', '避让', '车距'}

    # Real Weibo data

    df1 = pd.read_csv(r"E:\aY4 semA\FYP\Data\shanghai_2012_jun_aug\weibo0914_1_2012_jun_aug.csv", usecols=[4])
    df1 = df1.values.tolist()
    df2 = pd.read_csv(r"E:\aY4 semA\FYP\Data\shanghai_2012_jun_aug\weibo0914_1_2012_jun_aug.csv", usecols=[9])
    df2 = df2.values.tolist()
    df = np.hstack((df1, df2))


    results = []
    for i in range(len(text)):
        results.append(preprocessingX(raw_text=str(text[i])))
    X = vectorize(corpus=results)
    X = dimension_reduction(X)
    Y = preprocessingY(label)
    # clusteringKmeans(X, Y)
    trainX, testX, trainY, testY = \
        model_selection.train_test_split(X, Y,
            train_size=0.80, test_size=0.20, random_state=4487)
    svm = SVM_classifier(trainX, trainY, testX, testY)

    # Classify real Weibo data
    real_weibo = []
    outputX = []
    print("After pre-processing the text set: ")
    for i in range(len(df)):
        real_weibo.append(preprocessingXtest(raw_text=str(df[i])))
    while None in real_weibo:
        real_weibo.remove(None)
    testX_real = vectorize(corpus=real_weibo)
    testX_real = dimension_reduction(testX_real)
    predY_real = svm.predict(testX_real)


    # Save prediction results to csv
    res = pd.DataFrame({'Text': outputX, 'Pred_Label': predY_real})
    res.to_excel(r'E:\aY4 semA\FYP\Data\shanghai_2012_jun_aug\Prediction\0914_1_2012_jun_aug.xlsx')



    end = datetime.datetime.now()
    print("Runnning time: %s Seconds" % (end - start))



#文件路径
file_dir = r'E:\aY4 semA\FYP\Data\shanghai_2012_jun_aug\Prediction'
#构建新的表格名称
new_filename = file_dir + '\\combined.xls'
#找到文件路径下的所有表格名称，返回列表
file_list = os.listdir(file_dir)
new_list = []

for file in file_list:
    #每个excel的路径
    file_path = os.path.join(file_dir,file)
    #将excel转换成DataFrame
    dataframe = pd.read_excel(file_path)
    #保存到新列表中
    new_list.append(dataframe)

#多个DataFrame合并为一个
df = pd.concat(new_list)
#写入到一个新excel表中
df.to_excel(new_filename,index=False)