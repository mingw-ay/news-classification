import os
import jieba.posseg as pseg

# 一、读取得到训练集，测试集，验证机文本，并得到其分类标签列表
base_dir = 'raw_data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
stopwords_dir = os.path.join(base_dir, 'cn_stopwords.txt')

# 1.得到训练，测试，验证文本list
train_texts = open(train_dir, encoding='utf-8').read().split('\n')
test_texts = open(test_dir, encoding='utf-8').read().split('\n')
val_texts = open(val_dir, encoding='utf-8').read().split('\n')

# 2.得到对应的类别list
train_label = []
test_label = []
val_label = []


def getLabelList(newsList, labelList):
    for i in range(len(newsList)):
        labelList.append(newsList[i].split('\t')[0])
        newsList[i] = newsList[i].split('\t')[1]


getLabelList(train_texts, train_label)
getLabelList(test_texts, test_label)
getLabelList(val_texts, val_label)

# 二、对数据进行分词以及去除其中的停用词

# 1.得到停用词列表stopwords,是他人整理的停用词
stopwords = set()

with open(stopwords_dir, encoding='utf-8') as f:
    for line in f:
        stopwords.add(line.strip())


# 2.对应的去除停用词方法
def remove_stopword(words):
    return [word for word in words if word not in stopwords]


# 3.将每条新闻的分词后数组转化为空格分隔的字符串
def join(text_list):
    return " ".join(text_list)


# 4.对应的分词方法（去除每个flag为x,即标点符号）,并且只截取前100个汉字
def cut2words(newsList):
    for i in range(len(newsList)):
        words = pseg.cut(newsList[i])
        newsList[i] = []
        wordsCount = 0
        for w in words:
            if w.flag != 'x' and wordsCount < 100:
                wordsCount += len(w.word)
                newsList[i].append(w.word)
        # 调用去除停用词方法
        newsList[i] = remove_stopword(newsList[i])
        # 调用转为字符串方法
        newsList[i] = join(newsList[i])
        print(i)


# 5.将类别标签离散化
def map2digits(x):
    if x == '体育':
        return '0'
    elif x == '财经':
        return '1'
    elif x == '房产':
        return '2'
    elif x == '家居':
        return '3'
    elif x == '教育':
        return '4'
    elif x == '科技':
        return '5'
    elif x == '时尚':
        return '6'
    elif x == '时政':
        return '7'
    elif x == '游戏':
        return '8'
    else:
        return '9'


train_label = list(map(map2digits, train_label))
test_label = list(map(map2digits, test_label))
val_label = list(map(map2digits, val_label))


# 6. 调用分词方法，对三个数据集进行分词操作
# cut2words(test_texts)
# f1 = open('test_contents.txt', 'w', encoding='utf-8')
# f1.write('\n'.join(test_texts))
# f1.close()
# print('test_text done!')
# f2 = open('test_label.txt', 'w')
# f2.write('\n'.join(test_label))F
# f2.close()
cut2words(val_texts)
f3 = open('val_contents.txt', 'w', encoding='utf-8')
f3.write('\n'.join(val_texts))
f3.close()
# f4 = open('val_label.txt', 'w')
# f4.write('\n'.join(val_label))
# f4.close()
cut2words(train_texts)
f5 = open('train_contents.txt', 'w', encoding='utf-8')
f5.write('\n'.join(train_texts))
f5.close()
# f6 = open('train_label.txt', 'w', encoding='utf-8')
# f6.write('\n'.join(train_label))
# f6.close()
