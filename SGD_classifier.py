import csv
import re

import nltk
import numpy as np
import pyprind
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

from prepro.load import json_to_csv, merge_and_shuffle


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]

    return tokenized


def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


def stream_csv(file):
    csv_reader = csv.reader(open(file), delimiter=',')
    # skip header
    next(csv_reader, None)
    for row in csv_reader:
        text, label = row[0], int(row[1])
        yield text, label


def size_csv(file):
    csv_reader = csv.reader(open(file), delimiter=',')
    data = list(csv_reader)
    return len(data)


def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


json_to_csv(input='./_approved',
            output='./data/approved.csv',
            label=1)
json_to_csv(input='./_not_approved',
            output='./data/not_approved.csv',
            label=0)

merge_and_shuffle(input=['./data/approved.csv', './data/not_approved.csv'],
                  output='./data/comments.csv')

doc_stream = stream_csv(file='./data/comments.csv')
size = size_csv(file='./data/comments.csv')

nltk.download('stopwords')
stop = stopwords.words('italian')

hashing_vectorizer = HashingVectorizer(decode_error='ignore',
                                       n_features=2 ** 21,
                                       preprocessor=None,
                                       tokenizer=tokenizer)

SGD_classifier = SGDClassifier(loss='log', random_state=1, n_iter=1)

to_test = 5000
batch_size = 1000
iterations = int((size - to_test) / batch_size)

print('Size: %s' % size)
print('Size to test: %s' % to_test)

p_bar = pyprind.ProgBar(iterations)

classes = np.array([0, 1])
for _ in range(iterations):
    X_train, y_train = get_minibatch(doc_stream, size=batch_size)
    if not X_train:
        break
    X_train = hashing_vectorizer.transform(X_train)
    SGD_classifier.partial_fit(X_train, y_train, classes=classes)
    p_bar.update()

X_test, y_test = get_minibatch(doc_stream, size=to_test)
X_test = hashing_vectorizer.transform(X_test)
print('Accuracy: %.3f' % SGD_classifier.score(X_test, y_test))
