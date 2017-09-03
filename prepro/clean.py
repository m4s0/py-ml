import re

from nltk.corpus import stopwords


def clean(corpus):
    stopList = stopwords.words('italian')
    # tagger = treetaggerwrapper.TreeTagger(TAGLANG='it')

    # removes apostrophes and splits into words
    filteredCorpus = [phrase.replace('l\'', 'la ').replace('un\'', 'una ').replace('m\'', 'mi ').replace('t\'', 'ti ').replace('c\'', 'ci ').replace('v\'', 'vi ').replace('s\'', 'si ').lower().split() for phrase in corpus]

    # remove punctuation
    filteredCorpus = [[re.sub("[^\w\d'\s]+", ' ', word) for word in phrase] for phrase in filteredCorpus]

    # lemmatize words
    # filteredCorpus = [tagger.make_tags(unicode(phrase,"utf-8")) for phrase in filteredCorpus]

    # remove stopwords and join the words back into one string separated by space
    filteredCorpus = [" ".join([word for word in phrase if word not in stopList and not word.isdigit()]) for phrase in filteredCorpus]

    return filteredCorpus
