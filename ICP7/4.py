from nltk.stem import PorterStemmer

import nltk

nltk.download('averaged_perceptron_tagger')
f = open('input.txt', 'r',encoding='utf-8')
input = f.read()

wtokens = nltk.word_tokenize(input)


ps = PorterStemmer()

for x in wtokens:
    print(ps.stem(x))