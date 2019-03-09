from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')

f = open('input.txt', 'r',encoding='utf-8')
input = f.read()

wtokens = nltk.word_tokenize(input)

lemmatizer = WordNetLemmatizer()

for x in wtokens:
    print(lemmatizer.lemmatize(x))

