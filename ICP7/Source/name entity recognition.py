from nltk import wordpunct_tokenize,pos_tag,ne_chunk
import nltk

nltk.download('maxent_ne_chunker')
nltk.download('words')

f = open('input.txt', 'r',encoding='utf-8')
input = f.read()

stokens = nltk.sent_tokenize(input)

for i in stokens:
    print(ne_chunk(pos_tag(wordpunct_tokenize(i))))