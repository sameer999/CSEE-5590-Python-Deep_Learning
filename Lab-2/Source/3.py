#reading data from input file
f1=open('nlp_input.txt','r')
text = f1.read()
#print(text)

from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

#word tokenizing
wtokens = nltk.word_tokenize(text)
#print(wtokens)

#lemmatizing the each word token
lemmatizer = WordNetLemmatizer()
for x in wtokens:
    lemmatizer.lemmatize(x)

#generating trigrams
from nltk.util import ngrams
trigrams=ngrams(wtokens,3)

#printing the trigrams
for x in trigrams:
    print(x)

#extracting the top 10 trigrams based on count
import collections
trigram_frequency = collections.Counter(trigrams)
top_10 = trigram_frequency.most_common(10)
#print(top_10)

#extracting only trigrams by excluding the count attribute
l=[]
for i in top_10:
    l.append(i[0])
#print(l)

#sentence tokenizing
stokens = nltk.sent_tokenize(text)

'''for i in stokens:
    print(i)'''

#extracting the sentences from input file which contains the top10 most frequent trigrams
s=[]
for i in stokens:
    for j in l:
        if j[0] in i and j[1] in i and j[2] in i:
            s.append(i)

#concatenating the extracted sentences with top10 trigrams
s1=''

#set(s) is used to eliminate the duplicate sentences
for i in set(s):
    s1+=i
print(s1)


