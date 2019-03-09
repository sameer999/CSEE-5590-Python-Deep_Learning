import nltk
nltk.download('punkt')
f = open('input.txt', 'r',encoding='utf-8')
input = f.read()
#print(f)
#input = "I am not good. I am very happy."
stokens = nltk.sent_tokenize(input)
wtokens = nltk.word_tokenize(input)

for s in stokens:
    print(s)

print('\n')

for t in wtokens:
    print(t)