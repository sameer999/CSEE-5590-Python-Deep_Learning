import nltk
nltk.download('averaged_perceptron_tagger')
f = open('input.txt', 'r',encoding='utf-8')
input = f.read()
#print(f)
#input = "I am not good. I am very happy."

wtokens = nltk.word_tokenize(input)

print(nltk.pos_tag(wtokens))