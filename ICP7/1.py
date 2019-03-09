from bs4 import BeautifulSoup
import requests

page_link = 'https://en.wikipedia.org/wiki/Google'
page_response = requests.get(page_link)

page_content = BeautifulSoup(page_response.content, "html.parser")
#data = page_content.find_all(text=True)
text = page_content.get_text()
print(text)

f = open('input.txt', 'w',encoding='utf-8')
f.write(text)

print(type(text))
