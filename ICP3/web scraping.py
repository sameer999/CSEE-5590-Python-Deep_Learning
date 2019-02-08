import requests
website_url=requests.get('https://en.wikipedia.org/wiki/Deep_learning').text
from bs4 import BeautifulSoup
soup=BeautifulSoup(website_url,'html.parser')
#print(soup.prettify())
print("Title of web page:",soup.title)
for link in soup.find_all('a'):
    print(link.get('href'))