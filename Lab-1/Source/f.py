import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States"
page = requests.get(url)                  #requesting given url and creating a response object called 'page'
soup=BeautifulSoup(page.content,'html.parser')  #parses it using python builtin html parser
tb=soup.find('table',class_='wikitable')
tb_rows=tb.find_all('tr')
file=open("href.txt",'w')                        #opening the file in the write mode

for  tr in tb_rows:
    th=tr.find_all('th')                         #table headings
    sdf=[j.text for j in th]                     #finding the table headings to display the states names
    print('\n'.join(sdf[:1]))
    file.write(':'.join(sdf[:1]))                 #writing the states to the file
    td=tr.find_all('td')
    row=[i.text for i in td]                      #capitals are extracted by td tag and written to the file
    print(' '.join(row[0:2]))
    file.write(':'.join(row[0:2]))
    file.write('\n')
file.close()
