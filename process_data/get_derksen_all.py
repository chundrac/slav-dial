from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.parse import quote

#etyma = []

#for i in range(1,109):
#    page = urlopen('https://ordbog.oesteuropastudier.dk/index.php/list/%5Bsla-eng%5D+Derksen+-+Slavic+Inherited+Lexicon/{}.xhtml'.format(i))
#    html = page.read()
#    soup = BeautifulSoup(html)
#    for t in soup.find_all('td',{'class':'termpreview'}):
#        etyma.append('https://ordbog.oesteuropastudier.dk'+t.find('a')['href'])


#etyma = sorted(set(etyma))


#f = open('derksen_etyma.txt','w')
#print('\n'.join(etyma),file=f)
#f.close()


etyma = [l.strip() for l in open('derksen_etyma.txt','r')]

f = open('derksen_uncleaned_data.tsv','w')
f.close()
reflexes = []
for e in etyma:
    print(e)
    #page = urlopen('https://ordbog.oesteuropastudier.dk/index.php/term/%5Bsla-eng%5D+Derksen+-+Slavic+Inherited+Lexicon,{}.xhtml'.format(quote(e)))
    page = urlopen(e)
    html = page.read()
    soup = BeautifulSoup(html)
    if '<strong>Grammatical information:</strong>' in str(soup) or '<strong>Proto-Slavic Meaning:</strong>' in str(soup):
        etymon = soup.find('h1',{'class':'term'}).text
        etymon_alt = soup.find('span',{'style':'color:red;'}).text
        #etymon = soup.find('span',{'style':'color:red;'}).find('strong').text
        gram = ''
        if 'Grammatical information' in soup.find('span',{'style':'color:green;'}).text:
            gram = soup.find('span',{'style':'color:green;'}).text.split(': ')[1].strip()
        accent = ''
        if len(soup.find_all('span',{'style':'color:green;'})) > 1 and 'Accent paradigm' in soup.find_all('span',{'style':'color:green;'})[1].text:
            accent = soup.find_all('span',{'style':'color:green;'})[1].text.split(': ')[1].strip()
        #gram = soup.find('span',{'style':'color:green;'}).text
        #if 'Grammatical information' in gram:
        #    gram = gram.split(': ')[1].strip()
        #else:
        #    gram = ''
        #accent = soup.find_all('span',{'style':'color:green;'})[1].text
        lang=''
        reflex=''
        reflex_class = ''
        counter = 0
        for s in soup.find_all(['strong','span']):
            if s.name=='strong':
                lang = s.text 
                counter = 1
            if s.name == 'span' and 'style' in s.attrs and s['style']=="color:darkred;" and counter == 1:
                reflex = s.text
            if s.name == 'span' and 'style' in s.attrs and s['style']=="color:green;" and counter == 1:
                reflex_class = s.text
                counter = 0
                f = open('derksen_uncleaned_data.tsv','a')
                print('\t'.join([etymon,gram,accent,lang,reflex,reflex_class,etymon_alt]),file=f)
                f.close()