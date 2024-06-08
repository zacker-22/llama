def get_links():
    with open('links.txt') as f:
        for line in f.readlines():
            if line.startswith('http'):
                yield line.strip()


import requests
import html2text
import unidecode


def get_page(url):
    # return the page content
    
    return unidecode.unidecode(html2text.html2text( requests.get(url).text ))



for i in get_links():
    print(i)
    if i.endswith('pdf'):
        continue
    with open('output.txt', 'a') as f:
        f.write(get_page(i))
        f.write('\n')

