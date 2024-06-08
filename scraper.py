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


def clean():
    with open('docs/output.txt', 'r') as f:
        lines = f.readlines()
    s = set()
    with open('docs/output.txt', 'w') as f:
        for line in lines:
            if line not in s:
                f.write(line)
                s.add(line)

clean()