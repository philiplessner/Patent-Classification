# coding: utf-8
from __future__ import print_function, unicode_literals
import io
import re
import requests
from lxml import html
from utils import RegexpReplacer

PATH_INPUT = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                      'clustering/data/publication-list.txt'])
PATH_OUTPUT = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                       'clustering/data/uspto-full-text/test/'])


def clean_patent(page):
    '''
    Clean up full text patents downloaded from USPTO site
    Parameters
        page: object that requests module returns from get method
    Returns
        cleaned page with html tags stripped
    '''
    # Clean up HTML
    patterns_h = [(r'\n', r' '),
                  (r'<BR>+', r'\n'),
                  (r'Abstract', r'\nAbstract\n')]
    semiclean = RegexpReplacer(patterns=patterns_h).replace(page.text)
    tree = html.fromstring(semiclean)
    # Remove tables so prior art patents, references, etc. don't appear
    # at begining of file
    tables = tree.xpath('.//table')
    for table in tables:
        table.getparent().remove(table)
    # Strip HTML and convert Element Tree object to unicode object
    text = unicode(tree.text_content())
    # Remove table headings and put 'Claims' on a separate line
    patterns_t = [(r'Prior Publication Data', ''),
                  (r'Foreign Application Priority Data', ''),
                  (r'References Cited', ''),
                  (r'Referenced By', ''),
                  (r'U[.]S[.] Patent Documents', ''),
                  (r'Foreign Patent Documents', ''),
                  (r'Primary Examiner:', ''),
                  (r'Attorney, Agent or Firm:', ''),
                  (r'United States Patent:[\s\d]+', r'Title\n'),
                  (r'\[[\W\w]+(?=Claims)', r'\n\n'),
                  (r'[ ]+', ' ')]
    cleantext = RegexpReplacer(patterns=patterns_t).replace(text)
    return cleantext


def clean_patentapp(page):
    '''
    Clean up full text patents downloaded from USPTO site
    Parameters
        page: object that requests module returns from get method
    Returns
        cleaned page with html tags stripped
    '''
    # Clean up HTML
    patterns_h = [(r'\n', r' '),
                  (r'<BR>+', r'\n'),
                  (r'Abstract', r'\nAbstract\n')]
    semiclean = RegexpReplacer(patterns=patterns_h).replace(page.text)
    tree = html.fromstring(semiclean)
    # Remove tables so prior art patents, references, etc. don't appear
    # at begining of file
    tables = tree.xpath('/html/body/table')
    for i, table in enumerate(tables):
        if i != 2:
            table.getparent().remove(table)
    styles = tree.xpath('.//style')
    for style in styles:
        style.getparent().remove(style)
    # Strip HTML and convert Element Tree object to unicode object
    text = unicode(tree.text_content())
    # Remove table headings and put 'Claims' on a separate line
    patterns_t = [(r'Foreign Application Priority Data', ''),
                  (r'Related U[.]S[.] Patent Documents', ''),
                  (r'Foreign Patent Documents', ''),
                  (r'Primary Examiner:', ''),
                  (r'Attorney, Agent or Firm:', ''),
                  (r'United States Patent Application:[\s\d]+', r'Title\n'),
                  (r'Inventors[\W\w]+(?=Claims)', r'\n\n'),
                  (r'Claims', r'\n\nClaims\n'),
                  (r'\[\d+\]', ''),
                  (r'[ ]+', ' ')]
    cleantext = RegexpReplacer(patterns=patterns_t).replace(text)
    return cleantext


with io.open(PATH_INPUT, 'r', encoding='latin-1') as f:
    patents = [line.strip() for line in f]
pat = re.compile(r'\D')
for patent in patents:
    patentn = pat.sub('', patent)
    if len(patentn) == 7:
        url = ''.join(['http://patft.uspto.gov/netacgi/nph-Parser?',
                       'Sect1=PTO1&Sect2=HITOFF',
                       '&d=PALL&p=1&u=%2Fnetahtml%2FPTO%2Fsrchnum.htm',
                       '&r=1&f=G&l=50', '&s1=', patentn, '.PN.&OS=PN/',
                       patentn, '&RS=PN/', patentn])
        page = requests.get(url)
        cleaned = clean_patent(page)
    else:
        url = ''.join(['http://appft.uspto.gov/netacgi/nph-Parser?',
                       'Sect1=PTO1&Sect2=HITOFF',
                       '&d=PG01&p=1&u=%2Fnetahtml%2FPTO%2Fsrchnum.html',
                       '&r=1&f=G&l=50&s1=%22', patentn, '%22.PGNR.&OS=DN/',
                       patentn, '&RS=DN/', patentn])
        page = requests.get(url)
        cleaned = clean_patentapp(page)
    filepath = ''.join([PATH_OUTPUT, patent, '.txt'])
    with io.open(filepath, 'w', encoding='latin-1') as f:
        f.write(cleaned)
