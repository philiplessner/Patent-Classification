#coding: utf-8
from __future__ import print_function, unicode_literals
import io
import re
import requests
from lxml import html
from utils import RegexpReplacer

PATH_OUTPUT = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                       'clustering/data/uspto-full-text/anode/'])


def clean(page):
    '''
    Clean up full text patents downloaded from USPTO site
    Parameters
        page: object that requests module returns from get method
    Returns
        cleaned page with html tags stripped
    '''
    patterns = [(r'\n', r' '),
                (r'<BR>+', r'\n'),
                (r'Abstract', r'\nAbstract\n'),
                (r'Prior Publication Data', ''),
                (r'Foreign Application Priority Data', ''),
                (r'References Cited', ''),
                (r'Referenced By', ''),
                (r'U[.]S[.] Patent Documents', ''),
                (r'Foreign Patent Documents', ''),
                (r'Primary Examiner:', ''),
                (r'Attorney, Agent or Firm:', ''),
                (r'United States Patent:[\s\d]+', r'Title\n')]
    replacer = RegexpReplacer(patterns=patterns)
    semiclean = replacer.replace(page.text)
    tree = html.fromstring(semiclean)
    # Remove tables so prior art patents, references, etc. don't appear
    # at begining of file
    tables = tree.xpath('.//table')
    for table in tables:
        table.getparent().remove(table)
    text = unicode(tree.text_content())
    cleantext = re.sub(r'\[[\W\w]+(?=Claims)', r'\n\n', text)
    return cleantext


patents = ['US3345545', 'US7116548', 'US7154742', 'US4945452', 'US5949639',
           'US6151205', 'US6191936', 'US7116548',
           'US7154742', 'US3818286', 'US4571626',
           'US5959831', 'US7190572']
pat = re.compile(r'\D')
for patent in patents:
    patentn = pat.sub('', patent)
    url = ''.join(['http://patft.uspto.gov/netacgi/nph-Parser?',
                   'Sect1=PTO1&Sect2=HITOFF',
                   '&d=PALL&p=1&u=%2Fnetahtml%2FPTO%2Fsrchnum.htm',
                   '&r=1&f=G&l=50', '&s1=',  patentn, '.PN.&OS=PN/',
                   patentn, '&RS=PN/',  patentn])
    page = requests.get(url)
    filepath = ''.join([PATH_OUTPUT, patent, '.txt'])
    with io.open(filepath, 'w', encoding='latin-1') as f:
        f.write(clean(page))
