# coding: utf-8
from __future__ import print_function, division, unicode_literals
import io
import re
import requests

patentn = 'EP2618351'
query = ''.join(['http://ops.epo.org/3.1/rest-services/published-data/',
                 'publication/epodoc/', patentn, '/description'])
headers = {'content-type': 'application/json'}
r = requests.get(query, headers=headers)
d = r.json()

t1 = d['ops:world-patent-data']['ftxt:fulltext-documents']
desc_body = t1['ftxt:fulltext-document']['description']['p']

with io.open('test_desc.txt', 'w', encoding='utf-8') as f:
    for item in desc_body:
        item_cleaned = re.sub(r'\[\d+\]', '', item['$'])
        f.write(item_cleaned + '\n')
