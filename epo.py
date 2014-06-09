# coding: utf-8
from __future__ import print_function, division, unicode_literals
import io
import re
import requests


def main():
    patentn = 'EP2618351'
    headers = {'content-type': 'application/json'}
    root_query = ''.join(['http://ops.epo.org/3.1/rest-services/',
                          'published-data/publication/epodoc/', patentn])
    biblo_query = ''.join([root_query, '/biblio'])
    desc_query = ''.join([root_query, '/description'])
    biblio_r = requests.get(biblo_query, headers=headers)
    biblio_d = biblio_r.json()
    t0 = biblio_d['ops:world-patent-data']['exchange-documents']
    title = t0[
        'exchange-document']['bibliographic-data']['invention-title'][0]['$']
    abstract = '\n'.join([item['$']
                          for item in t0['exchange-document']['abstract']['p']])
    desc_r = requests.get(desc_query, headers=headers)
    desc_d = desc_r.json()
    t1 = desc_d['ops:world-patent-data']['ftxt:fulltext-documents']
    desc_body = t1['ftxt:fulltext-document']['description']['p']
    description = '\n'.join([re.sub(r'\[\d+\]', '', item['$'])
                             for item in desc_body])
    contents = ''.join(['Title\n', title, '\nAbstract\n', abstract,
                        '\n', description])
    with io.open('test_epo.txt', 'w', encoding='utf-8') as f:
        f.write(contents)

if __name__ == '__main__':
    main()
