# coding: utf-8
from __future__ import print_function, division, unicode_literals
import io
import re
import requests


def find(key, dictionary):
    for k, v in dictionary.iteritems():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                for result in find(key, d):
                    yield result


def main2():
    PATH_INPUT = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                          'clustering/nonus_toclassify.txt'])
    PATH_OUTPUT = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                           'clustering/data/unclassified/'])
    with io.open(PATH_INPUT, 'r', encoding='latin-1') as f:
        patents = [line.strip() for line in f]
    headers = {'Accept': 'application/json'}
    for patentn in patents:
        root_query = ''.join(['http://ops.epo.org/3.1/rest-services/',
                              'published-data/publication/epodoc/', patentn])
        fulltext_query = ''.join([root_query, '/fulltext'])
        fulltext_r = requests.get(fulltext_query, headers=headers)
        if fulltext_r.status_code == 200:
            biblio_query = ''.join([root_query, '/biblio'])
            desc_query = ''.join([root_query, '/description'])
            biblio_r = requests.get(biblio_query, headers=headers)
            biblio_d = biblio_r.json()
            t0 = biblio_d['ops:world-patent-data']['exchange-documents']
            if isinstance(list(find('invention-title', t0))[0], dict):
                title = ''.join(list(find('$',
                                          list(find('invention-title', t0))[0])))
            elif isinstance(list(find('invention-title', t0))[0], list):
                title = ''.join(list(find('$',
                                          list(find(
                                              'invention-title', t0))[0][0])))
            else:
                title = 'XXXX'
            if isinstance(list(find('abstract', t0))[0], dict):
                abstract = '\n'.join(list(find('$',
                                               list(find('abstract', t0))[0])))
            elif isinstance(list(find('abstract', t0))[0], list):
                abstract = '\n'.join(list(find('$',
                                               list(find('abstract', t0))[0][0])))
            else:
                abstract = 'XXXX'

            desc_r = requests.get(desc_query, headers=headers)
            desc_d = desc_r.json()
            t1 = desc_d['ops:world-patent-data']['ftxt:fulltext-documents']
            desc_body = t1['ftxt:fulltext-document']['description']
            if isinstance(desc_body['p'], dict):
                if isinstance(desc_body['p']['$'], dict):
                    description = '\n'.join([re.sub(r'\[\d+\]', '', value)
                                             for key, value in desc_body[
                                                 'p']['$'].iteritems()])
                elif isinstance(desc_body['p']['$'], unicode):
                    description = desc_body['p']['$']

            elif isinstance(desc_body['p'], list):
                description = '\n'.join([re.sub(r'\[\d+\]', '', item['$'])
                                         for item in desc_body['p']])
            else:
                description = 'XXXX'

            contents = ''.join(['Title\n', title, '\nAbstract\n', abstract,
                                '\n', description])
            filepath = ''.join([PATH_OUTPUT, patentn, '.txt'])
            with io.open(filepath, 'w', encoding='utf-8') as f:
                f.write(contents)
        else:
            print(patentn,  '\tInformation Not Available')


if __name__ == '__main__':
    main2()
