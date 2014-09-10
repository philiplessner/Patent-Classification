'''Combine stop word files
'''
import re
import codecs

BASE_PATH = ''.join(['/Users/dpmlto1/Documents/Patent/',
                     'Thomson Innovation/clustering/custom/'])


def generate_set(fname):
    '''Generate a set of stop words from a file
        Param:
                fname: file with one stop word per line
        Returns:
                set of stop words
    '''
    with open(fname, 'r') as f:
        # Strip newline character
        words = [re.sub(r'\s', '', line) for line in f]
    return set(words)

english = generate_set(''.join([BASE_PATH, 'english-stop-words.txt']))
print '*****English Stop Words*****'
print 'Number of Words= ', len(english)
print english

uspto = generate_set(''.join([BASE_PATH, 'uspto-stop-words.txt']))
print '*****USPTO Stop Words*****'
print 'Number of Words= ', len(uspto)
print uspto

custom = generate_set(''.join([BASE_PATH, 'corpus-specific-stop-words.txt']))
print '*****Custom Stop Words*****'
print 'Number of Words= ', len(uspto)
print custom

combined = english | uspto | custom
print '*****Combined Stop Words*****'
print 'Number of Words= ', len(combined)
print combined

cl = list(combined)
csort = cl.sort()
cs = '\n'.join(cl)

with codecs.open(''.join([BASE_PATH, 'combined-stop-words.txt']),
                 'w', 'utf-8') as f:
    f.write(cs)
