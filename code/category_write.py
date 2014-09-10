import codecs
import pandas as pd

# Global Constants
PATH_BASE = ''.join(['/Users/dpmlto1/Documents/Patent/',
                     'Thomson Innovation/clustering/'])
df = pd.read_csv(PATH_BASE + 'output/output.csv')

for index, row in df.iterrows():
    with codecs.open(''.join([PATH_BASE, 'data/summaries/',
                              str(row['filenames'])]), 'r', 'latin_1') as f:
        data = f.read()
    with codecs.open(''.join([PATH_BASE, 'data/categories/',
                              str(row['Category']), '/',
                              str(row['filenames'])]),
                     'w', 'latin_1') as f:
        f.write(data)
