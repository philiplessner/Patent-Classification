import codecs
import pandas as pd


def decoder(pd_object):
    return pd_object.decode('latin_1', 'replace')


def write_datafiles(df, col_names, fname, fpath):
    '''Write pandas dataframe columns to a file
        Params:
            df: DataFrame
            col_names: list of column names
            fname: DataFrame column containing the filenames
            fpath: path to directory for files to be written
    '''
    for index, row in df.iterrows():
        with codecs.open(''.join([fpath, row[fname],
                                  '.txt']), 'w', 'latin_1') as f:
            for col_name in col_names:
                f.write(''.join([row[col_name], '\n']))


if __name__ == '__main__':
    READ_FILE = u''.join(['/Users/dpmlto1/Documents/Patent/',
                         'Thomson Innovation/patent-with-DWPI.csv'])
    cdict = {'Description': decoder}
    df = pd.read_csv(READ_FILE,
                     converters=cdict)
    print df
    col_names = ['Description']
    print df[col_names].head()
    df = df.dropna(how='all', subset=col_names)
    print 'New DataFrame'
    print '\n'
    print df
    fname = 'Publication Number'
    WRITE_PATH = u''.join(['/Users/dpmlto1/Documents/Patent/',
                           'Thomson Innovation/clustering/data/full_text/'])
    write_datafiles(df, col_names, fname, WRITE_PATH)
