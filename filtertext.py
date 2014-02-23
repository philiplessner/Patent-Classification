# coding: utf-8
from __future__ import print_function, unicode_literals
import os
import io
import re
from utils import pipe


def gen_filelist(top):
    '''
    List of files in a directory tree
    Parameters
        top: path to top of directory tree
    Returns
        generator object for path to each file in tree
    '''
    for path, dirlist, filelist in os.walk(top):
        for filename in filelist:
            yield os.path.join(path, filename)


def gen_open(filenames, mode='r', encode='latin-1'):
    '''
    File objects for files
    Parameters
        filnames: object to generate filenames
        mode: w, r, a (r is default)
        encode: encoding
    Returns
        generator object for file objects
    '''
    for filename in filenames:
        yield io.open(filename, mode, encoding=encode)


def gen_cat(sources):
    '''
    Lines in files
    Parameters
        sources: file objects
    Returns
        generator for lines in files
    '''
    for s in sources:
        for item in s:
            yield item


def gen_grep(pat, lines):
    '''
    Lines matching pattern
        pat: pattern to match (string)
        lines: object that returns lines in a file
    Returns
        line matching pattern
    '''
    patc = re.compile(pat)
    for line in lines:
        if patc.match(line.strip()):
            yield line.strip()


def lines_fromfiles(top):
    '''
    Generate lines from files
    Parameter
        top: path to top of directory tree
    Returns
        generator for lines in files
    '''
    return pipe((gen_filelist, ), (gen_open, ), (gen_cat, ))(top)


if __name__ == '__main__':
    PATH_BASE = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                         'clustering/data/uspto-full-text'])
    lines = pipe((lines_fromfiles, ), (gen_grep, r'^[A-Z\s]+$'))(PATH_BASE)
    capset = set([line for line in lines])
    for member in capset:
        print(member)
