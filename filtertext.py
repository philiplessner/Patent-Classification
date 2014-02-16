# coding: utf-8
from __future__ import print_function, unicode_literals
import os
import io
import re
from itertools import starmap
from functools import partial


def compose_two(g, f):
    '''
    Function composition for two functions,
    e.g. compose_two(f, g)(x) == f(g(x))
     '''
    return lambda *args, **kwargs: g(f(*args, **kwargs))


def compose(*funcs):
    '''
    Compose an arbitrary number of functions
    left-to-right passed as args
    '''
    return reduce(compose_two, funcs)


def transform_args(func, transformer):
    return lambda *args: func(*transformer(args))

composed_partials = transform_args(compose, partial(starmap, partial))
pipe = transform_args(composed_partials, reversed)


def gen_filelist(top):
    for path, dirlist, filelist in os.walk(top):
        for filename in filelist:
            yield os.path.join(path, filename)


def gen_open(filenames, mode='r', encode='latin-1'):
    for filename in filenames:
        yield io.open(filename, mode, encoding=encode)


def gen_cat(sources):
    for s in sources:
        for item in s:
            yield item


def gen_grep(pat, lines):
    patc = re.compile(pat)
    for line in lines:
        if patc.match(line.strip()):
            yield line.strip()


def lines_fromfiles(top):
    return pipe((gen_filelist, ), (gen_open, ), (gen_cat, ))(top)


if __name__ == '__main__':
    PATH_BASE = ''.join(['/Users/dpmlto1/Documents/Patent/Thomson Innovation/',
                         'clustering/data/uspto-full-text'])
    lines = pipe((lines_fromfiles, ), (gen_grep, r'^[A-Z\s]+$'))(PATH_BASE)
    capset = set([line for line in lines])
    for member in capset:
        print(member)
