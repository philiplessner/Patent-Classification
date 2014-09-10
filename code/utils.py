# coding: utf-8
import re
from itertools import starmap
from functools import partial


class RegexpReplacer(object):

    '''
    Replaces regular expression in a text.
    '''

    def __init__(self, patterns=None):
        self.patterns = [(re.compile(regex), repl)
                         for (regex, repl) in patterns]

    def replace(self, text):
        s = text

        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)

        return s


def lower_case(strings):
    '''Takes a list of strings and converts them
       to lower case.
        Params:
            strings: list containing strings, e.g.,
            [str1, str2, str3... strN]
        Returns:
            New list of strings converted to lower case, e.g.,
            [lower(str1), lower(str2), lower(str3)... lower(strN)]
    '''
    return [string.lower() for string in strings]


def removeNonAscii(s):
    return "".join(i for i in s if ord(i) < 128)


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
