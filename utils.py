import re


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
