# coding: utf-8
from __future__ import print_function, unicode_literals, division
import io
from subprocess import call
import tempfile
import os
import re
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from rake_nltk import RakeKeywordExtractor
from utils import lower_case, RegexpReplacer


class MakeSummary(object):
    '''
    Methods for summarizing text
    '''
    def __init__(self, r):
        '''
        Parameter
            r: compression ratio for summary (float)
        '''
        self.r = r

    def _preprocess(self, document):
        '''
        Strip excess white space and newlines
        Parameter
            document: document string
        Returns
            a string with spaces stripped from beginning and end and
            newlines replace with spaces
        '''
        return ' '.join(document.strip().split('\n'))

    def _find_strsent(self, sentences, strs=None):
        '''
        Find if strings are present in a list of strings.
        Parameters
            sentences: a list of strings
            strs: a list of strings to find in sentences
        Returns
            nfound_strs: a list of the total number of matches in
                         each sentence
        '''
        exact_match = re.compile(r'\b%s\b' % '\\b|\\b'.join(strs))
        found_strs = [exact_match.findall(sentence)
                      for sentence in sentences]
        nfound_strs = [len(found) for found in found_strs]
        return nfound_strs

    def _sent_tok(self, string):
        '''
        Takes a string and tokenizes it into sentences.
        Parameters
            string: a string
        Returns
            a list of strings with each string a sentence
        '''
        replacement_patterns = [(r'U[.]S[.]', 'US '),
                                (r'Pat[.]', 'Patent '),
                                (r'Fig[.]', 'Fig '),
                                (r'FIG[.]', 'Fig')]
        replacer = RegexpReplacer(patterns=replacement_patterns)
        cleaned = replacer.replace(string)
        stok = re.compile('[.!?][\s]{1,2}(?=[A-Z])')
        return stok.split(cleaned)

    def _get_topsent(self, sentence_scores):
        '''
        Get the top ranked sentences and join them in original order
        Parameter
            sentence_scores:
            list of tuples [(score, sentence=-string, sentence-position)...]
        Returns
            string of top ranked sentence in original order
        '''
        sentence_ranked = sorted(sentence_scores, key=lambda y: y[0],
                                 reverse=True)
        sum_length = int(round((self.r / 100.0) * len(sentence_scores)))
        tops = sentence_ranked[0:sum_length]
        tops_ordered = sorted(tops, key=lambda y: y[2])
        tops_ordered.append((0, '', 0))
        return '. '.join([x[1] for x in tops_ordered])

    def textrank(self, document):
        '''
        Use TextRank algorithm to find most relvant sentence in document
        Parameters
            document: document string
        Returns
            string of top ranked sentences in original order
        '''
        sentences = self._sent_tok(self._preprocess(document))

        bow_matrix = CountVectorizer().fit_transform(sentences)
        normalized = TfidfTransformer().fit_transform(bow_matrix)

        similarity_graph = normalized * normalized.T

        nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
        scores = nx.pagerank(nx_graph)
        # List of tuples [(score, sentence-string, sentence-position)...]
        sentence_scores = [(scores[i], s, i)
                           for i, s in enumerate(sentences)]
        return self._get_topsent(sentence_scores)

    def patentsum(self, document):
        '''
        Rank sentences according to presence of clue words and keywords (Rake)
        Parameter
            document: document string
        Returns
            string of top ranked sentences in original order
        '''
        clue_words = ['advantage', 'avoid', 'cost', 'costly', 'decrease',
                      'difficult', 'effectivenss', 'efficiency',
                      'goal', 'important', 'improved',
                      'increase', 'issue', 'limit', 'needed',
                      'overhead', 'performance', 'problem',
                      'reduced', 'resolve', 'shorten', 'simplify',
                      'suffer', 'superior', 'weakness']
        sent_tokens = self._sent_tok(self._preprocess(document))
        lcsent_tokens = lower_case(sent_tokens)
        # Check for clues in sentences
        nfound_clues = np.array(self._find_strsent(lcsent_tokens,
                                                   strs=clue_words))
        # Generate a list of keywords (rake)
        rake = RakeKeywordExtractor()
        keyphrases = rake.extract(document, incl_scores=True)
        phrases_only = [phrase for phrase, score in keyphrases]
        # Check for keywords phrases in sentences
        nfound_keyphrases = np.array(self._find_strsent(lcsent_tokens,
                                     strs=phrases_only[0:7]))
        # Rank sentences
        subtotals = nfound_clues + nfound_keyphrases
        # List of tuples [(score, sentence-stirng, sentence-position)...]
        sentence_scores = zip(subtotals, sent_tokens,
                              range(0, len(sent_tokens) + 1))
        return self._get_topsent(sentence_scores)

    def ots(self, document):
        '''
        Use ots to summarize content
        Parameters
            content: full text
        Returns
            outs: summarized text
        '''
        temp_dir = tempfile.mkdtemp()
        temp1 = tempfile.NamedTemporaryFile(
            suffix=".txt", dir=temp_dir, delete=False)
        temp2 = tempfile.NamedTemporaryFile(
            suffix=".txt", dir=temp_dir, delete=False)

        with io.open(temp1.name, 'w', encoding='utf-8') as f:
            f.write(self._preprocess(document))
        with io.open(temp2.name, 'w', encoding='utf-8') as outfile:
            call(['ots', '-r', unicode(int(self.r)), temp1.name],
                 stdout=outfile)
        with io.open(temp2.name, 'r', encoding='utf-8') as f:
            outs = f.read()
        os.remove(temp1.name)
        os.remove(temp2.name)
        return outs


if __name__ == '__main__':
    document = """To Sherlock Holmes she is always the woman. I have
seldom heard him mention her under any other name. In his eyes she
eclipses and predominates the whole of her sex. It was not that he
felt any emotion akin to love for Irene Adler. All emotions, and that
one particularly, were abhorrent to his cold, precise but admirably
balanced mind. He was, I take it, the most perfect reasoning and
observing machine that the world has seen, but as a lover he would
have placed himself in a false position. He never spoke of the softer
passions, save with a gibe and a sneer. They were admirable things for
the observer-excellent for drawing the veil from men’s motives and
actions. But for the trained reasoner to admit such intrusions into
his own delicate and finely adjusted temperament was to introduce a
distracting factor which might throw a doubt upon all his mental
results. Grit in a sensitive instrument, or a crack in one of his own
high-power lenses, would not be more disturbing than a strong emotion
in a nature such as his. And yet there was but one woman to him, and
that woman was the late Irene Adler, of dubious and questionable
memory.
"""
    with io.open('./test.txt', 'r', encoding='latin-1') as f:
        document2 = f.read()

    summary = MakeSummary(30.0)
    tops = summary.textrank(document2)
    print('\nTextRank\n', tops)
    outs = summary.ots(document2)
    print('\nOTS\n', outs)
    sect_summary = summary.patentsum(document2)
    print('\nPatent Summarizer\n', sect_summary)
