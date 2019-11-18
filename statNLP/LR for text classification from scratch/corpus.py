# -*- mode: Python; coding: utf-8 -*-

"""For the purposes of classification, a corpus is defined as a collection
of labeled documents. Such documents might actually represent words, images,
etc.; to the classifier they are merely instances with features."""

from abc import ABCMeta, abstractmethod
from csv import reader as csv_reader
from glob import glob
import json
from os.path import basename, dirname, split, splitext
import string

class Document(object):
    """A document completely characterized by its features."""

    max_display_data = 10 # limit for data abbreviation

    def __init__(self, data, label=None, source=None):
        self.data = data
        self.label = label
        self.source = source
        self.feature_vector = []

    def __repr__(self):
        return ("<%s: %s>" % (self.label, self.abbrev()) if self.label else
                "%s" % self.abbrev())

    def abbrev(self):
        return (self.data if len(self.data) < self.max_display_data else
                self.data[0:self.max_display_data] + "...")

    def features(self):
        """A list of features that characterize this document."""
        return [self.data]

class Corpus(object):
    """An abstract collection of documents."""

    __metaclass__ = ABCMeta

    def __init__(self, datafiles, document_class=Document):
        self.documents = []
        self.datafiles = glob(datafiles)
        for datafile in self.datafiles:
            self.load(datafile, document_class)

    # Act as a mutable container for documents.
    def __len__(self): return len(self.documents)
    def __iter__(self): return iter(self.documents)
    def __getitem__(self, key): return self.documents[key]
    def __setitem__(self, key, value): self.documents[key] = value
    def __delitem__(self, key): del self.documents[key]

    @abstractmethod
    def load(self, datafile, document_class):
        """Make labeled document instances for the data in a file."""
        pass

class PlainTextFiles(Corpus):
    """A corpus contained in a collection of plain-text files."""

    def load(self, datafile, document_class):
        """Make a document from a plain-text datafile. The document is labeled
        using the last component of the datafile's directory."""
        label = split(dirname(datafile))[-1]
        with open(datafile, "r") as file:
            data = file.read()
            self.documents.append(document_class(data, label, datafile))

class PlainTextLines(Corpus):
    """A corpus in which each document is a line in a datafile."""

    def load(self, datafile, document_class):
        """Make a document from each line of a plain text datafile.
        The document is labeled using the datafile name, sans directory
        and extension."""
        label = splitext(basename(datafile))[0]
        with open(datafile, "r") as file:
            for line in file:
                data = line.strip()
                self.documents.append(document_class(data, label, datafile))

class BagOfWords(Document):
    def features(self):
        """Trivially tokenized words."""
        text = self.data.lower()
        text = text.translate(str.maketrans('','',string.punctuation+'\n'))
        return text.split()

class BagOfWordsBigram(Document):
    def features(self):
        """Trivially tokenized words."""
        text = self.data.lower()
        text = text.translate(str.maketrans('','',string.punctuation+'\n'))
        onegram = text.split()
        bigram = [onegram[i] + onegram[i+1] for i in range(len(onegram)-1)]
        onegram.extend(bigram)
        return onegram

class BagOfWordsTrigram(Document):
    def features(self):
        """Trivially tokenized words."""
        text = self.data.lower()
        text = text.translate(str.maketrans('','',string.punctuation+'\n'))
        onegram = text.split()
        bigram = [onegram[i] + onegram[i+1] for i in range(len(onegram)-1)]
        trigram = [onegram[i] + onegram[i+1] +onegram[i+2] for i in range(len(onegram)-2)]
        onegram.extend(bigram)
        onegram.extend(trigram)
        return onegram


class Name(Document):
    def features(self):
        name = self.data
        return ['First=%s' % name[0], 'Last=%s' % name[-1]] 

class NamesCorpus(PlainTextLines):
    """A collection of names, labeled by gender. See names/README for
    copyright and license."""

    def __init__(self, datafiles="names/*.txt", document_class=Document):
        super(NamesCorpus, self).__init__(datafiles, document_class)

class ReviewCorpus(Corpus):
    """Yelp dataset challenge. A collection of business reviews. 
    """

    def load(self, datafile, document_class):
        """Make a document from each row of a json-formatted Yelp reviews
        """
        with open(datafile, "r") as file:
            for line in file:
                review = json.loads(line)
                label = review['sentiment']
                data = review['text']
                #every data in corpus is a document class
                self.documents.append(document_class(data, label, datafile))
