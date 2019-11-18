# -*- mode: Python; coding: utf-8 -*-

"""A simple framework for supervised text classification."""

from abc import ABCMeta, abstractmethod, abstractproperty
from pickle import dump, load, HIGHEST_PROTOCOL as HIGHEST_PICKLE_PROTOCOL
import io

class Classifier(object):
    """An abstract text classifier.

    Subclasses must provide training and classification methods, as well as
    an implementation of the model property. The internal representation of
    a classifier's model is entirely up to the subclass, but the read/write
    model property must return/accept a single object (e.g., a list of
    probability distributions)."""

    __metaclass__ = ABCMeta

    def __init__(self, model=None):
        if isinstance(model, (str, io.IOBase)):
            self.load(model)
        else:
            self.model = model

    def get_model(self): return None
    def set_model(self, model): pass
    model = abstractproperty(get_model, set_model)

    def save(self, file_):
        """Save the current model to the given file."""
        if isinstance(file_, str):
            with open(file_, "wb") as file_:
                self.save(file_)
        else:
            print(type(file_))
            dump(self.model, file_, HIGHEST_PICKLE_PROTOCOL)

    def load(self, file_):
        """Load a saved model from the given file."""
        if isinstance(file_, str):
            with open(file_, "rb") as file_:
                self.load(file_)
        else:
            self.model = load(file_)

    @abstractmethod
    def train(self, instances):
        """Construct a statistical model from labeled instances."""
        pass

    @abstractmethod
    def classify(self, instance):
        """Classify an instance and return the expected label."""
        return None
