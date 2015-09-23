#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module to load a self-assembled training set

    The dataset needs to be stored in a pickle file. This file should contain
    two datastructures, one for the samples, and for the corresponding
    labels.

    After loading the dataset, PCA is applied to the data.
"""

import cv2
import numpy as np

import csv
from matplotlib import cm
from matplotlib import pyplot as plt

from os import path
import cPickle as pickle

__author__ = "Michael Beyeler"
__license__ = "GNU GPL 3.0 or later"


def load_data(load_from_file, test_split=0.2, num_components=50,
              save_to_file=None, plot_samples=False, seed=113):
    """Loads the dataset from a pickle file

        This function loads the dataset from a pickle file, performs feature
        extraction (PCA), and partitions the data into mutually exclusive
        training and test sets.

        :param load_from_file: filename to load dataset from
        :param test_split:     fraction of samples to reserve for the test set
        :param num_components: the number of PCA components to consider
        :param save_to_file:   filename to store preprocessed dataset
        :param plot_samples:   flag whether to plot samples (True) or not
                               (False)
        :param seed:           which random seed to use
        :returns:              (X_train, y_train), (X_test, y_test), V, m
    """
    # prepare lists for samples and labels
    X = []
    labels = []
    if not path.isfile(load_from_file):
        print "Could not find file", load_from_file
        return (X, labels), (X, labels), None, None
    else:
        print "Loading data from", load_from_file
        f = open(load_from_file, 'rb')
        samples = pickle.load(f)
        labels = pickle.load(f)
        print "Loaded", len(samples), "training samples"

        # perform feature extraction
        # returns preprocessed samples, PCA basis vectors & mean
        X, V, m = extract_features(samples, num_components=num_components)

        if plot_samples:
            print "Plotting samples not implemented"

    # shuffle dataset
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)

    # split data according to test_split
    X_train = X[:int(len(X)*(1-test_split))]
    y_train = labels[:int(len(X)*(1-test_split))]

    X_test = X[int(len(X)*(1-test_split)):]
    y_test = labels[int(len(X)*(1-test_split)):]

    if save_to_file is not None:
        # dump all relevant data structures to file
        f = open(save_to_file, 'wb')
        pickle.dump(X_train, f)
        pickle.dump(y_train, f)
        pickle.dump(X_test, f)
        pickle.dump(y_test, f)
        pickle.dump(V, f)
        pickle.dump(m, f)
        f.close()
        print "Save preprocessed data to", save_to_file

    return (X_train, y_train), (X_test, y_test), V, m


def load_from_file(file):
    """Loads the already preprocessed dataset from file

        This function loads a previously preprocessed dataset from a pickle
        file. The pickle file should contain X_train, y_train, X_test, y_test,
        V, and m.

        :returns: (X_train, y_train), (X_test, y_test), V, m
    """
    if path.isfile(file):
        # load all relevant data structures from file
        f = open(file, 'rb')
        X_train = pickle.load(f)
        y_train = pickle.load(f)
        X_test = pickle.load(f)
        y_test = pickle.load(f)
        V = pickle.load(f)
        m = pickle.load(f)
        f.close()

    return (X_train, y_train), (X_test, y_test), V, m


def extract_features(X, V=None, m=None, num_components=None):
    """Performs feature extraction

        This function is used to extract features from the dataset
        (currently only PCA is supported).
        It can also be used to preprocess a single data sample using some
        previously obtained PCA output. This makes it possible to obtain a
        set of basis vectors from an entire training set and apply this same
        set of basis vectors to a single test sample.

        :param X:   data (rows=images, cols=pixels)
        :param V:   PCA basis vectors
        :param m:   PCA mean vector
        :returns:       X (rows=samples, cols=features), V, m
    """
    if V is None or m is None:
        # need to perform PCA from scratch
        if num_components is None:
            num_components = 50

        # cols are pixels, rows are frames
        Xarr = np.squeeze(np.array(X).astype(np.float32))

        # perform PCA, returns mean and basis vectors
        m, V = cv2.PCACompute(Xarr)

        # use only the first num_components principal components
        V = V[:num_components]

    # backproject
    for i in xrange(len(X)):
        X[i] = np.dot(V, X[i] - m[0, i])

    return X, V, m
