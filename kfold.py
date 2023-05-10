import numpy as np
import numbers
import pandas as pd
import os
from configparser import ConfigParser

"""
*    Title: Scikit-learn: Machine Learning in Python
*    Author: Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.
*    Date: 2011
*    Code version: 1.2.1
*    Availability: https://scikit-learn.org/stable/index.html
"""


def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


class BaseCrossValidator():
    """Base class for all cross-validators

    Implementations must define `_iter_test_masks` or `_iter_test_indices`.
    """

    def split(self, X, y=None, groups=None):
        # X, y, groups = indexable(X, y, groups)
        indices = np.arange(len(X))
        for test_index in self._iter_test_masks(X, y, groups):
            train_index = indices[np.logical_not(test_index)]
            test_index = indices[test_index]
            yield train_index, test_index

    def _iter_test_masks(self, X=None, y=None, groups=None):
        """Generates boolean masks corresponding to test sets.

        By default, delegates to _iter_test_indices(X, y, groups)
        """
        for test_index in self._iter_test_indices(X, y, groups):
            test_mask = np.zeros(len(X), dtype=bool)
            test_mask[test_index] = True
            yield test_mask

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Generates integer indices corresponding to test sets."""
        raise NotImplementedError


class _BaseKFold(BaseCrossValidator):
    def __init__(self, n_splits, *, shuffle, random_state):
        n_splits = int(n_splits)

        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits)
            )

        if not isinstance(shuffle, bool):
            raise TypeError(
                "shuffle must be True or False; got {0}".format(shuffle))

        if not shuffle and random_state is not None:  # None is the default
            raise ValueError(
                "Setting a random_state has no effect since shuffle is "
                "False. You should leave "
                "random_state to its default (None), or set shuffle=True.",
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        # X, y, groups = indexable(X, y, groups)
        n_samples = len(X)
        if self.n_splits > n_samples:
            raise ValueError(
                (
                    "Cannot have number of splits n_splits={0} greater"
                    " than the number of samples: n_samples={1}."
                ).format(self.n_splits, n_samples)
            )

        for train, test in super().split(X, y, groups):
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class KFold(_BaseKFold):
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            check_random_state(self.random_state).shuffle(indices)

        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
        fold_sizes[: n_samples % n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            yield indices[start:stop]
            current = stop


def kfold_data(save_path: str, csv_path: str, n_splits: int, random_state: int):
    df = pd.read_csv(csv_path, sep=' ', dtype=str, index_col=0)
    df = pd.DataFrame(df)
    X = np.array(df)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        trainset = pd.DataFrame(X[train_index])
        testset = pd.DataFrame(X[test_index])
        trainsaveto = str(i)+'_'+parser["common"]["kfoldTrainDataCsv"]
        testsaveto = str(i)+'_'+parser["common"]["kfoldTestDataCsv"]
        trainset.to_csv(save_path+'/'+trainsaveto, sep=' ',
                        header=['input', 'output'])
        testset.to_csv(save_path+'/'+testsaveto, sep=' ',
                       header=['input', 'output'])


if __name__ == "__main__":

    parser = ConfigParser()
    parser.read("config.ini")

    csv_path = parser["common"]["fullDataCsv"]
    save_path = parser["common"]['kfoldpath']
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    kfold_data(save_path, csv_path, 10, 3)
