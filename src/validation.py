import numpy as np  # used to build arrays of row positions


def walk_forward_splits(n_weeks, n_folds=5, horizon=8):  # yields one (train, test) pair per fold
    first_cutoff = n_weeks - n_folds * horizon  # length of the first training window
    for i in range(n_folds):  # loop over each fold
        cutoff = first_cutoff + i * horizon  # training window grows by one horizon each fold
        train_idx = np.arange(0, cutoff)  # row positions from the start up to the cutoff
        test_idx = np.arange(cutoff, cutoff + horizon)  # the next 8 row positions after the cutoff
        yield train_idx, test_idx  # hand this fold's split back to the caller