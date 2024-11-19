import numpy as np


class LFApplier:
    """
    Applies the labelling functions to the data that is passed in. Returns a matrix of the data with the lfs applied.
    """

    def __init__(self, lfs: list, testing: list):
        """
        Pass in the labelling functions and the data to be tested on.

        :param lfs: List of labelling functions.
        :param testing: List of data to use the labelling functions on.
        """
        self.lfs = lfs
        self.testing = testing

    def apply_lfs(self) -> np.ndarray:
        """
        Applies the lfs to the testing data.

        :return: Matrix of the testing data with the lfs applied, rows are the sentences, columns are the lfs.
        """
        # Apply the lfs to the testing data
        lfs_testing = None
        print("Applying lfs to testing data...")
        for lf in self.lfs:
            if lfs_testing is None:
                lfs_testing = lf.apply(self.testing)
            else:
                lfs_testing = np.hstack((lfs_testing, lf.apply(self.testing)))
        return lfs_testing
