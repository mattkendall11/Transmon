import numpy as np
from scipy.linalg import eigvals

class Transmon:
    def __int__(self, ng, EjEc, cutoff):
        self.ng = ng
        self.EjEc = EjEc
        self.cutoff = cutoff

    def egtrans(self):
        if self.cutoff <= 1:
            raise ValueError("Cutoff must be greater than 1")

            # Sparse matrix h
        h = 4 * (np.arange(-self.cutoff, self.cutoff + 1) - self.ng) ** 2

        # Sparse matrix hv
        hv = -np.eye(2 * self.cutoff + 1)
        hv[np.diag_indices(2 * self.cutoff)] = 0.0

        # Sparse matrix n
        n = np.diag(np.arange(-self.cutoff, self.cutoff + 1))

        # Eigenvalue computation
        e, v = eigvals(h + self.EjEc * hv / 2, overwrite_a=True)

        # Other calculations, derivatives, and transformations

        return e, v