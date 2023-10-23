import numpy as np
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import eigs
from scipy.interpolate import InterpolatedUnivariateSpline


class TransmonInterpolation:
    def __init__(self, ng, EjEc, cutoff):
        if cutoff < 2:
            raise ValueError("Cutoff is too low; it must be at least 2.")
        self.ng = ng
        self.EjEc = EjEc
        self.cutoff = cutoff
        self.energy, self.coupling = self.calculate_transmon_interpolation()

    def calculate_transmon_interpolation(self):
        h = diags([4 * (np.arange(-self.cutoff, self.cutoff + 1) - self.ng) ** 2], 0)
        hv = spdiags([-np.ones(2 * self.cutoff), -np.ones(2 * self.cutoff)], [1, -1],
                     (2 * self.cutoff + 1, 2 * self.cutoff + 1))
        n = diags([np.arange(-self.cutoff, self.cutoff + 1)], 0)
        e, v = eigs(h + self.EjEc * hv / 2, k=2 * self.cutoff + 1)
        o = np.argsort(e)
        e2 = e[o]
        v2 = v[:, o]
        g = v2 @ n @ v2.T
        sgn = np.sign(g)
        g = sgn * g
        de = np.array([v2[:, i] @ hv @ v2[:, i] / 2 for i in range(2 * self.cutoff + 1)])
        dv2 = np.array(
            [np.sum([v2[j, :] @ hv @ v2[i, :] / (e2[i] - e2[j]) if i != j else 0 for j in range(2 * self.cutoff + 1)])
             for i in range(2 * self.cutoff + 1)])
        dg = sgn * (dv2 @ n @ v2.T + v2 @ n @ dv2.T) / 2
        energy = [e2 - e2[0], de - de[0], g]
        coupling = [dg, np.gradient(g, axis=0)]
        return energy, coupling

    def energy_interp(self, i, EjEc):
        if not (0 <= i < self.cutoff):
            raise ValueError(f"Invalid level index: {i}, should be in the range 0 to {self.cutoff - 1}")
        if not (self.energy[0][0] <= EjEc <= self.energy[0][-1]):
            raise ValueError(
                f"Invalid value of EjEc: {EjEc}, should be between {self.energy[0][0]} and {self.energy[0][-1]}")
        spline = InterpolatedUnivariateSpline(self.energy[0], self.energy[i + 1])
        return spline(EjEc)

    def coupling_interp(self, i, j, EjEc):
        if not (0 <= i <= self.cutoff) or not (0 <= j <= self.cutoff):
            raise ValueError(f"Invalid level indices: {i}, {j}, should be in the range 0 to {self.cutoff}")
        if not (self.energy[0][0] <= EjEc <= self.energy[0][-1]):
            raise ValueError(
                f"Invalid value of EjEc: {EjEc}, should be between {self.energy[0][0]} and {self.energy[0][-1]}")
        spline = InterpolatedUnivariateSpline(self.energy[0], self.coupling[0][i, j])
        return spline(EjEc)

    def egtrans(self, ng, EjEc, cutoff):
        if cutoff < 2:
            raise ValueError("Cutoff is too low; it must be at least 2.")
        h = diags([4 * (np.arange(-cutoff, cutoff + 1) - ng) ** 2], 0)
        hv = spdiags([-np.ones(2 * cutoff), -np.ones(2 * cutoff)], [1, -1], (2 * cutoff + 1, 2 * cutoff + 1))
        n = diags([np.arange(-cutoff, cutoff + 1)], 0)
        e, v = eigs(h + EjEc * hv / 2, k=2 * cutoff + 1)
        o = np.argsort(e)
        e2 = e[o]
        v2 = v[:, o]
        g = v2 @ n @ v2.T
        sgn = np.sign(g)
        g = sgn * g
        de = np.array([v2[:, i] @ hv @ v2[:, i] / 2 for i in range(2 * cutoff + 1)])
        dv2 = np.array(
            [np.sum([v2[j, :] @ hv @ v2[i, :] / (e2[i] - e2[j]) if i != j else 0 for j in range(2 * cutoff + 1)]) for i
             in range(2 * cutoff + 1)])
        dg = sgn * (dv2 @ n @ v2.T + v2 @ n @ dv2.T) / 2

        energy = [e2 - e2[0], de - de[0], g]
        dg_over_dEjEc = (dv2 @ n @ v2.T + v2 @ n @ dv2.T) / 2
        dg_over_dEjEc[0, 0] = 0.0  # Avoid division by zero
        dg_over_dEjEc = np.abs(dg_over_dEjEc)
        differential_g_over_dEjEc = np.gradient(dg_over_dEjEc, axis=0)

        return {
            "EjEcValues": e2 - e2[0],
            "dEjEcValues": de - de[0],
            "gValues": g,
            "differential_g_over_dEjEc": differential_g_over_dEjEc
        }
