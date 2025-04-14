"""This module computes general constrained equilibria from
`webnucleo <https://webnucleo.readthedocs.io>`_ files."""

from scipy.optimize import elementwise
import numpy as np
import wneq.base as wqb


class Equil(wqb.Base):
    """A class for handling constrained equilibria."""

    def __init__(self, nuc):
        wqb.Base.__init__(self, nuc)

        self.ye = None
        self.fac = {}
        self.clusters = {}
        self.cluster_nuclides = {}
        self.cluster_mus = {}
        self.mup_kt = 0

    def compute(self, t_9, rho, ye=None, clusters=None):
        """Method to compute a nuclear equilibrium.

        Args:
            ``t_9`` (:obj:`float`): The temperature (in 10 :sup:`9` Kelvin)
            at which to compute the equilibrium.

            ``rho`` (:obj:`float`): The mass density in grams per cc  at which
            to compute the equilibrium.

            ``mun`` (:obj:`float`): The neutron chemical potential (in MeV)
            at which to compute the equilibrium..

            ``ye`` (:obj:`float`): The electron fraction at which to compute
            the equilibrium.  If not supplied, the routine computes the equilibrium
            without a fixed total neutron-to-proton ratio.

            ``clisters`` (:obj:`dict`): A dictionary with the key for each
            entry giving the XPath describing the cluster and the value giving the
            abundance constraint for the cluster.

        Returns:
            A `wnutils <https://wnutils.readthedocs.io>`_ zone data dictionary
            with the results of the calculation.

        """

        self.ye = ye
        self._update_fac(t_9, rho)

        if clusters:
            self._set_clusters(clusters)
        else:
            self.clusters.clear()
            self.cluster_nuclides.clear()
            self.cluster_mus.clear()

        res_bracket = elementwise.bracket_root(self._compute_a_root, -10)
        res_root = elementwise.find_root(
            self._compute_a_root, res_bracket.bracket
        )
        self.mun_kt = res_root.x

        y = self._compute_abundances(self.mup_kt, self.mun_kt)
        mass_fracs = self._convert_to_mass_fractions(y)
        x_sum = 0
        for key, value in mass_fracs.items():
            if value > 1.0e-30:
                x_sum += value
                print(key, value)
        print(x_sum)

    def _convert_to_mass_fractions(self, _y):
        result = {}
        for key, value in self.nuc.get_nuclides().items():
            if isinstance(_y[key], float):
                result[key] = value["a"] * _y[key]
            else:
                result[key] = value["a"] * _y[key][0]
        return result

    def _set_clusters(self, clusters):
        self.clusters = clusters
        for key, value in clusters.items():
            self.cluster_nuclides[key] = self.nuc.get_nuclides(nuc_xpath=value)
        cluster_nuclide_set = set()
        for key in self.clusters:
            for nuc in self.cluster_nuclides[key]:
                assert nuc not in cluster_nuclide_set
                cluster_nuclide_set.add(nuc)

    def _check_fac(self, _x):
        x_max = 300
        if isinstance(_x, float):
            return min(_x, x_max)

        for i, v_x in enumerate(_x):
            _x[i] = min(v_x, x_max)
        return _x

    def _compute_abundances(self, mup_kt, mun_kt):
        exp_fac = {}
        for key, value in self.nuc.get_nuclides().items():
            exp_fac[key] = (
                value["z"] * mup_kt
                + (value["a"] - value["z"]) * mun_kt
                + self.fac[key]
            )

        for cluster in self.clusters:
            for nuc in self.cluster_nuclides[cluster]:
                exp_fac[nuc] += self.cluster_mus[cluster]

        result = {}
        for nuc in self.nuc.get_nuclides():
            result[nuc] = np.exp(self._check_fac(exp_fac[nuc]))

        return result

    def _compute_a_root(self, x):

        if self.ye:
            res_bracket = elementwise.bracket_root(
                self._compute_z_root, -10, args=(x,)
            )
            res_root = elementwise.find_root(
                self._compute_z_root, res_bracket.bracket, args=(x,)
            )
            self.mup_kt = res_root.x

        y = self._compute_abundances(self.mup_kt, x)

        result = 1.0
        for key, value in self.nuc.get_nuclides().items():
            result -= value["a"] * y[key]

        print(x, result)

        return result

    def _compute_z_root(self, x, mun_kt):

        y = self._compute_abundances(x, mun_kt)

        result = self.ye
        for key, value in self.nuc.get_nuclides().items():
            result -= value["z"] * y[key]

        return result
