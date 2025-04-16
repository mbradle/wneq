"""This module computes general constrained equilibria from
`webnucleo <https://webnucleo.readthedocs.io>`_ files."""

from dataclasses import dataclass
import scipy.optimize as op
import numpy as np
import wnnet.consts as wc
import wneq.base as wqb


@dataclass
class _Cluster:
    name: str
    constraint: float
    mu: float
    nuclides: list


class Equil(wqb.Base):
    """A class for handling constrained equilibria."""

    def __init__(self, nuc):
        wqb.Base.__init__(self, nuc)

        self.fac = {}
        self.clusters = {}
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

        res_bracket = self._bracket_root(self._compute_a_root, -10)
        res_root = op.root_scalar(self._compute_a_root, bracket=res_bracket)
        self.mun_kt = res_root.root

        props = self._set_base_properties(t_9, rho)

        props["mun_kT"] = self.mun_kt
        props["mup_kT"] = self.mup_kt
        props["mun"] = wc.ergs_to_MeV * (self.mun_kt * (wc.k_B * t_9 * 1.0e9))
        props["mup"] = wc.ergs_to_MeV * (self.mup_kt * (wc.k_B * t_9 * 1.0e9))

        for value in self.clusters.values():
            props[("cluster", value.name, "mu_kT")] = value.mu
            props[("cluster", value.name, "constraint")] = value.constraint

        y = self._compute_abundances(self.mup_kt, self.mun_kt)
        return self._make_equilibrium_zone(props, y)

    def _set_clusters(self, clusters):
        for key, value in clusters.items():
            self.clusters[key] = _Cluster(
                name=key,
                constraint=value,
                mu=0,
                nuclides=list(self.nuc.get_nuclides(nuc_xpath=key).keys()),
            )
        cluster_nuclide_set = set()
        for value in self.clusters.values():
            for nuc in value.nuclides:
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

        for value in self.clusters.values():
            for nuc in value.nuclides:
                exp_fac[nuc] += value.mu

        result = {}
        for nuc in self.nuc.get_nuclides():
            result[nuc] = np.exp(self._check_fac(exp_fac[nuc]))

        return result

    def _compute_a_root(self, x):

        if self.ye:
            res_bracket = self._bracket_root(
                self._compute_z_root, -10, args=(x,)
            )
            res_root = op.root_scalar(
                self._compute_z_root, bracket=res_bracket, args=(x,)
            )
            self.mup_kt = res_root.root

        y = self._compute_abundances(self.mup_kt, x)

        result = 1.0
        for key, value in self.nuc.get_nuclides().items():
            result -= value["a"] * y[key]

        return result

    def _compute_z_root(self, x, mun_kt):

        for key, value in self.clusters.items():
            res_bracket = self._bracket_root(
                self._compute_cluster_root, -10, args=(x, mun_kt, key)
            )
            res_root = op.root_scalar(
                self._compute_cluster_root,
                bracket=res_bracket,
                args=(x, mun_kt, key),
            )
            value.mu = res_root.root

        y = self._compute_abundances(x, mun_kt)

        result = self.ye
        for key, value in self.nuc.get_nuclides().items():
            result -= value["z"] * y[key]

        return result

    def _compute_cluster_root(self, x, mup_kt, mun_kt, cluster):

        self.clusters[cluster].mu = x
        y = self._compute_abundances(mup_kt, mun_kt)

        result = self.clusters[cluster].constraint
        for nuc in self.clusters[cluster].nuclides:
            result -= y[nuc]

        return result
