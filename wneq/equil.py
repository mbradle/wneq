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
    index: int
    nuclides: list


class Equil(wqb.Base):
    """A class for handling constrained equilibria."""

    def __init__(self, nuc):
        wqb.Base.__init__(self, nuc)

        self.mup_kt = 0

    def compute(self, t_9, rho, ye=None, clusters=None):
        """Method to compute a nuclear equilibrium.

        Args:
            ``t_9`` (:obj:`float`): The temperature (in 10 :sup:`9` Kelvin)
            at which to compute the equilibrium.

            ``rho`` (:obj:`float`): The mass density in grams per cc  at which
            to compute the equilibrium.

            ``ye`` (:obj:`float`, optional): The electron fraction at which to compute
            the equilibrium.  If not supplied, the routine computes the equilibrium
            without a fixed total neutron-to-proton ratio.

            ``clusters`` (:obj:`dict`, optional): A dictionary with the key for each
            entry giving the XPath describing the cluster and the value giving the
            abundance constraint for the cluster.

        Returns:
            A `wnutils <https://wnutils.readthedocs.io>`_ zone data dictionary
            with the results of the calculation.

        """

        self.ye = ye
        self._update_fac(t_9, rho)

        self.clusters.clear()
        if clusters:
            self._set_clusters(clusters)

        self._set_initial_guesses()

        x0 = self._get_initial_multi_vector()

        sol = op.root(self._compute_multi_root, x0)

        if sol.success:
            self.mup_kt = sol.x[0]
            self.mun_kt = sol.x[1]
            for value in self.clusters.values():
                value.mu = sol.x[value.index]
        else:
            res_bracket = self._bracket_root(
                self._compute_a_root, self.guess.mu["n"]
            )
            res_root = op.root_scalar(
                self._compute_a_root, bracket=res_bracket
            )
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
        i = 2
        for key, value in clusters.items():
            self.clusters[key] = _Cluster(
                name=key,
                constraint=value,
                mu=0,
                index=i,
                nuclides=list(self.nuc.get_nuclides(nuc_xpath=key).keys()),
            )
            i += 1
        cluster_nuclide_set = set()
        for value in self.clusters.values():
            for nuc in value.nuclides:
                assert nuc not in cluster_nuclide_set
                cluster_nuclide_set.add(nuc)

    def _get_initial_multi_vector(self):
        n_var = 2
        if self.clusters:
            n_var += len(self.clusters)

        x0 = np.full(n_var, self.guess.x0)

        x0[0] = self.guess.mu["p"]
        x0[1] = self.guess.mu["n"]
        for key, value in self.clusters.items():
            x0[value.index] = self.guess.mu[key]

        return x0

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

    def _compute_multi_root(self, x):
        result = np.zeros(len(x))
        for key, value in self.clusters.items():
            value.mu = x[value.index]
            result[value.index] = value.constraint

        y = self._compute_abundances(x[0], x[1])

        result[0] = self.ye
        result[1] = 1

        for key, value in self.nuc.get_nuclides().items():
            result[0] -= value["z"] * y[key]
            result[1] -= value["a"] * y[key]

        for value in self.clusters.values():
            for nuc in value.nuclides:
                result[value.index] -= y[nuc]

        return result

    def _compute_a_root(self, x):

        if self.ye:
            res_bracket = self._bracket_root(
                self._compute_z_root, self.guess.mu["p"], args=(x,)
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
                self._compute_cluster_root,
                self.guess.mu[key],
                args=(x, mun_kt, key),
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

    def compute_from_zone(self, zone, compute_ye=True, clusters=None):
        """Method to compute an equilibrium from input zone data.  The resulting
        equilibrium is that the system would relax to in the absence of
        charge-changing reactions and given sufficient time.

        Args:
            ``zone``: A `wnutils <https://wnutils.readthedocs.io>`_ zone
            data dictionary with the physical conditions and abundances
            from which to compute the equilibrium.

            ``compute_ye`` (:obj:`bool`, optional): A boolean to determine whether to
            compute the electron fraction in the zone and use it for the equilibrium calculation.

            ``clusters`` (:obj:`list`, optional): A list of XPath strings describing the desired
            clusters for the equilibrium.

        Returns:
            A `wnutils <https://wnutils.readthedocs.io>`_ zone data
            dictionary with the results of the calculation.

        """

        t_9 = float(zone["properties"]["t9"])
        rho = float(zone["properties"]["rho"])

        x_m = zone["mass fractions"]

        _y = {}

        ye = None
        if compute_ye:
            ye = 0

        for key, value in x_m.items():
            _y[key[0]] = value / key[2]
            if compute_ye:
                ye += key[1] * _y[key[0]]

        eq_clusters = None

        if clusters:
            eq_clusters = {}

            for cluster in clusters:
                y_c = 0
                for nuc in self.nuc.get_nuclides(nuc_xpath=cluster):
                    if nuc in _y:
                        y_c += _y[nuc]
                eq_clusters[cluster] = y_c

        return self.compute(t_9, rho, ye=ye, clusters=eq_clusters)
