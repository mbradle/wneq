"""This module computes general constrained equilibria from
`webnucleo <https://webnucleo.readthedocs.io>`_ files."""

import sys
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

    def compute(self, t9, rho, ye=None, clusters=None):
        """Method to compute a nuclear equilibrium.

        Args:
            ``t9`` (:obj:`float`): The temperature (in 10 :sup:`9` Kelvin)
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
        self._update_fac(t9, rho)

        self.clusters.clear()
        if clusters:
            self._set_clusters(clusters)

        self._set_initial_guesses()

        x0 = self._get_initial_multi_vector()

        sol = op.root(self._compute_multi_root, x0)

        if (
            not sol.success
            or np.linalg.norm(self._compute_multi_root(sol.x)) > 1.0e-4
        ):
            res_bracket = self._bracket_root(
                self._compute_a_root, self.guess.mu["n"]
            )
            assert res_bracket, "Root not bracketed."
            res_root = op.root_scalar(
                self._compute_a_root, bracket=res_bracket
            )

            x0[0] = self.mup_kt
            x0[1] = res_root.root
            for value in self.clusters.values():
                x0[value.index] = value.mu

            sol = op.root(self._compute_multi_root, x0)

            assert sol.success, "Root polishing failed."

        self.mup_kt = sol.x[0]
        self.mun_kt = sol.x[1]
        for value in self.clusters.values():
            value.mu = sol.x[value.index]

        props = self._set_base_properties(t9, rho)

        props["mun_kT"] = self.mun_kt
        props["mup_kT"] = self.mup_kt
        props["mun"] = wc.ergs_to_MeV * (self.mun_kt * (wc.k_B * t9 * 1.0e9))
        props["mup"] = wc.ergs_to_MeV * (self.mup_kt * (wc.k_B * t9 * 1.0e9))

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
            assert res_bracket, "Root not bracketed."
            res_root = op.root_scalar(
                self._compute_z_root, bracket=res_bracket, args=(x,)
            )
            self.mup_kt = res_root.root
        else:
            print("Feature not yet implemented.")
            sys.exit()

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
            assert res_bracket, "Root not bracketed."
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
        """Method to compute an equilibrium from input zone data.  The\
        resulting equilibrium is that to which the system would relax given\
        sufficient time.

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

        t9 = float(zone["properties"]["t9"])
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

        return self.compute(t9, rho, ye=ye, clusters=eq_clusters)

    def compute_low_temperature_nse(self, ye=None):
        """Method to compute a nuclear statistical equilibrium at low\
        temperature in a one- or two-species approximation.

        Args:
            ``ye`` (:obj:`float`, optional): The electron fraction at which\
            to compute the equilibrium.  If not supplied, the routine computes\
            the equilibrium without a fixed total neutron-to-proton ratio,\
            in which case, the equilibrium is computed in a one-species\
            approximation.

        Returns:
            A `wnutils <https://wnutils.readthedocs.io>`_ zone data dictionary\
            with the results of the calculation.

        """
        if ye is not None:
            ye_t = ye
            min_pair = self._find_min_pair(ye)
            y = self._compute_pair_abundances(min_pair, ye_t)
        else:
            species = self._find_min_species()
            y = {species: 1.0 / self.nuc.get_nuclides()[species]["a"]}
            ye_t = self.nuc.get_nuclides()[species]["z"] * y[species]
        props = {
            "note": "computed in 1- or 2-species, low-temperature approximation",
            "Ye": ye_t,
        }
        return self._make_equilibrium_zone(props, y)

    def _compute_pair_ye_min(self, ye):
        if ye < 0 or ye > 1:
            return 1.0e6
        min_pair = self._find_min_pair(ye)
        y = self._compute_pair_abundances(min_pair, ye)
        return self._compute_mass_per_nucleon_for_pair(y)

    def _find_min_species(self):
        dm_min = np.inf
        result = ""
        for key, value in self.nuc.get_nuclides().items():
            dm = self.nuc.compute_atomic_mass(key) / value["a"]
            if dm < dm_min:
                dm_min = dm
                result = key
        return result

    def _find_min_pair(self, ye):
        nuc_list = list(self.nuc.get_nuclides().keys())
        dm_min = np.inf
        min_pair = (nuc_list[0], nuc_list[0])
        for i, first_nuc in enumerate(nuc_list):
            for j in range(i + 1, len(nuc_list)):
                y = self._compute_pair_abundances((first_nuc, nuc_list[j]), ye)
                dm = self._compute_mass_per_nucleon_for_pair(y)
                if dm < dm_min:
                    dm_min = dm
                    min_pair = (first_nuc, nuc_list[j])
        return min_pair

    def _compute_pair_abundances(self, pair, ye):
        nucs = self.nuc.get_nuclides()
        z_0 = nucs[pair[0]]["z"]
        a_0 = nucs[pair[0]]["a"]
        z_1 = nucs[pair[1]]["z"]
        a_1 = nucs[pair[1]]["a"]

        ye_0 = float(z_0) / float(a_0)
        ye_1 = float(z_1) / float(a_1)

        result = {}

        if ye_0 != ye_1:
            denom = ye_1 - ye_0
            result[pair[0]] = (1.0 / a_0) * (ye_1 - ye) / denom
            result[pair[1]] = (1.0 / a_1) * (ye - ye_0) / denom
            return result
        if ye_0 == ye:
            result[pair[0]] = 1.0 / a_0
            result[pair[1]] = 0.0
            return result
        return {pair[0]: -1.0, pair[1]: 1.0}

    def _compute_mass_per_nucleon_for_pair(self, y):
        for value in y.values():
            if value < 0:
                return np.inf

        result = 0
        for key, value in y.items():
            result += value * self.nuc.compute_atomic_mass(key)

        return result
