"""This module computes (n,g)-(g,n) equilibrium from
`webnucleo <https://webnucleo.readthedocs.io>`_ files."""

import numpy as np
import wnnet as wn
from scipy import optimize


class Ng:
    """A class for handling (n,g)-(g,n) equilibria.

    Args:
        ``nuc``: A wnnet \
        `nuclear data <https://wnnet.readthedocs.io/en/latest/wnnet.html#module-wnnet.nuc>`_\
        object.


    """

    def __init__(self, nuc):
        self.nuc = nuc

    def get_nuclides(self, nuc_xpath=""):
        """Method to return a collection of nuclides.

        Args:
            ``nuc_xpath`` (:obj:`str`, optional): An XPath expression to
            select the nuclides.  Default is all species.

        Returns:
            A :obj:`dict` containing `wnutils <https://wnutils.readthedocs.io>`_ nuclides.

        """

        return self.nuc.get_nuclides(nuc_xpath=nuc_xpath)

    def _compute_ng(self, fac, t_9, mun_kt, y_z):
        y_zt = {}
        y_l = {}
        ylm = {}
        mass_frac = {}
        props = {}

        for key in y_z.keys():
            ylm[key] = float("-inf")
            y_zt[key] = 0

        for key, value in self.nuc.get_nuclides().items():
            if value["z"] in y_z:
                y_t = fac[key] + value["a"] * mun_kt
                if y_t > ylm[value["z"]]:
                    ylm[value["z"]] = y_t
                y_l[(key, value["z"], value["a"])] = y_t

        for key in y_l:
            y_l[key] -= ylm[key[1]]
            y_zt[key[1]] += np.exp(y_l[key])

        for key, value in y_z.items():
            props[("muz_kt", str(key))] = str(np.log(value / y_zt[key]))

        for key, value in y_l.items():
            s_z = str(key[1])
            mass_frac[key] = (
                np.exp(value + float(props[("muz_kt", s_z)])) * key[2]
            )

        for key in y_z.keys():
            muz_kt = float(props[("muz_kt", str(key))])
            props[("muz_kt", str(key))] = str(muz_kt - ylm[key])

        mass_frac[("n", 0, 1)] = np.exp(fac["n"] + mun_kt)

        props["mun_kt"] = str(mun_kt)

        props["mun"] = str(
            wn.consts.ergs_to_MeV * (mun_kt * (wn.consts.k_B * t_9 * 1.0e9))
        )

        return {"properties": props, "mass fractions": mass_frac}

    def _compute_fac(self, t_9, rho):
        fac = {}

        nuclides = self.nuc.get_nuclides()

        delta_n = nuclides["n"]["mass excess"]

        for nuc in nuclides:
            fac[nuc] = np.log(
                self.nuc.compute_quantum_abundance(nuc, t_9, rho)
            ) + (
                nuclides[nuc]["a"] * delta_n - nuclides[nuc]["mass excess"]
            ) * wn.consts.MeV_to_ergs / (
                wn.consts.k_B * t_9 * 1.0e9
            )

        return fac

    def compute(self, t_9, rho, mun, y_z):
        """Method to compute an (n,g)-(g,n) equilibrium.

        Args:
            ``t_9`` (:obj:`float`): The temperature (in 10 :sup:`9` Kelvin)
            at which to compute the equilibrium.

            ``rho`` (:obj:`float`): The mass density in grams per cc  at which
            to compute the equilibrium.

            ``mun`` (:obj:`float`): The neutron chemical potential (in MeV)
            at which to compute the equilibrium..

            ``y_z`` (:obj:`dict`): A dictionary with the elemental abundances
            for the calculation.  The keys of the dictionary are :obj:`int`
            giving the atomic number while the value is the abundance per
            nucleon for that atomic number.  On successful return,
            the equilibrium abundances will have the same elemental abundances
            as those given in *y_z*.

        Returns:
            A `wnutils <https://wnutils.readthedocs.io>`_ zone data dictionary
            with the results of the calculation.

        """

        fac = self._compute_fac(t_9, rho)

        mun_kt = mun * wn.consts.MeV_to_ergs / (wn.consts.k_B * t_9 * 1.0e9)

        return self._compute_ng(fac, t_9, mun_kt, y_z)

    def _root_func(self, x_var, fac, t_9, y_z):

        result = 1

        n_g = self._compute_ng(fac, t_9, x_var[0], y_z)

        x_m = n_g["mass fractions"]

        for x_t in x_m.values():
            result -= x_t

        return [result]

    def compute_with_root(self, t_9, rho, y_z):
        """Method to compute an (n,g)-(g,n) equilibrium.  The resulting
        equilibrium is that the system would relax to in the absence of
        charge-changing reactions and given sufficient time.  The return
        result contains the neutron abundance and chemical potential for the
        appropriate equilibrium.

        Args:
            ``t_9`` (:obj:`float`): The temperature (in 10 :sup:`9` Kelvin)
            at which to compute the equilibrium.

            ``rho`` (:obj:`float`): The mass density in grams per cc
            at which to compute the equilibrium.

            ``y_z`` (:obj:`dict`): A dictionary with the elemental abundances
            for the calculation.  The keys of the dictionary are :obj:`int`
            giving the atomic number while the value is the abundance per
            nucleon for that atomic number.  On successful return,
            the equilibrium abundances will have the save elemental abundances
            as those given in *y_z*.

        Returns:
            A `wnutils <https://wnutils.readthedocs.io>`_ zone data dictionary
            with the results of the calculation.

        """

        fac = self._compute_fac(t_9, rho)

        sol = optimize.root(self._root_func, [-10], args=(fac, t_9, y_z))

        result = self._compute_ng(fac, t_9, sol.x[0], y_z)

        return result

    def compute_with_root_from_zone(self, zone):
        """Method to compute an (n,g)-(g,n) equilibrium.  The resulting
        equilibrium is that the system would relax to in the absence of
        charge-changing reactions and given sufficient time.  The return
        result contains the neutron abundance and chemical potential for the
        appropriate equilibrium.

        Args:
            ``zone``: A `wnutils <https://wnutils.readthedocs.io>`_ zone
            data dictionary with the physical conditions and abundances
            from which to compute the equilibrium.

        Returns:
            A `wnutils <https://wnutils.readthedocs.io>`_ zone data
            dictionary with the results of the calculation.

        """

        t_9 = float(zone["properties"]["t9"])
        rho = float(zone["properties"]["rho"])

        fac = self._compute_fac(t_9, rho)

        x_m = zone["mass fractions"]

        y_z = {}

        for key, value in x_m.items():
            if key[1] != 0:
                if key[1] in y_z:
                    y_z[key[1]] += value / key[2]
                else:
                    y_z[key[1]] = value / key[2]

        sol = optimize.root(self._root_func, [-10], args=(fac, t_9, y_z))

        result = self._compute_ng(fac, t_9, sol.x[0], y_z)

        return result
