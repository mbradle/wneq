"""This module computes (n,g)-(g,n) equilibrium from
`webnucleo <https://webnucleo.readthedocs.io>`_ files."""

import scipy.optimize as op
import numpy as np
import wnnet as wn
import wneq.base as wqb


class Ng(wqb.Base):
    """A class for handling (n,g)-(g,n) equilibria."""

    def _compute_ng(self, t9, rho, mun_kt, y_z):
        y_zt = {}
        y_l = {}
        ylm = {}
        mass_frac = {}
        props = self._set_base_properties(t9, rho)

        for key in y_z.keys():
            ylm[key] = float("-inf")
            y_zt[key] = 0

        for key, value in self.nuc.get_nuclides().items():
            if value["z"] in y_z:
                y_t = self.fac[key] + value["a"] * mun_kt
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

        mass_frac[("n", 0, 1)] = np.exp(self.fac["n"] + mun_kt)

        props["mun_kt"] = str(mun_kt)

        props["mun"] = str(
            wn.consts.ergs_to_MeV * (mun_kt * (wn.consts.k_B * t9 * 1.0e9))
        )

        return {"properties": props, "mass fractions": mass_frac}

    def compute(self, t9, rho, mun, y_z):
        """Method to compute an (n,g)-(g,n) equilibrium.

        Args:
            ``t9`` (:obj:`float`): The temperature (in 10 :sup:`9` Kelvin)
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

        self._update_fac(t9, rho)

        mun_kt = mun * wn.consts.MeV_to_ergs / (wn.consts.k_B * t9 * 1.0e9)

        return self._compute_ng(t9, rho, mun_kt, y_z)

    def _root_func(self, x_var, t9, rho, y_z):

        result = 1

        n_g = self._compute_ng(t9, rho, x_var, y_z)

        x_m = n_g["mass fractions"]

        for x_t in x_m.values():
            result -= x_t

        return result

    def compute_with_root(self, t9, rho, y_z):
        """Method to compute an (n,g)-(g,n) equilibrium.  The resulting
        equilibrium is that the system would relax to in the absence of
        charge-changing reactions and given sufficient time.  The return
        result contains the neutron abundance and chemical potential for the
        appropriate equilibrium.

        Args:
            ``t9`` (:obj:`float`): The temperature (in 10 :sup:`9` Kelvin)
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

        self._update_fac(t9, rho)

        self._set_initial_guesses()

        res_bracket = self._bracket_root(
            self._root_func, self.guess.mu["n"], args=(t9, rho, y_z)
        )
        res_root = op.root_scalar(
            self._root_func, bracket=res_bracket, args=(t9, rho, y_z)
        )

        result = self._compute_ng(t9, rho, res_root.root, y_z)

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

        t9 = float(zone["properties"]["t9"])
        rho = float(zone["properties"]["rho"])

        self._update_fac(t9, rho)

        x_m = zone["mass fractions"]

        y_z = {}

        for key, value in x_m.items():
            if key[1] != 0:
                if key[1] in y_z:
                    y_z[key[1]] += value / key[2]
                else:
                    y_z[key[1]] = value / key[2]

        self._set_initial_guesses()

        res_bracket = self._bracket_root(
            self._root_func, self.guess.mu["n"], args=(t9, rho, y_z)
        )
        res_root = op.root_scalar(
            self._root_func, bracket=res_bracket, args=(t9, rho, y_z)
        )

        result = self._compute_ng(t9, rho, res_root.root, y_z)

        return result
