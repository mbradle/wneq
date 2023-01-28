"""This module computes (n,g)-(g,n) equilibrium from `webnucleo <https://webnucleo.readthedocs.io>`_ files."""

import wnnet.consts as wc
import wnnet.nuc as wn
import numpy as np
from scipy import optimize


class Ng:
    """A class for handling (n,g)-(g,n) equilibria.

    Args:
        ``file`` (:obj:`str`): A string giving the XML file name with the nuclide data.

        ``nuc_xpath`` (:obj:`str`, optional): An XPath expression to select nuclides.  Default is all nuclides.

    """

    def __init__(self, file, nuc_xpath=""):
        self.nuc = wn.Nuc(file, nuc_xpath)

    def get_nuclides(self, nuc_xpath=""):
        """Method to return a collection of nuclides.

        Args:
            ``nuc_xpath`` (:obj:`str`, optional): An XPath expression to select the nuclides.  Default is all species.

        Returns:
            A :obj:`dict` containing `wnutils <https://wnutils.readthedocs.io>`_ nuclides.

        """

        return self.nuc.get_nuclides(nuc_xpath=nuc_xpath)

    def _compute_ng(self, f, t9, rho, munpkT, yz):
        yzt = {}
        yl = {}
        ylm = {}
        mass_frac = {}
        props = {}

        for z in yz:
            ylm[z] = 0
            yzt[z] = 0

        nuclides = self.nuc.get_nuclides()

        for nuc in nuclides:
            z = nuclides[nuc]["z"]
            if z in yz:
                a = nuclides[nuc]["a"]
                yt = f[nuc] + a * munpkT
                if yt > ylm[z]:
                    ylm[z] = yt
                yl[(nuc, z, a)] = yt

        for t in yl:
            yl[t] -= ylm[t[1]]
            yzt[t[1]] += np.exp(yl[t])

        for z in yz:
            props[("muzkT", str(z))] = str(np.log(yz[z] / yzt[z]))

        for t in yl:
            s_z = str(t[1])
            mass_frac[t] = np.exp(yl[t] + float(props[("muzkT", s_z)])) * t[2]

        for z in yz:
            muzkT = float(props[("muzkT", str(z))])
            props[("muzkT", str(z))] = str(muzkT - ylm[z])

        mass_frac[("n", 0, 1)] = np.exp(f["n"] + munpkT)

        props["munpkT"] = str(munpkT)

        props["munp"] = str(wc.ergs_to_MeV * (munpkT * (wc.k_B * t9 * 1.0e9)))

        return {"properties": props, "mass fractions": mass_frac}

    def _compute_f(self, t9, rho):
        f = {}

        nuclides = self.nuc.get_nuclides()

        delta_n = nuclides["n"]["mass excess"]

        for nuc in nuclides:
            f[nuc] = np.log(self.nuc.compute_quantum_abundance(nuc, t9, rho)) + (
                nuclides[nuc]["a"] * delta_n - nuclides[nuc]["mass excess"]
            ) * wc.MeV_to_ergs / (wc.k_B * t9 * 1.0e9)

        return f

    def compute(self, t9, rho, munp, yz):

        f = self._compute_f(t9, rho)

        munpkT = munp * wc.MeV_to_ergs / (wc.k_B * t9 * 1.0e9)

        return self._compute_ng(f, t9, rho, munpkT, yz)

    def _root_func(self, x, f, t9, rho, yz):

        result = 1

        ng = self._compute_ng(f, t9, rho, x[0], yz)

        xm = ng["mass fractions"]

        for t in xm:
            result -= xm[t]

        return [result]

    def compute_with_root(self, t9, rho, yz):

        f = self._compute_f(t9, rho)

        sol = optimize.root(self._root_func, [-10], args=(f, t9, rho, yz))

        result = self._compute_ng(f, t9, rho, sol.x[0], yz)

        return result

    def compute_with_root_from_zone(self, zone):

        t9 = float(zone["properties"]["t9"])
        rho = float(zone["properties"]["rho"])

        f = self._compute_f(t9, rho)

        xm = zone["mass fractions"]

        yz = {}

        for t in xm:
            if t[1] != 0:
                if t[1] in yz:
                    yz[t[1]] += xm[t] / t[2]
                else:
                    yz[t[1]] = xm[t] / t[2]

        sol = optimize.root(self._root_func, [-10], args=(f, t9, rho, yz))

        result = self._compute_ng(f, t9, rho, sol.x[0], yz)

        return result
