"""This module computes (n,g)-(g,n) equilibrium from `webnucleo <https://webnucleo.readthedocs.io>`_ files."""

import wnnet.consts as wc
import wnnet.nuc as wn
import wnnet.zones as wz
import numpy as np
from scipy import optimize


class Ng:
    """A class for handling (n,g)-(g,n) equilibria.

    Args:
        ``file`` (:obj:`str`): A string giving the XML file name with the nuclide data.

        ``nuc_xpath`` (:obj:`str`, optional): An XPath expression to select nuclides.  Default is all nuclides.

        ``zone_xpath`` (:obj:`str`, optional): An XPath expression to select zones.  Default is all zones.
    """

    def __init__(self, file, nuc_xpath=""):
        self.nuc = wn.Nuc(file, nuc_xpath)
        self.zones_xml = wz.Zones_Xml(file)

    def get_nuclides(self, nuc_xpath=""):
        """Method to return a collection of nuclides.

        Args:
            ``nuc_xpath`` (:obj:`str`, optional): An XPath expression to select the nuclides.  Default is all species.

        Returns:
            A :obj:`dict` containing `wnutils <https://wnutils.readthedocs.io>`_ nuclides.

        """

        return self.nuc.get_nuclides(nuc_xpath=nuc_xpath)

    def get_zones(self, zone_xpath=""):
        """Method to return a collection of zones.

        Args:
            ``zone_xpath`` (:obj:`str`, optional): An XPath expression to select the zones.  Default is all zones

        Returns:
            A :obj:`dict` containing `wnutils <https://wnutils.readthedocs.io>`_ zones.

        """

        return self.zones_xml.get_zones(zone_xpath=zone_xpath)

    def _compute_ng(self, f, t9, rho, munpkT, yz):
        yzt = {}
        yl = {}
        ylm = {}
        mass_frac = {}
        props = {}

        for z in yz:
            ylm[z] = float("-inf")
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
        """Method to compute an (n,g)-(g,n) equilibrium.

        Args:
            ``t9`` (:obj:`float`): The temperature (in 10 :sup:`9` Kelvin) at which to compute the equilibrium.

            ``rho`` (:obj:`float`): The mass density in grams per cc  at which to compute the equilibrium.

            ``munp`` (:obj:`float`): The neutron chemical potential (in MeV) at which to compute the equilibrium..

            ``yz`` (:obj:`dict`): A dictionary with the elemental abundances for the calculation.  The keys of the dictionary are :obj:`int` giving the atomic numbr while the value is the abundance per nucleon for that atomic number.  On successful return, the equilibrium abundances will have the same elemental abundances as those given in *yz*.

        Returns:
            A `wnutils <https://wnutils.readthedocs.io>`_ zone object with the results of the calculation.

        """

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
        """Method to compute an (n,g)-(g,n) equilibrium.  The resulting equilibrium is that the system would relax to in the absence of charge-changing reactions and sufficient time.  The return result contains the neutron abundance and chemical potential for the appropriate equilibrium.

        Args:
            ``t9`` (:obj:`float`): The temperature (in 10 :sup:`9` Kelvin) at which to compute the equilibrium.

            ``rho`` (:obj:`float`): The mass density in grams per cc  at which to compute the equilibrium.

            ``yz`` (:obj:`dict`): A dictionary with the elemental abundances for the calculation.  The keys of the dictionary are :obj:`int` giving the atomic numbr while the value is the abundance per nucleon for that atomic number.  On successful return, the equilibrium abundances will have the save elemental abundances as those given in *yz*.

        Returns:
            A `wnutils <https://wnutils.readthedocs.io>`_ zone object with the results of the calculation.

        """

        f = self._compute_f(t9, rho)

        sol = optimize.root(self._root_func, [-10], args=(f, t9, rho, yz))

        result = self._compute_ng(f, t9, rho, sol.x[0], yz)

        return result

    def compute_with_root_from_zone(self, zone):
        """Method to compute an (n,g)-(g,n) equilibrium.  The resulting equilibrium is that the system would relax to in the absence of charge-changing reactions and sufficient time.  The return result contains the neutron abundance and chemical potential for the appropriate equilibrium.

        Args:
            ``zone``: A `wnutils <https://wnutils.readthedocs.io>`_ zone object with the physical conditions and abundances from which to compute the equilibrium.

        Returns:
            A `wnutils <https://wnutils.readthedocs.io>`_ zone object with the results of the calculation.

        """

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
