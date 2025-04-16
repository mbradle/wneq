"""This module contains base elements for the equilibrium classes."""


class Base:
    """A class for handling data for the equilibrium codes.

    Args:
        ``nuc``: A wnnet \
        `nuclear data <https://wnnet.readthedocs.io/en/latest/wnnet.html#module-wnnet.nuc>`_\
        object.


    """

    def __init__(self, nuc):
        self.nuc = nuc
        self.fac = {}
        self.mun_kt = 0
        self.ye = None

    def get_nuclides(self, nuc_xpath=""):
        """Method to return a collection of nuclides.

        Args:
            ``nuc_xpath`` (:obj:`str`, optional): An XPath expression to
            select the nuclides.  Default is all species.

        Returns:
            A :obj:`dict` containing\
            `wnutils <https://wnutils.readthedocs.io>`_ nuclides.

        """

        return self.nuc.get_nuclides(nuc_xpath=nuc_xpath)

    def _update_fac(self, t_9, rho):

        for nuc in self.nuc.get_nuclides():
            self.fac[nuc] = self.nuc.compute_nse_factor(nuc, t_9, rho)

    def _bracket_root(self, f, x0, args=()):
        factor = 1.6
        max_iter = 1000
        x1 = x0
        x2 = x1 + 1
        f1 = f(x1, *args)
        f2 = f(x2, *args)
        for _ in range(max_iter):
            if f1 * f2 < 0:
                return (x1, x2)
            if abs(f1) < abs(f2):
                x1 += factor * (x1 - x2)
                f1 = f(x1, *args)
            else:
                x2 += factor * (x2 - x1)
                f2 = f(x2, *args)
        return None

    def _set_base_properties(self, t_9, rho):
        result = {}
        result["t9"] = t_9
        result["rho"] = rho
        if self.ye:
            result["ye"] = self.ye
        return result

    def _make_equilibrium_zone(self, props, y):
        mass_fracs = {}
        nucs = self.nuc.get_nuclides()
        for key, value in y.items():
            if value > 0:
                nuc = nucs[key]
                mass_fracs[(key, nuc["z"], nuc["a"])] = nuc["a"] * value

        return {"properties": props, "mass fractions": mass_fracs}
