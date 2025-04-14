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
