"""This module contains base elements for the equilibrium classes."""

from dataclasses import dataclass


@dataclass
class _Guess:
    x0: float
    user: {}
    mu: {}


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
        self.guess = _Guess(-10.0, {}, {})
        self.clusters = {}

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

    def _update_fac(self, t9, rho):

        for nuc in self.nuc.get_nuclides():
            self.fac[nuc] = self.nuc.compute_nse_factor(nuc, t9, rho)

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

    def update_initial_guesses(self, guesses):
        """Method to update initial guesses for chemical potentials divided by kT.

        Args:
            ``guesses`` (:obj:`dict`, optional): A dictionary of the guesses.  The allowed keys\
            of the dictionary are \"n\" (for the neutrons), \"p\" (for the protons), or\
            XPath expressions defining the valid clusters for the equilibrium.\
            The value for each key should be a float giving the initial guess for the\
            chemical potential divided by kT for the species or cluster.

        Returns:
            On successful return, the initial guesses for the species or clusters defined by the
            input keys are updated to the corresponding values.  These guesses will then be
            applied in the next calculation of the equilibrium.

        """

        self.guess.user.clear()
        for key, value in guesses.items():
            self.guess.user[key] = float(value)

    def _set_initial_guesses(self):
        self.guess.mu.clear()
        self.guess.mu["n"] = self.guess.x0
        self.guess.mu["p"] = self.guess.x0
        for cluster in self.clusters:
            self.guess.mu[cluster] = self.guess.x0

        for key, value in self.guess.user.items():
            if key in self.guess.mu:
                self.guess.mu[key] = value

    def _set_base_properties(self, t9, rho):
        result = {}
        result["t9"] = t9
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
