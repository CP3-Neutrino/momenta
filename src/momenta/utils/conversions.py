"""
    Copyright (C) 2024  Mathieu Lamoureux

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import abc
import astropy.cosmology.units as acu
import astropy.units as u
import numpy as np

from astropy.constants import M_sun, c


second_to_day = 1 / 86400
solarmass_to_erg = (M_sun * c**2).to(u.erg).value


class JetModelBase(metaclass=abc.ABCMeta):
    """Abstract model for neutrino emission jetting."""

    def __init__(self, jet_opening: float):
        self.jet_opening = jet_opening

    @abc.abstractmethod
    def eiso_to_etot(self, viewing_angle: float):
        """Conversion to total energy to the equivalent isotropic energy."""
        pass

    @property
    def str_filename(self):
        return self.__repr__().lower().replace(",", "_").replace(" ", "")


class JetIsotropic(JetModelBase):
    """Isotropic emission of neutrinos."""

    def __init__(self):
        super().__init__(np.inf)

    def eiso_to_etot(self, viewing_angle: float) -> float:
        return 1

    def __repr__(self):
        return "Isotropic"


class JetVonMises(JetModelBase):
    """Emission in a Gaussian jet."""

    def __init__(self, jet_opening: float, with_counter: bool = False):
        super().__init__(jet_opening)
        self.with_counter = with_counter
        self.kappa = np.longdouble(1 / (self.jet_opening**2))

    def eiso_to_etot(self, viewing_angle: float) -> float:
        if np.isinf(self.jet_opening):
            return 1
        if self.with_counter:
            return np.sinh(self.kappa) / (self.kappa * np.cosh(self.kappa * np.cos(viewing_angle)))
        return np.sinh(self.kappa) / (self.kappa * np.exp(self.kappa * np.cos(viewing_angle)))

    def __repr__(self):
        return "VonMises,%.1f deg%s" % (np.rad2deg(self.jet_opening), ",w/counter" if self.with_counter else "")


class JetRectangular(JetModelBase):
    """Emission in a rectangular jet (constant inside a cone, zero elsewhere)."""

    def __init__(self, jet_opening: float, with_counter: bool = False):
        super().__init__(jet_opening)
        self.with_counter = with_counter

    def eiso_to_etot(self, viewing_angle: float) -> float:
        if not 0 < self.jet_opening < np.pi:
            return 1
        if self.with_counter:
            if viewing_angle <= self.jet_opening or viewing_angle >= np.pi - self.jet_opening:
                return 1 - np.cos(self.jet_opening)
            return 0
        if viewing_angle <= self.jet_opening:
            return (1 - np.cos(self.jet_opening)) / 2
        return 0

    def __repr__(self):
        return "Constant,%.1f deg%s" % (self.jet_opening, ",w/counter" if self.with_counter else "")


def list_jet_models() -> list[JetModelBase]:
    """List all available jet models, with a scanning in opening angles."""
    full_list = []
    full_list.append(JetIsotropic())
    for opening in range(5, 60 + 1, 5):
        for with_counter in (False, True):
            full_list.append(JetVonMises(opening, with_counter=with_counter))
            full_list.append(JetRectangular(opening, with_counter=with_counter))
    return full_list


def redshift_to_lumidistance(redshift: float):
    return (redshift * acu.redshift).to(u.Mpc, acu.redshift_distance(kind="luminosity")).value


def lumidistance_to_redshift(distance: float):
    return (distance * u.Mpc).to(acu.redshift, acu.redshift_distance(kind="luminosity")).value


def distance_scaling(distance: float, redshift: float | None = None):
    """Returns the factor to scale from flux [/GeV/cm^2] to isotropic energy [erg]"""
    f = 4 * np.pi
    f *= ((distance * u.Mpc).to(u.cm).value) ** 2  # distance in cm
    if redshift is not None:
        f *= 1 / (1 + redshift)
    else:
        f *= 1 / (1 + lumidistance_to_redshift(distance))
    f *= (1 * u.GeV).to(u.erg).value  # energy in erg
    return f
