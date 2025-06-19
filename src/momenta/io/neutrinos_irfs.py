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
import astropy.coordinates
import astropy.time
import healpy as hp
import numpy as np
from copy import deepcopy

from astropy.units import deg
from scipy.integrate import quad
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.stats import norm
from typing import Callable, Iterable

from momenta.utils.flux import Component


def angular_distance(ra1: np.ndarray, dec1: np.ndarray, ra2: np.ndarray, dec2: np.ndarray, degrees1=False, degrees2=False):
    if degrees1:
        r1, d1 = np.deg2rad(ra1), np.deg2rad(dec1)
    else:
        r1, d1 = ra1, dec1
    if degrees2:
        r2, d2 = np.deg2rad(ra2), np.deg2rad(dec2)
    else:
        r2, d2 = ra2, dec2
    s = np.cos(r1 - r2) * np.cos(d1) * np.cos(d2) + np.sin(d1) * np.sin(d2)
    return np.arccos(np.clip(s, -1, 1))


class EffectiveAreaBase:
    """Class to handle detector effective area for a given sample and neutrino flavour.
    This default class handles only energy-dependent effective area."""

    def __init__(self):
        self._acceptances = {}  # pre-computed acceptances
        self._scaling_factor = 1  # global scaling factor

    def evaluate(self, energy: float | np.ndarray, ipix: int, nside: int):
        """Evaluate the effective area for a given energy, pixel index, and skymap resolution."""
        return 0 * self._scaling_factor

    def _compute_acceptance(self, fluxcomponent: Component, ipix: int, nside: int):
        """Compute the acceptance integrating the effective area x flux in log scale between emin and emax.
        Only used internally by `compute_acceptance_map`, may be overriden in a inheriting class."""

        def func(x: float):
            return fluxcomponent.evaluate(np.exp(x)) * self.evaluate(np.exp(x), ipix, nside) * np.exp(x)

        return quad(func, np.log(fluxcomponent.emin), np.log(fluxcomponent.emax), limit=500)[0]

    def compute_acceptance_map(self, fluxcomponent: Component, nside: int):
        """Compute the acceptance map for a given flux component, iterating over all pixels."""
        return np.array([self._compute_acceptance(fluxcomponent, ipix, nside) for ipix in range(hp.nside2npix(nside))])

    def compute_acceptance_maps(self, fluxcomponents: list[Component], nside: int):
        """Compute the acceptance maps for a list of flux components, iterating over all components and pixels.
        May be overriden by a smarter implementation for specific cases where computation can be optimized."""
        return [self.compute_acceptance_map(c, nside) for c in fluxcomponents]

    def get_acceptance_map(self, fluxcomponent: Component, nside: int):
        """Get the acceptance for a given flux component.
        If the component `store` attribute is 'exact', it is retrieved from the `acceptances` dictionary (added there if not yet available).
        If the component `store` attribute is 'interpolate', the dictionary with the interpolation function (+inputs) is returned.
        """
        if fluxcomponent.store == "exact":
            if (str(fluxcomponent), nside) not in self._acceptances:
                self._acceptances[(str(fluxcomponent), nside)] = self.compute_acceptance_map(fluxcomponent, nside)
            return self._acceptances[(str(fluxcomponent), nside)]
        if fluxcomponent.store == "interpolate":
            if (str(fluxcomponent), nside) not in self._acceptances:
                accs = {str(c): a for c, a in zip(fluxcomponent.grid.flatten(), self.compute_acceptance_maps(fluxcomponent.grid.flatten(), nside))}

                def f(_c):
                    return accs[str(_c)]

                accs = np.vectorize(f, signature="()->(n)")(fluxcomponent.grid)
                grid = [*fluxcomponent.shapevar_grid, np.arange(hp.nside2npix(nside))]
                self._acceptances[(str(fluxcomponent), nside)] = RegularGridInterpolator(grid, accs)
            return self._acceptances[(str(fluxcomponent), nside)]
        return self.compute_acceptance_map(fluxcomponent, nside)

    def get_acceptance(self, fluxcomponent: Component, ipix: int, nside: int):
        """Get the acceptance"""
        if fluxcomponent.store == "exact":
            return self.get_acceptance_map(fluxcomponent, nside)[ipix]
        if fluxcomponent.store == "interpolate":
            return self.get_acceptance_map(fluxcomponent, nside)([*fluxcomponent.shapevar_values, ipix])
        return self._compute_acceptance(fluxcomponent, ipix, nside)

    def __mul__(self, factor: float):
        """Multiply effective area and all data members by factor."""
        # returning a copy
        new = deepcopy(self)
        # multiply scaling factor for future evaluation
        new._scaling_factor *= factor
        # multiply already stored acceptances
        new._acceptances = {key: val * factor for key, val in self._acceptances.items()}
        # FIXME this will not work with the ingredients_internal one
        return new

    def __rmul__(self, factor: float):
        return self.__mul__(factor)


class EffectiveAreaAllSky(EffectiveAreaBase):

    def __init__(self, csvfile: str | None = None):
        """Effective area depending only on energy.

        Args:
            csvfile (str, optional): CSV file to read effective area.
            Format per line: E/GeV,Aeff/m^2
        """
        super().__init__()
        self.func = None
        if csvfile:
            self.read_csv(csvfile)

    def read_csv(self, csvfile: str):
        x, y = np.loadtxt(csvfile, delimiter=",").T
        self.func = interp1d(x, y, bounds_error=False, fill_value=0)
        self._acceptances = {}  # replace if already precomputed

    def evaluate(self, energy: float | np.ndarray, ipix: int, nside: int):
        val = self.func(energy)
        return val * self._scaling_factor

    def compute_acceptance_map(self, fluxcomponent: Component, nside: int):
        acc = self._compute_acceptance(fluxcomponent, 0, nside) * np.ones(hp.nside2npix(nside))
        return acc


class EffectiveAreaDeclinationDep(EffectiveAreaBase):

    def __init__(self):
        super().__init__()
        self.mapping = {}

    def evaluate(self, energy: float | np.ndarray, ipix: int, nside: int):
        if self.func == None:
            raise RuntimeError("The function calculating the effective area is missing")
        if nside not in self.mapping:
            self.mapping[nside] = self.map_ipix_to_declination(nside)
        val = self.func(energy, self.mapping[nside][ipix])
        return val * self._scaling_factor

    def compute_acceptance_map(self, fluxcomponent: Component, nside: int):
        if nside not in self.mapping:
            self.mapping[nside] = self.map_ipix_to_declination(nside)
        acc = np.zeros(hp.nside2npix(nside))
        for dec, ipix in zip(*np.unique(self.mapping[nside], return_index=True)):
            acc[self.mapping[nside] == dec] = self._compute_acceptance(fluxcomponent, ipix, nside)
        return acc

    def map_ipix_to_declination(self, nside: int):
        ipix = np.arange(hp.nside2npix(nside))
        _, dec = hp.pix2ang(nside, ipix, lonlat=True)
        return dec


class EffectiveAreaAltitudeDep(EffectiveAreaBase):

    def __init__(self):
        super().__init__()
        self.func = None
        self.mapping = {}

    def read(self):
        bins_logenergy = ...  # shape (M,)
        bins_altitude = ...  # shape (N,)
        aeff = ...  # shape (M,N)
        self.func = RegularGridInterpolator((bins_logenergy, bins_altitude), aeff, bounds_error=False, fill_value=0)

    def set_location(self, time: astropy.time.Time, lat_deg: float, lon_deg: float):
        self.obstime = time
        self.location = astropy.coordinates.EarthLocation(lat=lat_deg * deg, lon=lon_deg * deg)
        self.mapping = {}

    def evaluate(self, energy: float | np.ndarray, ipix: int, nside: int):
        if nside not in self.mapping:
            self.mapping[nside] = self.map_ipix_to_altitude(nside)
        val = self.func((np.log10(energy), self.mapping[nside][ipix]))
        return val * self._scaling_factor

    def compute_acceptance_map(self, fluxcomponent: Component, nside: int):
        if nside not in self.mapping:
            self.mapping[nside] = self.map_ipix_to_altitude(nside)
        acc = np.zeros(hp.nside2npix(nside))
        for alt, ipix in zip(*np.unique(self.mapping[nside], return_index=True)):
            acc[self.mapping[nside] == alt] = self._compute_acceptance(fluxcomponent, ipix, nside)
        return acc

    def map_ipix_to_altitude(self, nside: int):
        ipix = np.arange(hp.nside2npix(nside))
        ra, dec = hp.pix2ang(nside, ipix, lonlat=True)
        coords_eq = astropy.coordinates.SkyCoord(ra=ra * deg, dec=dec * deg, frame="icrs")
        coords_loc = coords_eq.transform_to(astropy.coordinates.AltAz(obstime=self.obstime, location=self.location))
        return coords_loc.alt.deg


class PDFBase:

    @abc.abstractmethod
    def __call__(self, evt):
        pass


class PDFFluxDependent(PDFBase):

    def __init__(self):
        super().__init__()
        self._pdfs = {}

    @abc.abstractmethod
    def __call__(self, evt, fluxcomponent: Component):
        pass

    @abc.abstractmethod
    def compute_pdf(self, fluxcomponent: Component):
        pass

    def compute_pdfs(self, fluxcomponents: list[Component]):
        """Compute the PDFs for a list of flux components, iterating over all components.
        May be overriden by a smarter implementation for specific cases where computation can be optimized."""
        return [self.compute_pdf(c) for c in fluxcomponents]

    def get_pdf(self, fluxcomponent: Component):
        """Get the PDF for a given flux component.
        If the component `store` attribute is 'exact', it is retrieved from the `pdfs` dictionary (added there if not yet available).
        If the component `store` attribute is 'interpolate', the dictionary with the PDFs for all points in the grid is returned.
        """
        if fluxcomponent.store == "exact":
            if str(fluxcomponent) not in self._pdfs:
                self._pdfs[str(fluxcomponent)] = self.compute_pdf(fluxcomponent)
            return self._pdfs[str(fluxcomponent)]
        if fluxcomponent.store == "interpolate":
            if str(fluxcomponent) not in self._pdfs:
                pdfs = {str(c): p for c, p in zip(fluxcomponent.grid.flatten(), self.compute_pdfs(fluxcomponent.grid.flatten()))}

                def f(_c):
                    return pdfs[str(_c)]

                self._pdfs[str(fluxcomponent)] = np.vectorize(f)(fluxcomponent.grid)
            return self._pdfs[str(fluxcomponent)]
        return self.compute_pdf(fluxcomponent)

    def __call__(self, evt, fluxcomponent: Component, **kwargs):
        """Compute the value of the PDF for a given event, flux component and other arguments.
        If the component `store` attribute is 'exact':
            the PDF is retrieved and called directly.
        If the component `store` attribute is 'interpolate' and len(kwargs)==0 (only NuEvent object is used):
            the interpolator object is created/stored if needed and used.
        If the component `store` attribute is 'interpolate' and len(kwargs)>0:
            the interpolator object is created and used (not stored).
        """
        if fluxcomponent.store == "exact":
            return self.get_pdf(fluxcomponent)(evt, **kwargs)
        if fluxcomponent.store == "interpolate" and len(kwargs) == 0:
            pdfs = self.get_pdf(fluxcomponent)
            if (str(fluxcomponent), str(evt)) not in self._pdfs:

                def f(func):
                    return func(evt)

                self._pdfs[(str(fluxcomponent), str(evt))] = RegularGridInterpolator(fluxcomponent.shapevar_grid, np.vectorize(f)(pdfs))
            return self._pdfs[(str(fluxcomponent), str(evt))](fluxcomponent.shapevar_values)
        if fluxcomponent.store == "interpolate" and len(kwargs) > 0:
            pdfs = self.get_pdf(fluxcomponent)

            def f(func):
                return func(evt, **kwargs)

            return RegularGridInterpolator(fluxcomponent.shapevar_grid, np.vectorize(f)(pdfs))(fluxcomponent.shapevar_values)


class EnergySignal(PDFFluxDependent):
    """The standard energy signal PDF is a function f(ra,dec,E,flux)."""

    def __init__(self, func: Callable | None = None):
        super().__init__()
        self.func = func


class AngularSignal(PDFBase):
    """The standard angular signal PDF is a function f(ra,dec,ra[src],dec[src],E)."""

    def __init__(self, func: Callable | None = None):
        super().__init__()
        self.func = func

    def __call__(self, evt, ra_src: float, dec_src: float, degrees_evt: bool = False):
        dpsi = angular_distance(evt.ra, evt.dec, ra_src, dec_src, degrees1=degrees_evt, degrees2=True)
        return self.func(dpsi, evt.energy)


class EnergyBackground(PDFBase):
    """The standard energy background PDF is a function f(ra,dec,E)."""

    def __init__(self, func: Callable | None = None):
        super().__init__()
        self.func = func

    def __call__(self, evt):
        return self.func(evt.ra, evt.dec, evt.energy)


class AngularBackground(PDFBase):
    """The standard angular background PDF is a function f(ra,dec,E)."""

    def __init__(self, func: Callable | None = None):
        super().__init__()
        self.func = func

    def __call__(self, evt):
        return self.func(evt.ra, evt.dec, evt.energy)


class TimeSignal(PDFFluxDependent):
    """The standard time signal PDF is a function f(deltaT)."""

    def __init__(self, func: Callable = None):
        super().__init__()
        self.func = func

    def __call__(self, evt):
        return self.func(evt.dt)


class TimeBackground(PDFBase):
    """The standard time background PDF is an uniform function f(deltaT) = 1/timewindow."""

    def __init__(self, timewindow_length: float):
        super().__init__()
        self.timewindow_length = timewindow_length

    def __call__(self, evt):
        return 1 / self.timewindow_length


class VonMisesSignal(AngularSignal):
    """A common angular signal PDF is Von Mises distribution f = VM(dpsi, sigma)."""

    def __init__(self):
        super().__init__()

    def __call__(self, evt, ra_src: float, dec_src: float, degrees_evt: bool = False):
        dpsi = angular_distance(evt.ra, evt.dec, ra_src, dec_src, degrees1=degrees_evt, degrees2=True)
        if evt.sigma > np.radians(7):
            kappa = 1.0 / evt.sigma**2
            return kappa * np.exp(kappa * np.cos(dpsi)) / (4 * np.pi * np.sinh(kappa))
        else:
            return 1 / (2 * np.pi * evt.sigma**2) * np.exp(-((dpsi / evt.sigma) ** 2) / 2)


class IsotropicBackground(AngularBackground):
    """A very simple background PDF which is uniform on the sky."""

    def __init__(self):
        super().__init__()

    def __call__(self, evt):
        return 1 / (4 * np.pi)


class TimeBoxSignal(TimeSignal):
    """A common time signal PDF is 1/dt for t0 <= t < t0+dt and 0 otherwise."""

    def __init__(self, t0: float | None = None, sigma_t: float | None = None):
        super().__init__()
        self.t0 = t0
        self.sigma_t = sigma_t

    def __call__(self, evt, t0: float | None = None, sigma_t: float | None = None):
        t0 = self.t0 if t0 is None else t0
        sigma_t = self.sigma_t if sigma_t is None else sigma_t
        return 1 / sigma_t * ((evt.dt >= t0) & (evt.dt < t0 + sigma_t))


class TimeGausSignal(TimeSignal):
    """A commin time signal PDF is a normal distribution centered on t0."""

    def __init__(self, t0: float | None = None, sigma_t: float | None = None):
        super().__init__()
        self.t0 = t0
        self.sigma_t = sigma_t

    def __call__(self, evt, t0: float | None = None, sigma_t: float | None = None):
        t0 = self.t0 if t0 is None else t0
        sigma_t = self.sigma_t if sigma_t is None else sigma_t
        return norm.pdf(evt.dt, loc=t0, scale=sigma_t)
