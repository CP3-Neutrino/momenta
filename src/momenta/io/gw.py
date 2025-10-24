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

import astropy.time
from cached_property import cached_property
import h5py
import healpy as hp
from ligo.skymap.io import fits
import logging
import numpy as np
import os
import pandas as pd
from typing import Tuple

import momenta.utils.conversions
from momenta.io.transient import Transient


default_samples_priorities = [
    "PublicationSamples",
    "Overall_posterior",
    "C01:IMRPhenomXPHM",
    "IMRPhenomXPHM",
    "C01:IMRPhenomPv3HM",
    "IMRPhenomPv3HM",
    "C01:IMRPhenomPv2",
    "IMRPhenomPv2",
    "C01:IMRPhenomNSBH:HighSpin",
    "IMRPhenomNSBH:HighSpin",
    "C01:IMRPhenomNSBH:LowSpin",
    "IMRPhenomNSBH:LowSpin",
    "C01:Mixed",
    "Mixed",
]


class GW(Transient):
    """Class to handle a full GW event, including FITS file (with skymap) and HDF5 file (with full posterior samples)."""

    def __init__(self, path_to_fits: str | None = None, path_to_samples: str | None = None, name: str | None = None, logger: str = "momenta"):
        super().__init__(name=name, logger=logger)
        self.fits = None
        self.samples = None
        self.samples_priorities = None
        self.catalog = ""
        if path_to_fits is not None:
            self.set_fits(path_to_fits)
        if path_to_samples is not None:
            self.set_samples(path_to_samples)

    def set_fits(self, file: str):
        """Set GWFits object."""
        self.fits = _GWFits(file)
        self.log.info("[GW] Fits is loaded from the file %s", os.path.basename(file))
        self.utc = self.fits.utc

    def set_samples(self, file: str):
        """Set GWSamples object."""
        self.samples = _GWSamples(file)
        self.samples.priorities = self.samples_priorities
        self.log.info("[GW] Samples are loaded from the file %s", os.path.basename(file))

    def set_parameters(self, pars: "momenta.io.Parameters"):
        """Define the relevant parameters (to be propagated)."""
        self.samples_priorities = pars.gw_posteriorsamples_priorities
        if self.samples is not None:
            self.samples.priorities = self.samples_priorities

    def prepare_prior_samples(self, nside: int) -> pd.DataFrame:
        if self.samples:
            if self.samples_priorities is None:
                raise RuntimeError("[GW] Preparing toys using posterior samples require to call set_parameters() first.")
            self.samples.priorities = self.samples_priorities
            toys = pd.DataFrame(self.samples.prepare_toys(nside))
            toys["distance_scaling"] = momenta.utils.conversions.distance_scaling(
                toys.pop("luminosity_distance").to_numpy(), toys.pop("redshift").to_numpy()
            )
            return toys.to_records(index=False)
        if self.fits:
            return pd.DataFrame(data=self.fits.prepare_toys(nside)).to_records(index=False)
        self.log.warning("[GW] No toys are generated as this GW event has no FITS file nor posterior samples.")
        return pd.DataFrame().to_records(index=False)


class _GWFits:
    """Internal class to handle GW FITS files."""

    def __init__(self, file):
        assert os.path.isfile(file)
        self.file = file
        header = fits.read_sky_map(self.file, nest=False)[1]
        self.utc = astropy.time.Time(header["gps_time"], format="gps").utc

    def get_skymap(self, nside: int | None = None) -> np.ndarray:
        """Get the skymap from FITS file."""
        skymap = fits.read_sky_map(self.file, nest=False)[0]
        if nside is not None:
            skymap = hp.pixelfunc.ud_grade(skymap, nside)
            skymap *= 1 / np.sum(skymap)
        return skymap

    def get_ra_dec_bestfit(self, nside: int):
        """Get the direction with the maximum probability for given nside."""
        map = self.get_skymap(nside)
        return hp.pix2ang(nside, np.argmax(map), lonlat=True)

    def get_signal_region(self, nside: int, contained_prob: float = 0.90) -> np.ndarray:
        """Get the region containing a given probability of the skymap, for a given resolution."""
        skymap = self.get_skymap(nside)
        npix = hp.get_map_size(skymap)
        nside = hp.npix2nside(npix)
        if contained_prob is None:
            return np.arange(npix)
        if contained_prob == 0:  # signal region = only best-fit point
            return [np.argmax(skymap)]
        iSort = np.flipud(np.argsort(skymap))
        sortedCumulProba = np.cumsum(skymap[iSort])
        iSortMax = np.argwhere(sortedCumulProba > contained_prob)[0][0]
        pixReg = iSort[: iSortMax + 1]
        return pixReg

    def prepare_toys(self, nside) -> dict:
        """Prepare GW toys with an eventual restriction to the considered sky region."""
        skymap = self.get_skymap(nside)
        toys = {}
        toys["ipix"] = np.random.choice(len(skymap), size=1000, p=skymap / np.sum(skymap))
        toys["ra"], toys["dec"] = hp.pix2ang(nside, toys["ipix"], lonlat=True)
        return toys

    def get_area_region(self, contained_prob: float, degrees: bool = True):
        region = self.get_signal_region(nside=128, contained_prob=contained_prob)
        return len(region) * hp.nside2pixarea(128, degrees=degrees)


class _GWSamples:
    """Internal class to handle GW posterior samples."""

    def __init__(self, file):
        assert os.path.isfile(file)
        self.file = file
        self.sample_name = None
        self.mass1 = None
        self.mass2 = None
        self.priorities = None

    def find_correct_sample(self) -> str:
        """Find the correct posterior samples, take the first available one
        from the list in the global variable 'priorities' in gw.py.
        """
        if self.priorities is None:
            raise RuntimeError("Missing priority list to select correct GW samples.")
        f = h5py.File(self.file, "r")
        keys = list(f.keys())
        for sample in self.priorities:
            if sample in keys:
                # check that radiated_energy is there
                cols = f[sample]["posterior_samples"][:].dtype.names
                if "radiated_energy" in cols or "radiated_energy_non_evolved" in cols:
                    self.sample_name = sample
                    break
        f.close()
        if self.sample_name is None:
            raise RuntimeError(f"Did not find a correct sample in {self.file}")
        return self.sample_name

    def get_variables(self, *variables) -> dict:
        """Get posterior samples from the GW data release."""

        if self.sample_name is None:
            self.find_correct_sample()
        f = h5py.File(self.file, "r")

        variables_h5 = f[self.sample_name]["posterior_samples"][:].dtype.names
        variables_corrected = []
        for var in variables:
            if var not in variables_h5:
                if (var + "_non_evolved") in variables_h5:
                    variables_corrected.append(var + "_non_evolved")
                else:
                    raise RuntimeError(f"Missing variable {var} in h5 file.")
            else:
                variables_corrected.append(var)
        data = {}
        for var, varC in zip(variables, variables_corrected):
            data[var] = f[self.sample_name]["posterior_samples"][varC]
            if var in ("ra", "dec"):
                data[var] = np.rad2deg(data[var])  # convert from rad to deg
        f.close()

        return data

    @cached_property
    def masses(self) -> Tuple[float, float]:
        """Tuple of the two source masses."""
        masses = self.get_variables("mass_1_source", "mass_2_source")
        self.mass1 = np.mean(masses["mass_1_source"])
        self.mass2 = np.mean(masses["mass_2_source"])
        return self.mass1, self.mass2

    @cached_property
    def distance_5_50_95(self) -> float:
        """Mean luminosity distance based on posterior samples."""
        dist = self.get_variables("luminosity_distance")
        return np.percentile(dist["luminosity_distance"], [5, 50, 95])

    @cached_property
    def type(self) -> str:
        """Most probable type of GW event"""
        if self.mass1 is None or self.mass2 is None:
            self.masses
        if self.mass1 > 3:
            if self.mass2 > 3:
                return "BBH"
            return "NSBH"
        return "BNS"

    def prepare_toys(self, nside) -> dict:
        """Prepare GW toys from posterior samples."""
        variables_to_get = ["ra", "dec", "luminosity_distance", "redshift", "radiated_energy", "theta_jn"]
        toys = self.get_variables(*variables_to_get)
        toys["energy_scaling"] = toys.pop("radiated_energy") * momenta.utils.conversions.solarmass_to_erg
        toys["vieving_angle"] = toys.pop("theta_jn")
        toys["ipix"] = hp.ang2pix(nside, toys["ra"], toys["dec"], lonlat=True)
        return toys


class GWDatabase:
    """Class to handle a database containing a list of GW events."""

    def __init__(self, filepath: str | None = None, db: pd.DataFrame | None = None, name: str | None = None):
        self._filepath = filepath
        self.db = db
        self.name = name
        if filepath is not None and db is not None:
            logging.getLogger("momenta").warning("[GWDatabase] Will use the database and ignore the input filepath.")
        elif filepath is not None:
            if not os.path.isfile(filepath):
                logging.getLogger("momenta").warning("[GWDatabase] Input files does not exist, starting from empty database.")
            else:
                self.db = pd.read_csv(filepath, index_col=0)
            if self.name is None:
                self.name = os.path.splitext(os.path.basename(filepath))[0]
        self.samples_priorities = None

    def set_parameters(self, pars: "momenta.io.Parameters"):
        """Define the relevant parameters (to be propagated)."""
        self.samples_priorities = pars.gw_posteriorsamples_priorities

    def add_entry(self, name: str, h5path: str, fitspath: str):
        entry = {"h5_filepath": h5path, "fits_filepath": fitspath}
        newline = pd.DataFrame([entry], index=[name])
        if self.db is None:
            self.db = newline
        else:
            self.db = pd.concat([self.db, newline])

    def find_gw(self, name: str):
        if name not in self.db.index:
            raise RuntimeError("[GWDatabase] Missing index %s in the database." % name)
        gw = GW(name=name, path_to_samples=self.db.loc[name]["h5_filepath"], path_to_fits=self.db.loc[name]["fits_filepath"])
        gw.samples_priorities = self.samples_priorities
        gw.samples.priorities = self.samples_priorities
        if self.name is not None:
            gw.catalog = self.name
        return gw

    def list_all(self):
        return list(self.db.index)

    def list(self, gwtype: str | None = None, mindist: float | None = None, maxdist: float | None = None):
        selected_gw = []
        for idx, ev in self.db.iterrows():
            gw = GW(name=idx, path_to_fits=ev["fits_filepath"], path_to_samples=ev["h5_filepath"], logger="")
            gw.samples.priorities = self.samples_priorities
            selected = True
            if gwtype is not None and gw.samples.type != gwtype:
                selected = False
            if mindist is not None and gw.samples.distance_5_50_95[1] < mindist:
                selected = False
            if maxdist is not None and gw.samples.distance_5_50_95[1] > maxdist:
                selected = False
            if selected:
                selected_gw.append(idx)
        return selected_gw

    def save(self, filepath: str | None = None):
        """Save the database to specified CSV or, by default, to the one defined when initialising the Database."""
        outfile = None
        if filepath is not None:
            outfile = filepath
        elif self._filepath is not None:
            outfile = self._filepath
        else:
            raise RuntimeError("[GWDatabase] No output file was provided.")
        self.db.sort_index(inplace=True)
        self.db.to_csv(outfile)
