"""
    Example to handle Super-Kamiokande specific format.
    The effective areas are the ones published in https://doi.org/10.5281/zenodo.4724822.
    The expected background is extracted from https://doi.org/10.3847/1538-4357/ac0d5a.
    The observed number of events is arbitrarily fixed to 0 (real values for O3a GW events in  https://doi.org/10.3847/1538-4357/ac0d5a).

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
import h5py
from scipy.interpolate import RegularGridInterpolator

import momenta.utils.flux as flux
from momenta.io import GWDatabase, NuDetector, Parameters
from momenta.io.neutrinos_irfs import EffectiveAreaAltitudeDep
from momenta.io.neutrinos import BackgroundFixed
from momenta.stats.run import run_ultranest
from momenta.stats.constraints import get_limits


class EffectiveAreaSK(EffectiveAreaAltitudeDep):
    def __init__(self, filename: str, samplename: str, time: astropy.time.Time):
        super().__init__()
        self.read(filename, samplename)
        self.set_location(time, 36.425634, 137.310340)

    def read(self, filename: str, samplename: str):
        with h5py.File(filename, "r") as f:
            bins_logenergy = f["bins_logenergy"][:]
            bins_altitude = f["bins_altitude"][:]
            aeff = f[f"aeff_{samplename}"][:]
        self.func = RegularGridInterpolator((bins_logenergy, bins_altitude), aeff, bounds_error=False, fill_value=0)


def single_event(gwname: str, gwdbfile: str, det_results: dict, pars: Parameters):
    """Compute the limits for a given GW event and using the detector results stored in dictionary.

    The `det_results` dictionary should contain the following keys:
        - nobs: list of observed number of events (length = 4 [number of samples])
        - nbkg: list of expected number of events (length = 4 [number of samples])
        - effarea: path to the effective area file
    """

    database_gw = GWDatabase(gwdbfile)
    database_gw.set_parameters(pars)

    sk = NuDetector("examples/input_files/detector_superk.yaml")
    gw = database_gw.find_gw(gwname)

    sk.set_effective_areas([EffectiveAreaSK(det_results["effarea"], s.name, gw.utc) for s in sk.samples])
    bkg = [BackgroundFixed(b) for b in det_results["nbkg"]]
    sk.set_observations(det_results["nobs"], bkg)

    _, result = run_ultranest(sk, gw, pars)
    limits = get_limits(result, CL=0.90)
    print(f"90% upper limit on flux normalisation: {limits['fluxnorm0']:.2e} /GeV/cm²")


if __name__ == "__main__":

    parameters = Parameters("examples/input_files/config.yaml")
    parameters.set_flux(flux.FluxFixedPowerLaw(0.1, 1e6, 2))
    parameters.nside = 8

    gwdb = "examples/input_files/gw_catalogs/database_example.csv"
    detresults = {
        "nobs": [0, 0, 0],
        "nbkg": [0.112, 0.007, 0.016],
        "effarea": "examples/input_files/effarea_superk.h5",
    }
    single_event("GW190412", gwdb, detresults, parameters)
