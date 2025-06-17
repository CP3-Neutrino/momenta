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

import matplotlib
import numpy as np
import tempfile
import time

from collections import defaultdict

from momenta.io import GW, NuDetector, Parameters
from momenta.io.neutrinos import BackgroundGaussian
from momenta.io.neutrinos_irfs import EffectiveAreaAllSky
from momenta.stats.run import run_ultranest
from momenta.stats.constraints import get_limits, get_limits_with_uncertainties
import momenta.utils.conversions
import momenta.utils.flux as flux

matplotlib.use("Agg")


tmpdir = tempfile.mkdtemp()


def test_onesample(src, parameters):

    det_str = """
    name: TestDet
    samples: ["A"]
    errors:
        acceptance: 0.10
        acceptance_corr: 0
    """
    det_file = f"{tmpdir}/detector.yaml"
    with open(det_file, "w") as f:
        f.write(det_str)

    det = NuDetector(det_file)

    class EffAreaTest1(EffectiveAreaAllSky):
        def evaluate(self, energy, ipix, nside):
            return energy**2 * np.exp(-energy / 10000)

    det.set_effective_areas([EffAreaTest1()])
    det.set_observations([0], [BackgroundGaussian(0.5, 0.1)])

    results = defaultdict(list)
    uncertainties = defaultdict(list)
    times = defaultdict(lambda: 0)

    N = 20
    parameters.apply_det_systematics = False
    parameters.prior_normalisation = "flat-linear"
    for _ in range(N):
        t0 = time.time()
        _, result = run_ultranest(det, src, parameters)
        if "weighted_samples" in result:
            limits = get_limits_with_uncertainties(result)["fluxnorm0"]
            limit, unc = limits[0], limits[1]
        else:
            limit = get_limits(result)["fluxnorm0"]
            unc = np.nan
        results["flatlin"].append(limit)
        uncertainties["flatlin"].append(unc)
        times["flatlin"] += time.time() - t0

    N = 20
    parameters.apply_det_systematics = False
    parameters.prior_normalisation = "flat-log"
    for _ in range(N):
        t0 = time.time()
        _, result = run_ultranest(det, src, parameters)
        if "weighted_samples" in result:
            limits = get_limits_with_uncertainties(result)["fluxnorm0"]
            limit, unc = limits[0], limits[1]
        else:
            limit = get_limits(result["samples"])["fluxnorm0"]
            unc = np.nan
        results["flatlog"].append(limit)
        uncertainties["flatlog"].append(unc)
        times["flatlog"] += time.time() - t0

    N = 20
    parameters.apply_det_systematics = False
    parameters.prior_normalisation = "jeffreys"
    for _ in range(N):
        t0 = time.time()
        _, result = run_ultranest(det, src, parameters)
        if "weighted_samples" in result:
            limits = get_limits_with_uncertainties(result)["fluxnorm0"]
            limit, unc = limits[0], limits[1]
        else:
            limit = get_limits(result)["fluxnorm0"]
            unc = np.nan
        results["jeffreys"].append(limit)
        uncertainties["jeffreys"].append(unc)
        times["jeffreys"] += time.time() - t0

    # compute naive upper limits
    nside = 8
    best_ipix = np.argmax(gw.fits.get_skymap(nside))
    acc = EffAreaTest1()._compute_acceptance(parameters.flux.components[0], best_ipix, nside)
    print(f"Naive UL = {2.3 / (acc/6):.2e}")

    for k in results.keys():
        print(f"{k:25s} => {np.average(results[k]):.2e} ± {np.std(results[k]):.2e} ({np.average(uncertainties[k]):.2e}), TIME = {times[k]/N:.2f} s")


if __name__ == "__main__":

    config_str = """
    skymap_resolution: 8
    detector_systematics: 0

    analysis:
        likelihood: poisson
        prior_normalisation:
            variable: flux
            type: flat-linear
            range: 
                min: 1.0e-10
                max: 1.0e+10
    """
    config_file = f"{tmpdir}/config.yaml"
    with open(config_file, "w") as f:
        f.write(config_str)

    f = flux.FluxFixedPowerLaw(1, 1e6, 2, eref=1)
    f.components[0].set_jet(momenta.utils.conversions.JetVonMises(np.deg2rad(10)))
    parameters = Parameters(config_file)
    parameters.set_flux(f)
    gw = GW(
        name="GW190412",
        path_to_fits="examples/input_files/gw_catalogs/GW190412/GW190412_PublicationSamples.fits",
        path_to_samples="examples/input_files/gw_catalogs/GW190412/GW190412_subset.h5",
    )
    gw.set_parameters(parameters)

    test_onesample(gw, parameters)
