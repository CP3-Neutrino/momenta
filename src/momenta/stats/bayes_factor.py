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

import copy
import logging
import numpy as np
import ultranest

from momenta.io import NuDetectorBase, Parameters, Transient
from momenta.io.neutrinos import BackgroundPoisson
from momenta.stats.model import ModelOneSource, ModelOneSource_BkgOnly


def build_minimal_experiment(detector: NuDetectorBase):
    """Build minimal experiment with 0 events in ON and OFF regions (can only work with ON/OFF bkg estimate)."""
    detector0 = copy.deepcopy(detector)
    for s in detector0.samples:
        if not isinstance(s.background, BackgroundPoisson):
            return None
        s.background.Noff = 0
        s.nobserved = 0
        s.events = []
    return detector0


def run_bkg(detector: NuDetectorBase, parameters: Parameters):
    """Compute Bayes evidence for background-only hypothesis."""
    model_bkg = ModelOneSource_BkgOnly(detector, parameters)
    sampler = ultranest.ReactiveNestedSampler(model_bkg.param_names, model_bkg.loglike, model_bkg.prior)
    result = sampler.run(show_status=False, viz_callback=False, dlogz=0.1)
    return result


def compute_correction_tobkg(detector: NuDetectorBase, src: Transient, parameters: Parameters, return_error: bool = False):
    """Compute correction factor to account for different dimensionalities of the background-only and bkg+signal models.
    This is computed using the "Arithmetic Intrinsic Bayes Factor" where minimal samples that cannot discriminate between the two models
    are used to estimate the correction. Other approaches may be implemented in the future.
    """

    detector0 = build_minimal_experiment(detector)
    if detector0 is None:
        logging.getLogger("momenta").warning("Cannot correct Bayes factor as one of the samples has non-Poisson background.")
        if return_error:
            return 0, 0
        return 0

    model0_bkg = ModelOneSource_BkgOnly(detector0, parameters)
    sampler0_bkg = ultranest.ReactiveNestedSampler(model0_bkg.param_names, model0_bkg.loglike, model0_bkg.prior)
    result0_bkg = sampler0_bkg.run(show_status=False, viz_callback=False)

    model0 = ModelOneSource(detector0, src, parameters)
    sampler0 = ultranest.ReactiveNestedSampler(model0.param_names, model0.loglike, model0.prior)
    result0 = sampler0.run(show_status=False, viz_callback=False)

    if return_error:
        return result0["logz"] - result0_bkg["logz"], np.sqrt(result0["logzerr"] ** 2 + result0_bkg["logzerr"] ** 2)
    return result0["logz"] - result0_bkg["logz"]


def compute_log_bayes_factor_tobkg(
    result: dict, detector: NuDetectorBase, src: Transient, parameters: Parameters, corrected: bool = True, return_error: bool = False
):
    """Compute Bayes factor Evidence[bkg+signal | data) / Evidence(bkg-only | data), applying or not the correction."""

    result_bkg = run_bkg(detector, parameters)
    logB = result["logz"] - result_bkg["logz"]
    if corrected:
        logB_corr, err_logB_corr = compute_correction_tobkg(detector, src, parameters, return_error=True)
    else:
        logB_corr = err_logB_corr = 0
    logB -= logB_corr

    if return_error:
        err_logB = np.sqrt(result["logzerr"] ** 2 + result_bkg["logzerr"] ** 2 + err_logB_corr**2)
        return logB, err_logB
    return logB
