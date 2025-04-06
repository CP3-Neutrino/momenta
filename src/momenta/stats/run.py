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

import contextlib
import logging
import os

from momenta.io import NuDetectorBase, Transient, Parameters, Stack
from momenta.stats.model import ModelOneSource, ModelStacked

import ultranest


logger = logging.getLogger("ultranest")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.WARNING)


@contextlib.contextmanager
def redirect_stdout(dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr e.g.:
    with sys.stdout_redirected(sys.stderr, os.devnull):
        ...
    """
    try:
        old = os.dup(1), os.dup(2)
        dest_file = open(dest_filename, "w")
        os.dup2(dest_file.fileno(), 1)
        os.dup2(dest_file.fileno(), 1)
        yield
    finally:
        os.dup2(old[0], 1)
        os.dup2(old[1], 2)
        dest_file.close()


def run_ultranest(
    detector: NuDetectorBase,
    src: Transient,
    parameters: Parameters,
    precision_dlogz: float = 0.3,
    precision_dKL: float = 0.1,
) -> tuple[ModelOneSource, dict]:
    """Run the ultranest nested sampling algorithm on a given observation for a given source.

    Args:
        detector (NuDetectorBase): object containing the detection inputs (observed/expected number of events, IRFs...)
        src (Transient): object containing the source inputs (localisation, distance...)
        parameters (Parameters): parameters of the analysis
        precision_dlogz (float, optional): wanted precision on the evidence. Defaults to 0.3.
        precision_dKL (float, optional): wanted precision on Kullback-Leibler divergence (stability of posterior distribution). Defaults to 0.1.

    Returns:
        ModelOneSource: model used for the computation
        dict: results of the nested sampling
    """
    model = ModelOneSource(detector, src, parameters)
    sampler = ultranest.ReactiveNestedSampler(model.param_names, model.loglike, model.prior, vectorized=True)
    result = sampler.run(show_status=False, viz_callback=False, dlogz=precision_dlogz, dKL=precision_dKL)

    result["samples"] = {k: v for k, v in zip(model.param_names, result["samples"].transpose())}
    result["samples"].update(model.calculate_deterministics(result["samples"]))
    for k in list(result["samples"].keys()):
        if k.startswith("norm"):
            result["samples"].pop(k)
    
    result["weighted_samples"]["points"] = {k: v for k, v in zip(model.param_names, result["weighted_samples"]["points"].transpose())}
    result["weighted_samples"]["points"].update(model.calculate_deterministics(result["weighted_samples"]["points"]))
    for k in list(result["weighted_samples"]["points"].keys()):
        if k.startswith("norm"):
            result["weighted_samples"]["points"].pop(k)
    
    return model, result


def run_ultranest_stack(
    stack: Stack,
    parameters: Parameters,
    precision_dlogz: float = 0.3,
    precision_dKL: float = 0.1,
) -> tuple[ModelStacked, dict]:
    """Run the ultranest nested sampling algorithm for a stacked analysis of multiple sources.

    Args:
        stack (Stack): object containing all detection/source inputs
        parameters (Parameters): parameters of the analysis
        precision_dlogz (float, optional): wanted precision on the evidence. Defaults to 0.3.
        precision_dKL (float, optional): wanted precision on Kullback-Leibler divergence (stability of posterior distribution). Defaults to 0.1.

    Returns:
        ModelStacked: model used for the computation
        dict: results of the nested sampling
    """
    model = ModelStacked(stack, parameters)
    sampler = ultranest.ReactiveNestedSampler(model.param_names, model.loglike, model.prior, vectorized=True)
    result = sampler.run(show_status=False, viz_callback=False, dlogz=precision_dlogz, dKL=precision_dKL)

    result["samples"] = {k: v for k, v in zip(model.param_names, result["samples"].transpose())}
    result["samples"].update(model.calculate_deterministics(result["samples"]))
    for k in list(result["samples"].keys()):
        if k.startswith("norm"):
            result["samples"].pop(k)
    
    result["weighted_samples"]["points"] = {k: v for k, v in zip(model.param_names, result["weighted_samples"]["points"].transpose())}
    result["weighted_samples"]["points"].update(model.calculate_deterministics(result["weighted_samples"]["points"]))
    for k in list(result["weighted_samples"]["points"].keys()):
        if k.startswith("norm"):
            result["weighted_samples"]["points"].pop(k)
            
    return model, result
