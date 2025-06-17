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

import awkward as ak
import numpy as np
from scipy.stats import norm, poisson

from momenta.io import NuDetectorBase, Parameters, Stack, Transient


class ModelOneSource:
    """Ultranest posterior model for a single source and set of observations."""

    def __init__(self, detector: NuDetectorBase, src: Transient, parameters: Parameters):
        detector.validate() # detector is sufficiently described
        parameters.validate() # all needed parameters e.g. flux model
        self.nobs = np.array([s.nobserved for s in detector.samples])
        self.bkg = np.array([s.background for s in detector.samples])
        self.nsamples = detector.nsamples
        self.bkg_variations = parameters.apply_det_systematics
        self.acc_variations = parameters.apply_det_systematics and np.any(detector.error_acceptance != 0)
        if self.acc_variations:
            self.chol_cov_acc = np.linalg.cholesky(detector.error_acceptance + 1e-5 * np.identity(self.nsamples))
        self.detector = detector
        self.parameters = parameters
        self.flux = parameters.flux
        self.src = src
        self.toys_src = src.prepare_prior_samples(parameters.nside)
        self.ntoys_src = len(self.toys_src)
        self.priornorm_var = self.parameters.prior_normalisation_var
        self.priornorm_type = self.parameters.prior_normalisation_type
        self.priornorm_range = self.parameters.prior_normalisation_range

    def __repr__(self):
        params = []
        for attr in ["detector", "src", "parameters"]:
            val = getattr(self, attr)
            if val is not None:
                params.append(f"{attr}={val}")
        params_str = ", ".join(params)
        return f"ModelOneSource({params_str})"
    
    def __str__(self):
        return self.__repr__()

    @property
    def ndims(self):
        nd = self.flux.nparameters + 1  # flux (norms + shapes) + GW toy
        if self.bkg_variations:
            nd += self.nsamples  # background
        if self.acc_variations:
            nd += self.nsamples  # acceptance
        return nd

    @property
    def param_names(self):
        params = [f"norm{i}" for i in range(self.flux.ncomponents)]
        params += [f"flux{i}_{s}" for i, c in enumerate(self.flux.components) for s in c.shapevar_names]
        params += ["itoy"]
        if self.bkg_variations:
            params += [f"bkg{i}" for i in range(self.nsamples)]  # background
        if self.acc_variations:
            params += [f"facc{i}" for i in range(self.nsamples)]  # acceptance
        return params

    def prior_norm(self, cube):
        if self.priornorm_type == "flat-linear":
            return self.priornorm_range[0] + (self.priornorm_range[1] - self.priornorm_range[0]) * cube
        elif self.priornorm_type == "flat-log":
            return np.power(10, np.log10(self.priornorm_range[0]) + (np.log10(self.priornorm_range[1]) - np.log10(self.priornorm_range[0])) * cube)
        elif self.priornorm_type == "jeffreys":
            return self.priornorm_range[0] + (self.priornorm_range[1] - self.priornorm_range[0]) * cube

    def prior(self, cube):
        """Convert from unit hypercube to hyperparameter space following the prior distributions.
        
        Args:
            cube (np.ndarray): unit cube of dimension = (N, D) where N is the number of points to evaluate and D the number of dimensions
        
        Returns:
            np.ndarray: same dimension as input, but values in real parameter space
        """
        x = cube.copy()
        i = 0
        x[..., i : i + self.flux.ncomponents] = self.prior_norm(x[..., i : i + self.flux.ncomponents])
        i += self.flux.ncomponents
        x[..., i : i + self.flux.nshapevars] = self.flux.prior_transform(x[..., i : i + self.flux.nshapevars])
        i += self.flux.nshapevars
        x[..., i] = np.floor(self.ntoys_src * x[..., i])
        i += 1
        if self.bkg_variations:
            for j in range(self.nsamples):
                x[..., i + j] = self.bkg[j].prior_transform(x[..., i + j])
            i += self.nsamples
        if self.acc_variations:
            rvs = norm.ppf(x[..., i : i + self.nsamples])
            x[..., i : i + self.nsamples] = np.ones(self.nsamples) + np.dot(rvs, self.chol_cov_acc)
        return x

    def loglike(self, cube):
        """Compute the log-likelihood.
        
        Args:
            cube (np.ndarray): parameter hypercube dimension = (N, D) where N is the number of points to evaluate and D the number of dimensions
            
        Returns:
            np.ndarray: value of log-likelihood for the N points
        """
        npoints = cube.shape[0]
        # INPUTS
        # > normalisation parameters
        i = 0
        norms = cube[:, i : i + self.flux.ncomponents]  # dims = (npoints, ncompflux)
        i += self.flux.ncomponents
        # > flux shape parameters
        shapes = cube[:, i : i + self.flux.nshapevars]  # dims = (npoints, nshapes)
        i += self.flux.nshapevars
        # > source parameter
        itoys = np.floor(cube[:, i]).astype(int)
        toys = self.toys_src[itoys]
        i += 1
        # > background parameters
        if self.bkg_variations:
            nbkg = cube[:, i : i + self.nsamples]  # dims = (npoints, nsamples)
            i += self.nsamples
        else:
            nbkg = np.tile([b.nominal for b in self.bkg], (npoints, 1))
        # > acceptance variation parameters
        if self.acc_variations:
            facc = cube[:, i : i + self.nsamples]  # dims = (npoints, nsamples)
        else:
            facc = np.ones((npoints, self.nsamples))
        # > get proper flux norms
        if self.priornorm_var == "flux":
            fluxnorms = norms
        else:
            _distance_scaling = toys.distance_scaling if "distance_scaling" in toys.dtype.names else [np.nan]*len(toys)  # dims = (npoints,)
            _viewing_angle = toys.viewing_angle if "viewing_angle" in toys.dtype.names else [np.nan]*len(toys)  # dims = (npoints,)
            _energy_denom = toys.energy_denom if "energy_denom" in toys.dtype.names else [np.nan]*len(toys)  # dims = (npoints,)
            etot_to_flux = self.flux.etot_to_flux(_distance_scaling, _viewing_angle)  # dims = (ncompflux, npoints)
            if self.priornorm_var == "etot":
                fluxnorms = norms * etot_to_flux.T
            elif self.priornorm_var == "fnu":
                fluxnorms = norms * _energy_denom * etot_to_flux.T
        # ACCEPTANCE
        accs = np.zeros((npoints, self.flux.ncomponents, self.detector.nsamples))  # dims = (npoints, ncompflux, nsamples)
        for ipoint in range(npoints):
            ishape = 0
            for iflux, c in enumerate(self.flux.components):
                if c.nshapevars > 0:
                    c.set_shapevars(shapes[ipoint, ishape : ishape + c.nshapevars])
                    ishape += c.nshapevars
                for isample, s in enumerate(self.detector.samples):
                    accs[ipoint, iflux, isample] = s.effective_area.get_acceptance(c, toys[ipoint].ipix, self.parameters.nside)
        # LOG-LIKELIHOOD
        nsigs = facc[:, np.newaxis, :] * (fluxnorms[:, :, np.newaxis] * accs / 6)  # dims = (npoints, ncompflux, nsamples)
        nexps = nbkg + np.sum(nsigs, axis=1)  # dims = (npoints, nsamples)
        if self.parameters.likelihood_method == "poisson":
            loglkl = np.sum(-nexps + self.nobs * np.log(nexps), axis=1)  # dims = (npoints, )
        if self.parameters.likelihood_method == "pointsource":
            loglkl = np.sum(-nexps, axis=1)  # dims = (npoints, )
            for isample, s in enumerate(self.detector.samples):
                if s.events is None:
                    loglkl += self.nobs[isample] * np.log(nexps[:, isample])  # dims = (npoints, )
                    continue
                psigs = np.zeros((npoints, self.flux.ncomponents, s.nobserved))  # dims = (npoints, ncompflux, nevents)
                ishape = 0
                for iflux, c in enumerate(self.flux.components):
                    for ipoint in range(npoints):
                        if c.nshapevars > 0:
                            c.set_shapevars(shapes[ipoint, ishape : ishape + c.nshapevars])
                        for ievt, evt in enumerate(s.events):
                            psigs[ipoint, iflux, ievt] = s.compute_signal_probability(evt, c, toys[ipoint].ra, toys[ipoint].dec)
                    ishape += c.nshapevars
                pbkgs = np.zeros(s.nobserved)
                for ievt, evt in enumerate(s.events):
                    pbkgs[ievt] = s.compute_background_probability(evt)
                probs = nbkg[:, isample, np.newaxis] * pbkgs + np.sum(nsigs[:, :, isample, np.newaxis] * psigs, axis=1)
                loglkl += np.sum(np.log(probs), axis=1)
        return loglkl

    def calculate_deterministics(self, samples):
        """Calculate different deterministic quantities:
        - eiso: total energy emitted in neutrinos assuming isotropic emission [in erg]
        - etot: total energy emitted in neutrinos assuming model=parameters.jet and using `theta_jn` as jet orientation w.r.t. Earth [in erg]
        - fnu: ratio between total energy in neutrinos `etot` and radiated energy in GW using `radiated_energy` [no units]
        """
        det = {}
        itoys = samples["itoy"].astype(int)
        nsamples = len(itoys)
        distance_scaling = self.toys_src[itoys]["distance_scaling"] if "distance_scaling" in self.toys_src.dtype.names else np.nan*np.ones(nsamples)
        energy_scaling = self.toys_src[itoys]["energy_scaling"] if "energy_scaling" in self.toys_src.dtype.names else np.nan*np.ones(nsamples)
        viewing_angle = self.toys_src[itoys]["viewing_angle"] if "viewing_angle" in self.toys_src.dtype.names else np.nan*np.ones(nsamples)
        shapes = np.array([samples[f"flux{i}_{s}"] for i, c in enumerate(self.flux.components) for s in c.shapevar_names])
        if self.priornorm_var == "flux":
            fluxnorms = np.array([samples[f"norm{i}"] for i in range(self.flux.ncomponents)])
            for i in range(self.flux.ncomponents):
                det[f"fluxnorm{i}"] = fluxnorms[i]
            det["etot"] = np.empty(nsamples)
            det["fnu"] = np.empty(nsamples)
            for i in range(self.flux.ncomponents):
                det[f"etot{i}"] = np.empty(nsamples)
                det[f"fnu{i}"] = np.empty(nsamples)
            for isample in range(nsamples):
                self.flux.set_shapevars(shapes[:, isample] if len(shapes)>0 else [])
                _etot = fluxnorms[:, isample] * self.flux.flux_to_etot(distance_scaling[isample], viewing_angle[isample])
                _fnu = _etot / energy_scaling[isample]
                det["etot"][isample], det["fnu"][isample] = np.sum(_etot), np.sum(_fnu)
                for i in range(self.flux.ncomponents):
                    det[f"etot{i}"][isample] = _etot[i]
                    det[f"fnu{i}"][isample] = _fnu[i]
        elif self.priornorm_var == "etot":
            etotnorms = np.array([samples[f"norm{i}"] for i in range(self.flux.ncomponents)])
            for i in range(self.flux.ncomponents):
                det[f"etot{i}"] = etotnorms[i]
            det["etot"] = np.sum(etotnorms, axis=0)
            det["fnu"] = np.empty(nsamples)
            for i in range(self.flux.ncomponents):
                det[f"fluxnorm{i}"] = np.empty(nsamples)
                det[f"fnu{i}"] = np.empty(nsamples)
            for isample in range(nsamples):
                self.flux.set_shapevars(shapes[:, isample] if len(shapes)>0 else [])
                _fluxnorm = etotnorms[:, isample] * self.flux.etot_to_flux(distance_scaling[isample], viewing_angle[isample])
                _fnu = etotnorms[:, isample] / energy_scaling[isample]
                det["fnu"][isample] = np.sum(_fnu)
                for i in range(self.flux.ncomponents):
                    det[f"fluxnorm{i}"][isample] = _fluxnorm[i]
                    det[f"fnu{i}"][isample] = _fnu[i]
        elif self.priornorm_var == "fnu":
            fnunorms = np.array([samples[f"norm{i}"] for i in range(self.flux.ncomponents)])
            for i in range(self.flux.ncomponents):
                det[f"fnu{i}"] = fnunorms[i]
            det["fnu"] = np.sum(fnunorms, axis=0)
            det["etot"] = np.empty(nsamples)
            for i in range(self.flux.ncomponents):
                det[f"fluxnorm{i}"] = np.empty(nsamples)
                det[f"etot{i}"] = np.empty(nsamples)
            for isample in range(nsamples):
                self.flux.set_shapevars(shapes[:, isample] if len(shapes)>0 else [])
                _etot = fnunorms[:, isample] * energy_scaling[isample]
                _fluxnorm = _etot * self.flux.etot_to_flux(distance_scaling[isample], viewing_angle[isample])
                det["etot"][isample] = np.sum(_etot)
                for i in range(self.flux.ncomponents):
                    det[f"fluxnorm{i}"][isample] = _fluxnorm[i]
                    det[f"etot{i}"][isample] = _etot[i]
        return det


class ModelOneSource_BkgOnly:
    """Same model as `ModelOneSource` but only with the background (used for Bayes factor computation)."""

    def __init__(self, detector: NuDetectorBase, parameters: Parameters):
        self.nobs = np.array([s.nobserved for s in detector.samples])
        self.bkg = np.array([s.background for s in detector.samples])
        self.nsamples = detector.nsamples
        self.bkg_variations = parameters.apply_det_systematics
        self.detector = detector
        self.parameters = parameters
    
    def __repr__(self):
        params = []
        for attr in ["detector", "parameters"]:
            val = getattr(self, attr)
            if val is not None:
                params.append(f"{attr}={val}")
        params_str = ", ".join(params)
        return f"ModelOneSource_BkgOnly({params_str})"
    
    def __str__(self):
        return self.__repr__()
    
    @property
    def ndims(self):
        nd = 0
        if self.bkg_variations:
            nd += self.nsamples  # background
        return nd

    @property
    def param_names(self):
        params = []
        if self.bkg_variations:
            params += [f"bkg{i}" for i in range(self.nsamples)]  # background
        return params

    def prior(self, cube):
        x = cube.copy()
        if self.bkg_variations:
            for j in range(self.nsamples):
                x[j] = self.bkg[j].prior_transform(x[j])
        return x

    def loglike(self, cube):
        # Format input parameters
        if self.bkg_variations:
            nbkg = cube
        else:
            nbkg = [b.nominal for b in self.bkg]
        # Compute log-likelihood
        loglkl = np.sum(poisson.logpmf(self.nobs, nbkg))
        return loglkl


class ModelStacked:
    """Ultranest posterior model for a catalogue of sources and observations."""
    
    def __init__(self, obs: Stack, parameters: Parameters):
        self.nobs, self.bkg, self.nsamples = obs.get_neutrino_data()
        self.bkg_variations = parameters.apply_det_systematics
        self.acc_variations = parameters.apply_det_systematics and np.any([np.any(e != 0) for e in obs.get_neutrino_error_acceptance()])
        if self.acc_variations:
            self.chol_cov_acc = []
            for i, err_acc in enumerate(obs.get_neutrino_error_acceptance()):
                if np.any(err_acc != 0):
                    self.chol_cov_acc.append(None)
                else:
                    self.chol_cov_acc.append(np.linalg.cholesky(err_acc + 1e-5 * np.identity(self.nsamples)))
        self.sources = list(obs.keys())
        self.detectors = list(obs.values())
        self.nsources = len(self.sources)
        self.parameters = parameters
        self.flux = parameters.flux
        self.toys_sources = [src.prepare_prior_samples(parameters.nside) for src in self.sources]
        self.ntoys_sources = [len(toys) for toys in self.toys_sources]
        self.priornorm_var = self.parameters.prior_normalisation_var
        self.priornorm_type = self.parameters.prior_normalisation_type
        self.priornorm_range = self.parameters.prior_normalisation_range
        self.validate()
        
    def validate(self):
        if self.priornorm_var == "flux":
            raise RuntimeError("[Model] Invalid variable used as normalisation, cannot be `flux` as it is source-dependent")

    @property
    def ndims(self):
        nd = self.flux.nparameters  # flux (norms + shapes)
        nd += self.nsources  # one index per source
        if self.bkg_variations:
            nd += int(np.sum(self.nsamples))  # background
        if self.acc_variations:
            nd += int(np.sum(self.nsamples))  # acceptance
        return nd

    @property
    def param_names(self):
        params = [f"norm{i}" for i in range(self.flux.ncomponents)]
        params += [f"flux{i}_{s}" for i, c in enumerate(self.flux.components) for s in c.shapevar_names]
        params += [f"itoy_{src.name}" for src in self.sources]
        if self.bkg_variations:
            params += [f"bkg{j}_{src.name}" for i, src in enumerate(self.sources) for j in range(self.nsamples[i])]  # background
        if self.acc_variations:
            params += [f"facc{j}_{src.name}" for i, src in enumerate(self.sources) for j in range(self.nsamples[i])]  # acceptance
        return params

    def prior_norm(self, cube):
        if self.priornorm_type == "flat-linear":
            return self.priornorm_range[0] + (self.priornorm_range[1] - self.priornorm_range[0]) * cube
        elif self.priornorm_type == "flat-log":
            return np.power(10, np.log10(self.priornorm_range[0]) + (np.log10(self.priornorm_range[1]) - np.log10(self.priornorm_range[0])) * cube)
        elif self.priornorm_type == "jeffreys":
            return self.priornorm_range[0] + (self.priornorm_range[1] - self.priornorm_range[0]) * cube

    def prior(self, cube):
        """Convert from unit hypercube to hyperparameter space following the prior distributions.
        
        Args:
            cube (np.ndarray): unit cube of dimension = (N, D) where N is the number of points to evaluate and D the number of dimensions
        
        Returns:
            np.ndarray: same dimension as input, but values in real parameter space
        """
        x = cube.copy()
        i = 0
        x[..., i : i + self.flux.ncomponents] = self.prior_norm(x[..., i : i + self.flux.ncomponents])
        i += self.flux.ncomponents
        x[..., i : i + self.flux.nshapevars] = self.flux.prior_transform(x[..., i : i + self.flux.nshapevars])
        i += self.flux.nshapevars
        for j in range(self.nsources):
            x[..., i+j] = np.floor(self.ntoys_sources[j] * x[..., i+j])
        i += self.nsources
        if self.bkg_variations:
            for j in range(self.nsources):
                for k in range(self.nsamples):
                    x[..., i + k] = self.bkg[i][k].prior_transform(x[..., i + k])
                i += self.nsamples[j]
        if self.acc_variations:
            for j in range(self.nsources):
                if self.chol_cov_acc is None:
                    x[..., i : i + self.nsamples[j]] = np.ones(self.nsamples[j])
                else:
                    rvs = norm.ppf(x[..., i : i + self.nsamples[j]])
                    x[..., i : i + self.nsamples[j]] = np.ones(self.nsamples[j]) + np.dot(rvs, self.chol_cov_acc[j])
                i += self.nsamples[j]
        return x

    def loglike(self, cube):
        """Compute the log-likelihood.
        
        Args:
            cube (np.ndarray): parameter hypercube dimension = (N, D) where N is the number of points to evaluate and D the number of dimensions
            
        Returns:
            np.ndarray: value of log-likelihood for the N points
        """
        npoints = cube.shape[0]
        # INPUTS
        # > normalisation parameters
        i = 0
        norms = cube[:, i : i + self.flux.ncomponents]  # dims = (npoints, ncompflux)
        i += self.flux.ncomponents
        # > flux shape parameters
        shapes = cube[:, i : i + self.flux.nshapevars]  # dims = (npoints, nshapes)
        i += self.flux.nshapevars
        # > source parameter
        itoys = np.floor(cube[:, i : i + self.nsources]).astype(int)
        toys = [[self.toys_sources[k][itoys[j,k]] for k in range(self.nsources)] for j in range(npoints)]
        i += self.nsources
        # > background parameters
        nbkgs = np.empty((npoints, sum(self.nsamples)))  # dims = (npoints, nsamples)
        if self.bkg_variations:
            j = 0
            for k in range(self.nsources):
                nbkgs[:, j : j + self.nsamples[k]] = cube[:, i : i + self.nsamples[k]]
                i += self.nsamples[k]
                j += self.nsamples[k]
        else:
            j = 0
            for k in range(self.nsources):
                nbkgs[:, j : j + self.nsamples[k]] = np.tile([b.nominal for b in self.bkg[k]], (npoints, 1))
                i += self.nsamples[k]
                j += self.nsamples[k]
        # > acceptance variation parameters
        faccs = np.ones((npoints, sum(self.nsamples)))  # dims = (npoints, nsamples)
        if self.acc_variations:
            j = 0
            for k in range(self.nsources):
                faccs[j : j + self.nsamples[k]] = cube[:, i : i + self.nsamples[k]]
                i += self.nsamples[k]
                j += self.nsamples[k]
        # > get proper flux norms
        _distance_scaling = np.nan * np.ones((npoints, self.nsources))
        _viewing_angle = np.nan * np.ones((npoints, self.nsources))
        _energy_denom = np.nan * np.ones((npoints, self.nsources))
        for isource in range(self.nsources):
            if "distance_scaling" in self.toys_sources[isource].dtype.names:
                _distance_scaling[:, isource] = [toys[ipoint][isource].distance_scaling for ipoint in range(npoints)]
            if "viewing_angle" in self.toys_sources[isource].dtype.names:
                _viewing_angle[:, isource] = [toys[ipoint][isource].viewing_angle for ipoint in range(npoints)]
            if "energy_denom" in self.toys_sources[isource].dtype.names:
                _energy_denom[:, isource] = [toys[ipoint][isource].energy_denom for ipoint in range(npoints)]
        etot_to_flux = self.flux.etot_to_flux(_distance_scaling, _viewing_angle)  # dims = (ncompflux, npoints, nsources)
        if self.priornorm_var == "etot":
            _fluxnorms = norms[..., np.newaxis] * np.swapaxes(etot_to_flux, 0, 1)  # dims = (npoints, ncompflux, nsources)
        elif self.priornorm_var == "fnu":
            _fluxnorms = norms[..., np.newaxis] * _energy_denom * np.swapaxes(etot_to_flux, 0, 1)  # dims = (npoints, ncompflux, nsources)
        fluxnorms = np.zeros((npoints, self.flux.ncomponents, sum(self.nsamples)))
        i = 0
        for isrc in range(self.nsources):
            fluxnorms[:, :, i:i+self.nsamples[isrc]] = _fluxnorms[:, :, isrc, np.newaxis]
            i += self.nsamples[isrc]
        # ACCEPTANCE
        accs = np.zeros((npoints, self.flux.ncomponents, sum(self.nsamples)))  # dims = (npoints, ncompflux, sum(nsamples))
        for ipoint in range(npoints):
            ishape = 0
            for iflux, c in enumerate(self.flux.components):
                if c.nshapevars > 0:
                    c.set_shapevars(shapes[ipoint, ishape : ishape + c.nshapevars])
                    ishape += c.nshapevars
                i = 0
                for isource in range(self.nsources):
                    for isample, s in enumerate(self.detectors[isource].samples):
                        accs[ipoint, iflux, i+isample] = s.effective_area.get_acceptance(c, toys[ipoint][isource].ipix, self.parameters.nside)
                    i += self.nsamples[isource]
        # LOG-LIKELIHOOD
        nsigs = faccs[:, np.newaxis, :] * (fluxnorms * accs / 6)  # dims = (npoints, ncompflux, nsamples)
        nexps = nbkgs + np.sum(nsigs, axis=1)  # dims = (npoints, nsamples)
        if self.parameters.likelihood_method == "poisson":
            loglkl = np.sum(-nexps + ak.flatten(self.nobs).to_list() * np.log(nexps), axis=1)  # dims = (npoints, )
        if self.parameters.likelihood_method == "pointsource":
            loglkl = np.sum(-nexps, axis=1)  # dims = (npoints, )
            i = 0
            for isource in range(self.nsources):
                for isample, s in enumerate(self.detectors[isource].samples):
                    if s.events is None:
                        loglkl += self.nobs[isource][isample] * np.log(nexps[:, i+isample])  # dims = (npoints, )
                        continue
                    psigs = np.zeros((npoints, self.flux.ncomponents, s.nobserved))  # dims = (npoints, ncompflux, nevents)
                    ishape = 0
                    for iflux, c in enumerate(self.flux.components):
                        for ipoint in range(npoints):
                            if c.nshapevars > 0:
                                c.set_shapevars(shapes[ipoint, ishape : ishape + c.nshapevars])
                            for ievt, evt in enumerate(s.events):
                                psigs[ipoint, iflux, ievt] = s.compute_signal_probability(evt, c, toys[ipoint][isource].ra, toys[ipoint][isource].dec)
                        ishape += c.nshapevars
                    pbkgs = np.zeros(s.nobserved)
                    for ievt, evt in enumerate(s.events):
                        pbkgs[ievt] = s.compute_background_probability(evt)
                    probs = nbkgs[:, i+isample, np.newaxis] * pbkgs + np.sum(nsigs[:, :, i+isample, np.newaxis] * psigs, axis=1)
                    loglkl += np.sum(np.log(probs), axis=1)
                i += self.nsamples[isource]
        return loglkl
    
    
    def calculate_deterministics(self, samples):
        det = {}
        if self.priornorm_var == "etot":
            etotnorms = np.array([samples[f"norm{i}"] for i in range(self.flux.ncomponents)])
            for i in range(self.flux.ncomponents):
                det[f"etot{i}"] = etotnorms[i]
            det["etot"] = np.sum(etotnorms, axis=0)
        if self.priornorm_var == "fnu":
            fnunorms = np.array([samples[f"norm{i}"] for i in range(self.flux.ncomponents)])
            for i in range(self.flux.ncomponents):
                det[f"fnu{i}"] = fnunorms[i]
            det["fnu"] = np.sum(fnunorms, axis=0)
        return det
