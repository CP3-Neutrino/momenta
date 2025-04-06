import awkward as ak
import numpy as np

from momenta.io import Transient, NuDetectorBase


class Stack(dict):
    
    def __init__(self):
        pass
    
    def __setitem__(self, key, value):
        if not isinstance(key, Transient):
            raise RuntimeError("Key should be a Transient object (or inheriting from it).")
        if not isinstance(value, NuDetectorBase):
            raise RuntimeError("Value should be a NuDetectorBase object (or inheriting from it).")
        super().__setitem__(key, value)
    
    def list_source_names(self):
        return [src.name for src in self.keys()]

    def get_neutrino_data(self):
        nobs = ak.Array([[s.nobserved for s in det.samples] for det in self.values()])
        bkg = [[s.background for s in det.samples] for det in self.values()]
        nsamples = np.array([det.nsamples for det in self.values()])
        return nobs, bkg, nsamples
    
    def get_neutrino_error_acceptance(self):
        return [det.error_acceptance for det in self.values()]