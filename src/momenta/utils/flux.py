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
import numpy as np
from functools import partial
from scipy.integrate import quad
from scipy.interpolate import interp1d, RegularGridInterpolator
import pandas as pd



class Component(abc.ABC):

    def __init__(self, emin, emax, store="exact"):
        self.emin = emin
        self.emax = emax
        self.store = store
        # fixed shape parameters
        self.shapefix_names = []
        self.shapefix_values = []
        # variable shape parameters
        self.shapevar_names = []
        self.shapevar_values = []
        self.shapevar_boundaries = []
        self.shapevar_grid = []

    def __str__(self):
        s = [f"{type(self).__name__}"]
        s.append(f"{self.emin:.1e}--{self.emax:.1e}")
        s.append(",".join([f"{n}={v}" for n, v in zip(self.shapefix_names, self.shapefix_values)]))
        s.append(",".join([f"{n}={':'.join([str(_v) for _v in v])}" for n, v in zip(self.shapevar_names, self.shapevar_boundaries)]))
        return "/".join([_s for _s in s if _s])

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self.__str__())

    def init_shapevars(self):
        self.shapevar_grid = [np.linspace(*s) for s in self.shapevar_boundaries]
        self.shapevar_values = [0.5 * (s[0] + s[1]) for s in self.shapevar_boundaries]

    @property
    def nshapevars(self):
        return len(self.shapevar_names)

    def set_shapevars(self, shapes):
        self.shapevar_values = shapes

    @abc.abstractmethod
    def evaluate(self, energy):
        return None

    def flux_to_eiso(self, distance_scaling):
        def f(x):
            return self.evaluate(np.exp(x)) * (np.exp(x)) ** 2

        integration = quad(f, np.log(self.emin), np.log(self.emax), limit=100)[0]
        return distance_scaling * integration

    def prior_transform(self, x):
        """Transform uniform parameters in [0, 1] to shape parameter space."""
        return x


class TabulatedData(Component):

    def __init__(self, emin, emax, datafile, alpha):
        """alpha unnecessary, just to help with the variable class"""
        super().__init__(emin=emin, emax=emax, store="exact")
        self.datafile = datafile
        self.shapefix_names = ["alpha"]
        self.shapefix_values = [alpha]


    def evaluate(self, energy):
        xdata, ydata = np.array(self.datafile[self.datafile.columns[0]]), np.array(self.datafile[self.datafile.columns[1]])
        xdata = xdata.astype(float)
        ydata = ydata.astype(float)
        interp = interp1d(xdata, ydata, kind='linear', fill_value="extrapolate")
        return np.where((self.emin <= energy) & (energy <= self.emax), interp(energy), 0)
    
class TemporaryData(Component):
    def __init__(self, emin, emax, alpha):
        super().__init__(emin, emax, store = 'exact')
        self.shapefix_names = ["alpha"]
        self.shapefix_values = [alpha]


class VariableTabulated(Component):
    def __init__(self, emin, emax, df_fluxes, alpha_range):
        """

        Parameters:
        - energy_distributions: list of pandas DataFrames, each with two columns [energy, flux].
        - alphas: list of parameter values (alpha) corresponding to each energy distribution.
        """
        super().__init__(emin, emax, store='interpolate')
        if len(df_fluxes) != len(alpha_range):
            raise ValueError("Number of energy distributions must match number of alphas.")    
        self.shapevar_names = ['alpha'] 
        self.shapevar_boundaries = [(alpha_range[0], alpha_range[-1])]
        self.shapevar_grid = [alpha_range]
        self.shapevar_values = [0.5 * (alpha_range[0] + alpha_range[-1])]


        # Ensure data is sorted by alpha
        sorted_indices = np.argsort(alpha_range)
        self.alphas = np.array(alpha_range)[sorted_indices]
        self.energy_distributions = [df_fluxes[i] for i in sorted_indices] 

        #Shouldn't we allow for any alpha possible ?
        self.grid = np.array([TabulatedData(emin, emax, df, alpha) for df, alpha in zip(self.energy_distributions, self.alphas)])

        # Interpolate within each dataframe
        self.energy_range = self._validate_energy_ranges()
        self.energy_interpolators = [
            interp1d(
                df[df.columns[0]], df[df.columns[1]], kind='linear', bounds_error=False, fill_value=0
            ) for df in self.energy_distributions
        ]

        # Create a regular grid interpolator for energy and alpha
        self._initialize_interpolator()


    def _validate_energy_ranges(self):
        """
        Ensures all energy distributions have overlapping energy ranges.
        Returns the common energy range. /!\ may be interesting to allow to change logspace or npoints !
        
        """
        energy_ranges = [
            (df[df.columns[0]].min(), df[df.columns[0]].max()) for df in self.energy_distributions
        ]
        common_min = max(r[0] for r in energy_ranges)
        common_max = min(r[1] for r in energy_ranges)
        
        if common_min >= common_max:
            raise ValueError("No common energy range across all distributions.")
        
        return np.logspace(np.log10(common_min), np.log10(common_max), 500)  # Define a common grid

    def _initialize_interpolator(self):

        interpolated_fluxes = np.array([
            energy_interp(self.energy_range) for energy_interp in self.energy_interpolators
        ])
        self.energy_alpha_interpolator = RegularGridInterpolator((self.alphas, self.energy_range), interpolated_fluxes)

    def evaluate(self, energy):
        energy = np.array(energy)
        pts = [[self.shapevar_values[0], e] for e in energy]
        return self.energy_alpha_interpolator(pts)

    def evaluateBoth(self, energy, alpha):

        energy = np.array(energy)
        alpha = np.array(alpha)

        points = np.array([[a, e] for a in alpha for e in energy])
        flux = self.energy_alpha_interpolator(points).reshape(len(alpha), len(energy))
        
        if flux.shape[0] == 1:
            flux = flux.flatten()  # Return 1D array if single alpha is passed
        
        return flux

    def prior_transform(self, x):
        return self.alphas[0] + (self.alphas[-1] - self.alphas[0]) * x

class FixedPowerLaw(Component):

    def __init__(self, emin: float, emax, gamma=2, eref=1):
        super().__init__(emin=emin, emax=emax, store="exact")
        self.eref = eref
        self.shapefix_names = ["gamma"]
        self.shapefix_values = [gamma]

    def evaluate(self, energy):
        return np.where((self.emin <= energy) & (energy <= self.emax), np.power(energy / self.eref, -self.shapefix_values[0]), 0)


class VariablePowerLaw(Component):

    def __init__(self, emin, emax, gamma_range=(1, 4, 16), eref=1):
        super().__init__(emin=emin, emax=emax, store="interpolate")
        self.eref = eref
        self.shapevar_names = ["gamma"]
        self.shapevar_boundaries = [gamma_range]
        self.init_shapevars()
        self.grid = np.vectorize(partial(FixedPowerLaw, self.emin, self.emax, eref=self.eref))(self.shapevar_grid[0])

    def evaluate(self, energy):
        return np.where((self.emin <= energy) & (energy <= self.emax), np.power(energy / self.eref, -self.shapevar_values[0]), 0)

    def prior_transform(self, x):
        return self.shapevar_boundaries[0][0] + (self.shapevar_boundaries[0][1] - self.shapevar_boundaries[0][0]) * x


class FixedBrokenPowerLaw(Component):

    def __init__(self, emin, emax, gamma1=2, gamma2=2, log10ebreak=5, eref=1):
        super().__init__(emin=emin, emax=emax, store="exact")
        self.eref = eref
        self.shapefix_values = [gamma1, gamma2, log10ebreak]
        self.shapefix_names = ["gamma1", "gamma2", "log(ebreak)"]

    def evaluate(self, energy):
        factor = (10 ** self.shapefix_values[2] / self.eref) ** (self.shapefix_values[1] - self.shapefix_values[0])
        f = np.where(
            np.log10(energy) < self.shapefix_values[2],
            np.power(energy / self.eref, -self.shapefix_values[0]),
            factor * np.power(energy / self.eref, -self.shapefix_values[1]),
        )
        return np.where((self.emin <= energy) & (energy <= self.emax), f, 0)


class VariableBrokenPowerLaw(FixedBrokenPowerLaw):

    def __init__(self, emin, emax, gamma_range=(1, 4, 16), log10ebreak_range=(3, 6, 4), eref=1):
        super().__init__(emin=emin, emax=emax)
        self.store = "interpolate"
        self.eref = eref
        self.shapevar_names = ["gamma1", "gamma2", "log(ebreak)"]
        self.shapevar_boundaries = np.array([[*gamma_range], [*gamma_range], [*log10ebreak_range]])
        self.init_shapevars()
        self.grid = np.vectorize(partial(FixedBrokenPowerLaw, self.emin, self.emax, eref=self.eref))(*np.meshgrid(*self.shapevar_grid))

    def evaluate(self, energy):
        factor = (10 ** (self.shapevar_values[2]) / self.eref) ** (self.shapevar_values[1] - self.shapevar_values[0])
        f = np.where(
            np.log10(energy) < self.shapevar_values[2],
            np.power(energy / self.eref, -self.shapevar_values[0]),
            factor * np.power(energy / self.eref, -self.shapevar_values[1]),
        )
        return np.where((self.emin <= energy) & (energy <= self.emax), f, 0)

    def prior_transform(self, x):
        return self.shapevar_boundaries[:, 0] + (self.shapevar_boundaries[:, 1] - self.shapevar_boundaries[:, 0]) * x


class SemiVariableBrokenPowerLaw(FixedBrokenPowerLaw):

    def __init__(self, emin, emax, gamma1, gamma_range=(1, 4, 16), log10ebreak_range=(3, 6, 4), eref=1):
        super().__init__(emin=emin, emax=emax)
        self.store = "interpolate"
        self.eref = eref
        self.shapefix_names = ["gamma1"]
        self.shapefix_values = [gamma1]
        self.shapevar_names = ["gamma2", "log(ebreak)"]
        self.shapevar_boundaries = np.array([[*gamma_range], [*log10ebreak_range]])
        self.init_shapevars()
        self.grid = np.vectorize(partial(FixedBrokenPowerLaw, self.emin, self.emax, gamma1=gamma1, eref=self.eref))(*np.meshgrid(*self.shapevar_grid))

    def evaluate(self, energy):
        factor = (10 ** (self.shapevar_values[2]) / self.eref) ** (self.shapevar_values[1] - self.shapevar_values[0])
        f = np.where(
            np.log10(energy) < self.shapevar_values[2],
            np.power(energy / self.eref, -self.shapevar_values[0]),
            factor * np.power(energy / self.eref, -self.shapevar_values[1]),
        )
        return np.where((self.emin <= energy) & (energy <= self.emax), f, 0)

    def prior_transform(self, x):
        return self.shapevar_boundaries[:, 0] + (self.shapevar_boundaries[:, 1] - self.shapevar_boundaries[:, 0]) * x


class FluxBase(abc.ABC):

    def __init__(self):
        self.components = []

    def __str__(self):
        return " + ".join([str(c) for c in self.components])

    @property
    def ncomponents(self):
        return len(self.components)

    @property
    def nshapevars(self):
        return np.sum([c.nshapevars for c in self.components])

    @property
    def nparameters(self):
        return self.ncomponents + self.nshapevars

    @property
    def shapevar_positions(self):
        return np.cumsum([c.nshapevars for c in self.components]).astype(int)

    @property
    def shapevar_boundaries(self):
        return np.concatenate([c.shapevar_boundaries for c in self.components], axis=0)

    def set_shapevars(self, shapes):
        for c, i in zip(self.components, self.shapevar_positions):
            c.set_shapevars(shapes[i - c.nshapevars : i])

    def evaluate(self, energy):
        return [c.evaluate(energy) for c in self.components]

    def flux_to_eiso(self, distance_scaling):
        return np.array([c.flux_to_eiso(distance_scaling) for c in self.components])

    def prior_transform(self, x):
        return np.concatenate([c.prior_transform(x[..., i - c.nshapevars : i]) for c, i in zip(self.components, self.shapevar_positions)], axis=-1)


class FluxTabulatedData(FluxBase):

    def __init__(self, emin, emax, datafile):
        super().__init__()
        self.components = [TabulatedData(emin, emax, datafile)]

class FluxVariableTabulatedData(FluxBase):

    def __init__(self, emin, emax, datafile, alpha):
        super().__init__()
        self.components = [VariableTabulated(emin, emax, datafile, alpha)]

class FluxFixedPowerLaw(FluxBase):

    def __init__(self, emin, emax, gamma, eref=1):
        super().__init__()
        self.components = [FixedPowerLaw(emin, emax, gamma, eref)]


class FluxVariablePowerLaw(FluxBase):

    def __init__(self, emin, emax, gamma_range=(1, 4, 16), eref=1):
        super().__init__()
        self.components = [VariablePowerLaw(emin, emax, gamma_range, eref)]


class FluxVariableBrokenPowerLaw(FluxBase):
    def __init__(self, emin, emax, gamma_range=(1, 4, 16), log10ebreak_range=(3, 6, 7), eref=1):
        super().__init__()
        self.components = [VariableBrokenPowerLaw(emin, emax, gamma_range, log10ebreak_range, eref)]
