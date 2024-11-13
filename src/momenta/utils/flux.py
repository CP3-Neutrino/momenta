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
from scipy.interpolate import interp1d
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

    def __init__(self, emin, emax, datafile):
        super().__init__(emin=emin, emax=emax, store="exact")
        self.datafile = datafile

    def evaluate(self, energy):
        xdata, ydata = np.array(self.datafile[self.datafile.columns[0]]), np.array(self.datafile[self.datafile.columns[1]])
        xdata = xdata.astype(float)
        ydata = ydata.astype(float)
        interp = interp1d(xdata, ydata, kind='linear', fill_value="extrapolate")
        return np.where((self.emin <= energy) & (energy <= self.emax), interp(energy), 0)
    

class FixedPowerLaw(Component):

    def __init__(self, emin, emax, gamma=2, eref=1):
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
