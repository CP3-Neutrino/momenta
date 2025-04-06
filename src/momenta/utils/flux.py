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
import warnings


from momenta.utils.conversions import JetModelBase, JetIsotropic


class Component(abc.ABC):

    def __init__(self, emin: float, emax: float, store="exact"):
        """ Generic class for the flux components. Several components may be added together
        in a FluxBase class.

        Args:
            emin (float): Lower energy bound of the component
            emax (float): Upper energy bound of the component
            store (str, optional): If the parameter is set to exact, the acceptance may be
            exactly computed with the given set of parameters. Otherwise, the acceptance
            will be computed on a parameter grid and later interpolated in the
            neutrinos_irfs.py file. Defaults to "exact".
        """
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
        # jet structure
        self.jet = JetIsotropic()

    def __str__(self):
        s = [f"{type(self).__name__}"]
        s.append(f"{self.emin:.1e}--{self.emax:.1e}")
        s.append(",".join([f"{n}={v}" for n, v in zip(self.shapefix_names, self.shapefix_values)]))
        s.append(",".join([f"{n}={':'.join([str(_v) for _v in v])}" for n, v in zip(self.shapevar_names, self.shapevar_boundaries)]))
        return "/".join([_s for _s in s if _s])

    # some cases like Jupyter we need a repr,
    # and default implementation is specific enough.
    def __repr__(self):
        return self.__str__()

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
        
    def set_jet(self, jet: JetModelBase):
        if not isinstance(jet, JetModelBase):
            raise TypeError(f"The provided jet model {jet} is not of the proper type (should inherit from JetModelBase).")
        self.jet = jet

    @abc.abstractmethod
    def evaluate(self, energy):
        return None

    def flux_to_eiso(self, distance_scaling: float):
        def f(x):
            return self.evaluate(np.exp(x)) * (np.exp(x)) ** 2

        integration = quad(f, np.log(self.emin), np.log(self.emax), limit=100)[0]
        return distance_scaling * integration

    def eiso_to_flux(self, distance_scaling: float):
        return 1/self.flux_to_eiso(distance_scaling)
    
    def eiso_to_etot(self, viewing_angle: float):
        return self.jet.eiso_to_etot(viewing_angle)
    
    def etot_to_eiso(self, viewing_angle: float):
        return 1/self.eiso_to_etot(viewing_angle)
    
    def flux_to_etot(self, distance_scaling: float, viewing_angle: float):
        return self.flux_to_eiso(distance_scaling) * self.eiso_to_etot(viewing_angle)
    
    def etot_to_flux(self, distance_scaling: float, viewing_angle: float):
        return self.etot_to_eiso(viewing_angle) * self.eiso_to_flux(distance_scaling)        
    
    def prior_transform(self, x):
        """Transform uniform parameters in [0, 1] to shape parameter space following prior.
        Default is uniform prior between 0 and 1."""
        return x


class TabulatedData(Component):

    def __init__(self, df_flux: pd.DataFrame, emin = None, emax = None, alpha = None, beta = None):
        """Custom tabulated flux with no shape parameter

        Args:
            df_flux (DataFrame): Dataframe with 2 columns: [energy, flux]
            alpha (float, optional): First shape parameter. Defaults to None.
            beta (float, optional): Second shape parameter. Defaults to None.
        """
   
        super().__init__(emin=emin, emax=emax, store="exact")
        energy_range = df_flux[df_flux.columns[0]]
        min_energy, max_energy = energy_range.min(), energy_range.max()
        self.emin = emin if emin is not None else min_energy
        self.emax = emax if emax is not None else max_energy
        if emin is not None and emin < min_energy:
            warnings.warn("Minimum energy is lower than the minimum tabulated energy")
        if emax is not None and emax > max_energy:
            warnings.warn("Maximum energy is higher than the maximum tabulated energy")   
        self.df = df_flux
        if alpha is not None:
            self.shapefix_names = ["alpha"] if beta is None else ["alpha", "beta"]
            self.shapefix_values = [alpha] if beta is None else [alpha, beta]


    def evaluate(self, energy):
        xdata, ydata = np.array(self.df[self.df.columns[0]]), np.array(self.df[self.df.columns[1]])
        xdata = xdata.astype(float)
        ydata = ydata.astype(float)
        interp = interp1d(xdata, ydata, kind='linear')
        return np.where((self.emin <= energy) & (energy <= self.emax), interp(energy), 0)
    


class VariableTabulated(Component):
    def __init__(self, df_fluxes: list[pd.DataFrame], alpha_grid, emin = None, emax = None):
        """Custom tabulated flux with one shape parameter

        Args:
            df_fluxes: list of pandas DataFrames, each with two columns [energy, flux].
            alpha_grid: list of parameter values corresponding to each energy distribution.
        """
        super().__init__(emin, emax, store='interpolate')
        if len(df_fluxes) != len(alpha_grid):
            raise ValueError("Number of energy distributions must match number of alphas.")    

        min_energy = min(df[df.columns[0]].min() for df in df_fluxes)
        max_energy = max(df[df.columns[0]].max() for df in df_fluxes)
        self.emin = emin if emin is not None else min_energy
        self.emax = emax if emax is not None else max_energy
        if emin is not None and emin < min_energy:
            warnings.warn("Minimum energy is lower than the minimum tabulated energy")
        if emax is not None and emax > max_energy:
            warnings.warn("Maximum energy is higher than the maximum tabulated energy")

        self.shapevar_names = ['alpha'] 
        #self.shapevar_boundaries = [(alpha_grid[0], alpha_grid[-1])] Useless
        self.shapevar_grid = [alpha_grid]
        self.shapevar_values = [0.5 * (alpha_grid[0] + alpha_grid[-1])]


        # Ensure data is sorted by alpha
        sorted_indices = np.argsort(alpha_grid)
        self.alphas = np.array(alpha_grid)[sorted_indices]
        self.energy_distributions = [df_fluxes[i] for i in sorted_indices] 

        #Shouldn't we allow for any alpha possible ? No, an interp is done in neutrino_irfs
        self.grid = np.array([TabulatedData(df_flux=df, emax=self.emax, emin=self.emin, alpha=alpha) for df, alpha in zip(self.energy_distributions, self.alphas)])

        # Interpolate within each dataframe
        self.energy_range = np.logspace(np.log10(self.emin), np.log10(self.emax), 1000)
        self.energy_interpolators = [
            interp1d(
                df[df.columns[0]], df[df.columns[1]], kind='linear'
            ) for df in self.energy_distributions
        ]

        # Create a regular grid interpolator for energy and alpha
        self._initialize_interpolator()


    def _initialize_interpolator(self):

        interpolated_fluxes = np.array([
            energy_interp(self.energy_range) for energy_interp in self.energy_interpolators
        ])
        self.energy_alpha_interpolator = RegularGridInterpolator((self.alphas, self.energy_range), interpolated_fluxes)

    def evaluate(self, energy):
        energy = np.array(energy)
        pts = [[self.shapevar_values[0], e] for e in energy]
        return self.energy_alpha_interpolator(pts)


    def prior_transform(self, x):
        return self.alphas[0] + (self.alphas[-1] - self.alphas[0]) * x
    
class VariableTabulated2Param(Component):
    def __init__(self, df_fluxes: list[list[pd.DataFrame]], alpha_grid: list[float], beta_grid: list[float], emin = None, emax = None):
        """Custom tabulated flux with two shape parameters

        Args:
            df_fluxes: 2D list of pandas DataFrames, where each element corresponds to a combination of alpha and beta.
            The DataFrames have two columns: 1st is the energy range and 2nd is the flux
            Each row corresponds to a new alpha, and different columns to different beta

            alpha_grid: List of alpha parameter values corresponding to the energy distributions.
            beta_grid: List of beta parameter values corresponding to the energy distributions.
        """
        super().__init__(emin, emax, store='interpolate')
        
        if len(df_fluxes) != len(alpha_grid) or any(len(row) != len(beta_grid) for row in df_fluxes):
            raise ValueError("Energy distributions must match the number of alphas and betas.")
        
        min_energy = min(df[df.columns[0]].min() for row in df_fluxes for df in row)
        max_energy = max(df[df.columns[0]].max() for row in df_fluxes for df in row)
        self.emin = emin if emin is not None else min_energy
        self.emax = emax if emax is not None else max_energy
        if emin is not None and emin < min_energy:
            warnings.warn("Minimum energy is lower than the minimum tabulated energy")
        if emax is not None and emax > max_energy:
            warnings.warn("Maximum energy is higher than the maximum tabulated energy")

        self.shapevar_names = ['alpha', 'beta']


        # Ensure data is sorted by alpha and beta
        sorted_alpha_indices = np.argsort(alpha_grid)
        sorted_beta_indices = np.argsort(beta_grid)
        
        self.alphas = np.array(alpha_grid)[sorted_alpha_indices]
        self.betas = np.array(beta_grid)[sorted_beta_indices]

        # Equivalent to the fct init_shapevars
        #self.shapevar_boundaries = [(alpha_grid[0], alpha_grid[-1]), (beta_grid[0], beta_grid[-1])] Useless too
        self.shapevar_grid = [self.alphas, self.betas]
        self.shapevar_values = [
            0.5 * (self.alphas[0] + self.alphas[-1]),
            0.5 * (self.betas[0] + self.betas[-1]),
        ]
        
        self.energy_distributions = [
            [df_fluxes[i][j] for j in sorted_beta_indices]
            for i in sorted_alpha_indices
        ]

        self.grid = np.array([
            [TabulatedData(df_flux=self.energy_distributions[i][j], emin=self.emin, emax=self.emax, alpha=self.alphas[i], beta=self.betas[j]) 
            for j in range(len(beta_grid))]
            for i in range(len(alpha_grid))
        ])

        # Interpolate within each dataframe
        self.energy_range = np.logspace(np.log10(self.emin), np.log10(self.emax), 1000)
        self.energy_interpolators = [
            [
                interp1d(
                    df[df.columns[0]], df[df.columns[1]], kind='linear'
                ) for df in row
            ] for row in self.energy_distributions
        ]

        # Create a regular grid interpolator for energy, alpha, and beta
        self._initialize_interpolator()

    def _initialize_interpolator(self):
        interpolated_fluxes = np.array([
            [
                energy_interp(self.energy_range) for energy_interp in beta_row
            ] for beta_row in self.energy_interpolators
        ])
        self.energy_alpha_beta_interpolator = RegularGridInterpolator(
            (self.alphas, self.betas, self.energy_range), interpolated_fluxes
        )

    def evaluate(self, energy):
        energy = np.array(energy)
        pts = [[self.shapevar_values[0], self.shapevar_values[1], e] for e in energy]
        return self.energy_alpha_beta_interpolator(pts)


    def prior_transform(self, x):
        """Transforms the 0-1 default parameter range to the actual prior range

        Args:
            x (ndarray): hypercube of dimension = (N, D) where N is the number of points to evaluate and D the number of shapevar

        Returns:
            ndarray: values in real parameter space
        """
        return np.array([self.alphas[0], self.betas[0]]) + (np.array([self.alphas[-1], self.betas[-1]]) - np.array([self.alphas[0], self.betas[0]])) * x




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
        self.grid = np.vectorize(partial(FixedBrokenPowerLaw, self.emin, self.emax, eref=self.eref))(*np.meshgrid(*self.shapevar_grid, indexing='ij'))

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

class AsymmetricBrokenPowerLaw(FixedBrokenPowerLaw):

    def __init__(self, emin, emax, gamma1_range=(1, 4, 16), gamma2_range = (0, 3, 10), log10ebreak_range=(3, 6, 4), eref=1):
        super().__init__(emin=emin, emax=emax)
        self.store = "interpolate"
        self.eref = eref
        self.shapevar_names = ["gamma1", "gamma2", "log(ebreak)"]
        self.shapevar_boundaries = np.array([[*gamma1_range], [*gamma2_range], [*log10ebreak_range]])
        self.init_shapevars()
        self.grid = np.vectorize(partial(FixedBrokenPowerLaw, self.emin, self.emax, eref=self.eref))(*np.meshgrid(*self.shapevar_grid, indexing='ij'))

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
        self.grid = np.vectorize(partial(FixedBrokenPowerLaw, self.emin, self.emax, gamma1=gamma1, eref=self.eref))(*np.meshgrid(*self.shapevar_grid, indexing='ij'))

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
        """Base class for the flux. It can be composed of different flux components
        stored in the components array.
        """
        self.components = []

    def __str__(self):
        return " + ".join([str(c) for c in self.components])
    
    def __repr__(self):
        return " + ".join([c.__repr__() for c in self.components])

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

    def flux_to_etot(self, distance_scaling: float, viewing_angle: float):
        return np.array([c.flux_to_etot(distance_scaling, viewing_angle) for c in self.components])
    
    def etot_to_flux(self, distance_scaling: float, viewing_angle: float):
        return 1 / self.flux_to_etot(distance_scaling, viewing_angle)

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
