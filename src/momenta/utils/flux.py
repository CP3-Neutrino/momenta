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
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d, RegularGridInterpolator
import pandas as pd
import warnings


from momenta.utils.conversions import JetModelBase, JetIsotropic


class Component(abc.ABC):

    def __init__(self, emin: float, emax: float, store="exact"):
        """Generic class for the flux components. Several components may be added together
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
    def evaluate(self, energy: np.ndarray) -> np.ndarray:
        """Compute the flux value on a given energy array"""
        return None

    def flux_to_eiso(self, distance_scaling: float):
        x = np.linspace(np.log(self.emin), np.log(self.emax), 500)
        y = self.evaluate(np.exp(x)) * (np.exp(x)) ** 2
        integration = trapezoid(y, x)
        return distance_scaling * integration

    def eiso_to_flux(self, distance_scaling: float):
        return 1 / self.flux_to_eiso(distance_scaling)

    def eiso_to_etot(self, viewing_angle: float):
        return self.jet.eiso_to_etot(viewing_angle)

    def etot_to_eiso(self, viewing_angle: float):
        return 1 / self.eiso_to_etot(viewing_angle)

    def flux_to_etot(self, distance_scaling: float, viewing_angle: float):
        return self.flux_to_eiso(distance_scaling) * self.eiso_to_etot(viewing_angle)

    def etot_to_flux(self, distance_scaling: float, viewing_angle: float):
        return self.etot_to_eiso(viewing_angle) * self.eiso_to_flux(distance_scaling)

    def prior_transform(self, x):
        """Transforms the 0-1 default parameter range to the actual prior range

        Args:
            x (ndarray): hypercube of dimension = (N, D) where N is the number of points to evaluate and D the number of shapevar

        Returns:
            ndarray: values in real parameter space
        """
        return x


class FixedTabulated(Component):

    def __init__(self, df_flux: pd.DataFrame, emin: float = None, emax: float = None, alpha: float = None, beta: float = None):
        """Custom tabulated flux with fixed shape parameters

        Args:
            df_flux (DataFrame): Dataframe with 2 to 4 columns: [energy, flux, (alpha, beta)]
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
        interp = interp1d(xdata, ydata, kind="linear", bounds_error=False, fill_value=0)
        return np.where((self.emin <= energy) & (energy <= self.emax), interp(energy), 0)


class VariableTabulated1D(Component):
    def __init__(self, df_fluxes: pd.DataFrame, emin: float = None, emax: float = None):
        """Custom tabulated flux with one free shape parameter

        Args:
            df_fluxes: A DataFrame with three columns: 'energy', 'flux', 'shape_parameter'.
                For each 'shape_parameter', the corresponding 'energy'-'flux' values must be provided.
                Therefore, len(DataFrame) = len(shape_parameter) * len(energy)
        """
        super().__init__(emin, emax, store="interpolate")

        min_energy = min(df_fluxes["energy"])
        max_energy = max(df_fluxes["energy"])
        self.emin = emin if emin is not None else min_energy
        self.emax = emax if emax is not None else max_energy
        if emin is not None and emin < min_energy:
            warnings.warn("Minimum energy is lower than the minimum tabulated energy")
        if emax is not None and emax > max_energy:
            warnings.warn("Maximum energy is higher than the maximum tabulated energy")

        param = df_fluxes.columns[2]
        self.shapevar_names = [param]
        self.alphas = np.array(sorted(df_fluxes[param].unique()))
        self.shapevar_grid = [self.alphas]
        self.shapevar_values = [0.5 * (self.alphas[0] + self.alphas[-1])]

        self.energy_distributions = [df_fluxes.loc[df_fluxes[param] == a] for a in self.alphas]
        self.grid = np.array(
            [FixedTabulated(df_flux=df, emax=self.emax, emin=self.emin, alpha=alpha) for df, alpha in zip(self.energy_distributions, self.alphas)]
        )

        # Interpolate within each dataframe
        self.energy_range = np.logspace(np.log10(self.emin), np.log10(self.emax), 1000)
        self.energy_interpolators = [
            interp1d(df["energy"], df["flux"], kind="linear", bounds_error=False, fill_value=0) for df in self.energy_distributions
        ]

        # Create a regular grid interpolator for energy and alpha
        self._initialize_interpolator()

    def _initialize_interpolator(self):
        interpolated_fluxes = np.array([energy_interp(self.energy_range) for energy_interp in self.energy_interpolators])
        self.energy_alpha_interpolator = RegularGridInterpolator(
            (self.alphas, self.energy_range), interpolated_fluxes, bounds_error=False, fill_value=0
        )

    def evaluate(self, energy):
        if np.isscalar(energy):
            return self.energy_alpha_interpolator([self.shapevar_values[0], energy])[0]
        energy = np.array(energy)
        pts = [[self.shapevar_values[0], e] for e in energy]
        return self.energy_alpha_interpolator(pts)

    def prior_transform(self, x):
        return self.alphas[0] + (self.alphas[-1] - self.alphas[0]) * x


class VariableTabulated2D(Component):
    def __init__(self, df_fluxes: pd.DataFrame, emin: float = None, emax: float = None):
        """Custom tabulated flux with two free shape parameters

        Args:
            df_fluxes: A DataFrame with four columns: 'energy', 'flux', and two shape parameters.
                For each combination of shape parameters, the corresponding 'energy'-'flux'
                values must be provided. Therefore, len(DataFrame) = len(param1) * len(param2) * len(energy)

        """
        super().__init__(emin, emax, store="interpolate")

        min_energy = min(df_fluxes["energy"])
        max_energy = max(df_fluxes["energy"])
        self.emin = emin if emin is not None else min_energy
        self.emax = emax if emax is not None else max_energy
        if emin is not None and emin < min_energy:
            warnings.warn("Minimum energy is lower than the minimum tabulated energy")
        if emax is not None and emax > max_energy:
            warnings.warn("Maximum energy is higher than the maximum tabulated energy")

        param1 = df_fluxes.columns[2]
        param2 = df_fluxes.columns[3]
        self.shapevar_names = [param1, param2]
        self.alphas = np.array(sorted(df_fluxes[param1].unique()))
        self.betas = np.array(sorted(df_fluxes[param2].unique()))

        # Equivalent to the fct init_shapevars
        self.shapevar_grid = [self.alphas, self.betas]
        self.shapevar_values = [
            0.5 * (self.alphas[0] + self.alphas[-1]),
            0.5 * (self.betas[0] + self.betas[-1]),
        ]

        self.energy_distributions = [[df_fluxes.loc[(df_fluxes[param1] == i) & (df_fluxes[param2] == j)] for j in self.betas] for i in self.alphas]

        self.grid = np.array(
            [
                [
                    FixedTabulated(df_flux=self.energy_distributions[i][j], emin=self.emin, emax=self.emax, alpha=self.alphas[i], beta=self.betas[j])
                    for j in range(len(self.betas))
                ]
                for i in range(len(self.alphas))
            ]
        )

        # Interpolate within each dataframe
        self.energy_range = np.logspace(np.log10(self.emin), np.log10(self.emax), 1000)
        self.energy_interpolators = [[interp1d(df["energy"], df["flux"], kind="linear") for df in row] for row in self.energy_distributions]

        # Create a regular grid interpolator for energy, alpha, and beta
        self._initialize_interpolator()

    def _initialize_interpolator(self):
        interpolated_fluxes = np.array([[energy_interp(self.energy_range) for energy_interp in beta_row] for beta_row in self.energy_interpolators])
        self.energy_alpha_beta_interpolator = RegularGridInterpolator(
            (self.alphas, self.betas, self.energy_range), interpolated_fluxes, bounds_error=False, fill_value=0
        )

    def evaluate(self, energy):
        if np.isscalar(energy):
            return self.energy_alpha_beta_interpolator([self.shapevar_values[0], self.shapevar_values[1], energy])[0]
        energy = np.array(energy)
        pts = [[self.shapevar_values[0], self.shapevar_values[1], e] for e in energy]
        return self.energy_alpha_beta_interpolator(pts)

    def prior_transform(self, x):
        return (
            np.array([self.alphas[0], self.betas[0]]) + (np.array([self.alphas[-1], self.betas[-1]]) - np.array([self.alphas[0], self.betas[0]])) * x
        )


class FixedPowerLaw(Component):

    def __init__(self, emin: float, emax: float, gamma: float = 2, eref: float = 1):
        """Analytic power law flux with a fixed spectral index.

        Args:
            emin (float): Minimum energy for which the flux is defined.
            emax (float): Maximum energy for which the flux is defined.
            gamma (int, optional): Spectral index. Defaults to 2.
            eref (float, optional): Energy at which the flux equals 1. Defaults to 1.
        """
        super().__init__(emin=emin, emax=emax, store="exact")
        self.eref = eref
        self.shapefix_names = ["gamma"]
        self.shapefix_values = [gamma]

    def evaluate(self, energy: np.ndarray):
        return np.where((self.emin <= energy) & (energy <= self.emax), np.power(energy / self.eref, -self.shapefix_values[0]), 0)


class VariablePowerLaw(Component):

    def __init__(self, emin: float, emax: float, gamma_range: tuple[int, int, int] = (1, 4, 16), eref: float = 1):
        """Analytic power law flux with a variable spectral index.

        Args:
            emin (float): Minimum energy for which the flux is defined.
            emax (float): Maximum energy for which the flux is defined.
            gamma_range (tuple[int, int, int], optional): Specifies the spectral index range as (start, stop, num_points). Defaults to (1, 4, 16).
            eref (float, optional): Energy at which the flux equals 1. Defaults to 1.
        """
        super().__init__(emin=emin, emax=emax, store="interpolate")
        self.eref = eref
        self.shapevar_names = ["gamma"]
        self.shapevar_boundaries = [gamma_range]
        self.init_shapevars()
        self.grid = np.vectorize(partial(FixedPowerLaw, self.emin, self.emax, eref=self.eref))(self.shapevar_grid[0])

    def evaluate(self, energy: np.ndarray):
        return np.where((self.emin <= energy) & (energy <= self.emax), np.power(energy / self.eref, -self.shapevar_values[0]), 0)

    def prior_transform(self, x):
        return self.shapevar_boundaries[0][0] + (self.shapevar_boundaries[0][1] - self.shapevar_boundaries[0][0]) * x


class FixedBrokenPowerLaw(Component):

    def __init__(self, emin: float, emax: float, gamma1: float = 2, gamma2: float = 2, log10ebreak: float = 5, eref: float = 1):
        """Analytic broken power law flux where every shape parameter, the two spectral indices and the break energy, are fixed.

        Args:
            emin (float): Minimum energy for which the flux is defined.
            emax (float): Maximum energy for which the flux is defined.
            gamma1 (float, optional): Low-energy spectral index. Defaults to 2.
            gamma2 (float, optional): High-energy spectral index. Defaults to 2.
            log10ebreak (float, optional): Break energy. Defaults to 5.
            eref (float, optional): Energy at which the flux equals 1. Defaults to 1.
        """
        super().__init__(emin=emin, emax=emax, store="exact")
        self.eref = eref
        self.shapefix_values = [gamma1, gamma2, log10ebreak]
        self.shapefix_names = ["gamma1", "gamma2", "log(ebreak)"]

    def evaluate(self, energy: np.ndarray):
        factor = (10 ** self.shapefix_values[2] / self.eref) ** (self.shapefix_values[1] - self.shapefix_values[0])
        f = np.where(
            np.log10(energy) < self.shapefix_values[2],
            np.power(energy / self.eref, -self.shapefix_values[0]),
            factor * np.power(energy / self.eref, -self.shapefix_values[1]),
        )
        return np.where((self.emin <= energy) & (energy <= self.emax), f, 0)


class VariableBrokenPowerLaw(FixedBrokenPowerLaw):

    def __init__(
        self,
        emin: float,
        emax: float,
        gamma1_range: tuple[int, int, int] = (1, 4, 16),
        gamma2_range: tuple[int, int, int] = (0, 3, 10),
        log10ebreak_range: tuple[int, int, int] = (3, 6, 4),
        eref: float = 1,
    ):
        """Analytic broken power law flux where every shape parameter, the two spectral indices and the break energy, are left free to be fitted.

        Args:
            emin (float): Minimum energy for which the flux is defined.
            emax (float): Maximum energy for which the flux is defined.
            gamma1_range (tuple[int, int, int], optional): Specifies the LE spectral index range as (start, stop, num_points). Defaults to (1, 4, 16).
            gamma2_range (tuple[int, int, int], optional): Specifies the HE spectral index range as (start, stop, num_points). Defaults to (0, 3, 10).
            log10ebreak_range (tuple[int, int, int], optional): Specifies the break energy range as (start, stop, num_points). Defaults to (3, 6, 4).
            eref (float, optional): Energy at which the flux equals 1. Defaults to 1.
        """
        super().__init__(emin=emin, emax=emax)
        self.store = "interpolate"
        self.eref = eref
        self.shapevar_names = ["gamma1", "gamma2", "log(ebreak)"]
        self.shapevar_boundaries = np.array([[*gamma1_range], [*gamma2_range], [*log10ebreak_range]])
        self.init_shapevars()
        self.grid = np.vectorize(partial(FixedBrokenPowerLaw, self.emin, self.emax, eref=self.eref))(*np.meshgrid(*self.shapevar_grid, indexing="ij"))

    def evaluate(self, energy: np.ndarray):
        factor = (10 ** (self.shapevar_values[2]) / self.eref) ** (self.shapevar_values[1] - self.shapevar_values[0])
        f = np.where(
            np.log10(energy) < self.shapevar_values[2],
            np.power(energy / self.eref, -self.shapevar_values[0]),
            factor * np.power(energy / self.eref, -self.shapevar_values[1]),
        )
        return np.where((self.emin <= energy) & (energy <= self.emax), f, 0)

    def prior_transform(self, x):
        return self.shapevar_boundaries[:, 0] + (self.shapevar_boundaries[:, 1] - self.shapevar_boundaries[:, 0]) * x


class QuasithermalSpectrum(Component):
    
    def __init__(self, emean: float = 100, alpha: float = 2):
        """Quasi thermal spectrum of the type f(E) = (E/Emean)^alpha * exp(-(alpha+1) * E/Emean). 
        It is defined from Emin = 0.001*Emean to Emax = 1000*Emax

        Args:
            emean (float, optional): Mean energy in GeV. Defaults to 100.
            alpha (float, optional): Pinch parameter. Defaults to 2.
        """
        super().__init__(emin=emean/1000, emax=emean*1000, store="exact")
        self.alpha = alpha
        self.shapefix_names = ["emean"]
        self.shapefix_values = [emean]

    def evaluate(self, energy: np.ndarray):
        x = energy / self.shapefix_values[0]
        return np.where((self.emin <= energy) & (energy <= self.emax), np.power(x, self.alpha) * np.exp(-(self.alpha+1)*x), 0)


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


class FluxFixedTabulated(FluxBase):

    def __init__(self, datafile, emin=None, emax=None):
        super().__init__()
        self.components = [FixedTabulated(datafile, emin, emax)]


class FluxVariableTabulated1D(FluxBase):

    def __init__(self, datafile, emin=None, emax=None):
        super().__init__()
        self.components = [VariableTabulated1D(datafile, emin, emax)]


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


class FluxQuasiThermal(FluxBase):
    
    def __init__(self, emean=100, alpha=2):
        super().__init__()
        self.components = [QuasithermalSpectrum(emean, alpha)]
