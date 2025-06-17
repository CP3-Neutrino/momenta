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

from momenta.io.gw import GW, GWDatabase
from momenta.io.transient import Transient, PointSource
from momenta.io.parameters import Parameters
from momenta.io.neutrinos import NuDetectorBase, NuDetector, SuperNuDetector, NuEvent
from momenta.io.stacking import Stack
