import tempfile
import unittest
import healpy as hp
import numpy as np

import momenta.utils.conversions
import momenta.utils.flux
from momenta.io import GWDatabase, NuDetector, Parameters, PointSource, Stack
from momenta.io.neutrinos import BackgroundGaussian, NuEvent
from momenta.io.neutrinos_irfs import EffectiveAreaAllSky, IsotropicBackground, VonMisesSignal
from momenta.stats.run import run_ultranest, run_ultranest_stack
from momenta.stats.constraints import get_limits


class EffAreaDet(EffectiveAreaAllSky):
    def evaluate(self, energy, ipix, nside):
        return (energy / 100) ** 2 * np.exp(-energy / 3000)


class TestExamples(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        config_str = """
            skymap_resolution: 8
            detector_systematics: 0

            analysis:
                likelihood: pointsource
                prior_normalisation:
                    variable: etot
                    type: flat-linear
                    range:
                        min: 0.0
                        max: 1.0e+60
        """
        self.config_file = f"{self.tmpdir}/config.yaml"
        with open(self.config_file, "w") as f:
            f.write(config_str)
        #
        self.gwdb_file = "examples/input_files/gw_catalogs/database_example.csv"
        self.db_file = f"{self.tmpdir}/db.csv"
        
        # configuration
        self.pars = Parameters(self.config_file)
        self.pars.set_flux(momenta.utils.flux.FluxFixedPowerLaw(1, 100, 2))
        # sources database
        database_gw = GWDatabase(self.gwdb_file)
        database_gw.set_parameters(self.pars)
        self.src1 = PointSource(100, 20, 0, name="GRB")
        self.src1.set_distance(80)
        self.src2 = database_gw.find_gw("GW190412")
        # detector
        det1_dict = {"name": "Detector1", "samples": ["SampleA"]}
        self.det1 = NuDetector(det1_dict)
        self.det1.set_effective_areas([EffAreaDet()])
        self.det1.set_observations([0], [BackgroundGaussian(0.5, 0.3)])
        det2_dict = {"name": "Detector2", "samples": ["SampleA", "SampleB"]}
        self.det2 = NuDetector(det2_dict)
        self.det2.set_effective_areas([EffAreaDet(), EffAreaDet()])
        self.det2.set_observations([0, 1], [BackgroundGaussian(0.5, 0.3), BackgroundGaussian(0.5, 0.1)])
        self.det2.set_events([[], [NuEvent(ra=0.1, dec=0.5, sigma=0.01)]])
        self.det2.set_pdfs([{}, {"sig_ang": VonMisesSignal(), "bkg_ang": IsotropicBackground()}])

    def test_limits_onesource_nosyst(self):
        self.pars.apply_det_systematics = False
        self.pars.likelihood_method = "pointsource"
        _, result = run_ultranest(self.det1, self.src1, self.pars)
        self.assertLessEqual(np.abs(get_limits(result)["etot0"]/7.9e54 - 1), 0.1)

    def test_limits_onesource_wsyst(self):
        self.pars.apply_det_systematics = True
        self.pars.likelihood_method = "pointsource"
        _, result = run_ultranest(self.det1, self.src1, self.pars)
        self.assertLessEqual(np.abs(get_limits(result)["etot0"]/7.9e54 - 1), 0.15)
        
    def test_limits_stacked_nosyst(self):
        self.pars.apply_det_systematics = False
        self.pars.likelihood_method = "pointsource"
        stack = Stack()
        stack[self.src1] = self.det1
        stack[self.src2] = self.det2
        _, result = run_ultranest_stack(stack, self.pars)
        self.assertLessEqual(np.abs(get_limits(result)["etot0"]/7.5e54 - 1), 0.15)
