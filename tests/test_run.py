import tempfile
import unittest
import healpy as hp
import numpy as np

import momenta.utils.conversions
import momenta.utils.flux
from momenta.io import GWDatabase, NuDetector, Parameters
from momenta.io.neutrinos import BackgroundGaussian
from momenta.io.neutrinos_irfs import EffectiveAreaBase
from momenta.stats.run import run_ultranest


class EffectiveArea(EffectiveAreaBase):
    def __init__(self, const):
        super().__init__()
        self.const = const

    def evaluate(self, energy, ipix, nside):
        return self.const


class TestExamples(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        config_str = """
            skymap_resolution: 8
            detector_systematics: 0

            analysis:
                likelihood: poisson
                prior_normalisation:
                    type: flat-linear
                    range: [0.0, 10000000000]
        """
        self.config_file = f"{self.tmpdir}/config.yaml"
        with open(self.config_file, "w") as f:
            f.write(config_str)
        #
        detector_str = """
            name: TestDet
            samples: ["A", "B"]

            errors:
              acceptance: 0.1
              acceptance_corr: 0.5
        """
        self.det_file = f"{self.tmpdir}/detector.yaml"
        with open(self.det_file, "w") as f:
            f.write(detector_str)
        self.accs = [np.ones(hp.nside2npix(8)), np.ones(hp.nside2npix(8))]
        #
        self.gwdb_file = "examples/input_files/gw_catalogs/database_example.csv"
        self.db_file = f"{self.tmpdir}/db.csv"

        # configuration
        self.pars = Parameters(self.config_file)
        self.pars.set_models(momenta.utils.flux.FluxFixedPowerLaw(1, 100, 2), momenta.utils.conversions.JetIsotropic())
        # GW database
        database_gw = GWDatabase(self.gwdb_file)
        database_gw.set_parameters(self.pars)
        self.gw = database_gw.find_gw("GW190412")
        # detector
        self.det = NuDetector(self.det_file)
        bkg = [BackgroundGaussian(b, b / 5) for b in [0.1, 0.3]]
        self.det.set_effective_areas([EffectiveArea(2), EffectiveArea(1)])
        self.det.set_observations([0, 0], bkg)

    def test_limits_nosyst(self):
        self.pars.apply_det_systematics = False
        self.pars.likelihood_method = "pointsource"
        run_ultranest(self.det, self.gw, self.pars)

    def test_limits_wsyst(self):
        self.pars.apply_det_systematics = True
        self.pars.likelihood_method = "pointsource"
        run_ultranest(self.det, self.gw, self.pars)
