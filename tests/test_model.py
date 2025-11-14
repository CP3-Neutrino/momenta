import tempfile
import unittest
import healpy as hp
import numpy as np
import pandas as pd

import momenta.utils.conversions
import momenta.utils.flux
from momenta.io import GWDatabase, NuDetector, Parameters, PointSource, Stack
from momenta.io.neutrinos import BackgroundGaussian, NuEvent
from momenta.io.neutrinos_irfs import EffectiveAreaAllSky, IsotropicBackground, VonMisesSignal
from momenta.stats.model import ModelOneSource, ModelOneSource_BkgOnly
from momenta.stats.run import run_ultranest, run_ultranest_stack
from momenta.stats.constraints import get_limits, get_limits_with_uncertainties, get_bestfit
from momenta.stats.constraints import compute_differential_limits, get_hpd_interval


class EffAreaDet(EffectiveAreaAllSky):
    def evaluate(self, energy, ipix, nside):
        return (energy / 100) ** 2 * np.exp(-energy / 3000)


class TestModels(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        config_str = """
            skymap_resolution: 8
            detector_systematics: 0

            analysis:
                likelihood: pointsource
                prior_normalisation:
                    variable: flux
                    type: flat-log
                    range:
                        min: 1.0e-10
                        max: 1.0e+10
        """
        self.config_file = f"{self.tmpdir}/config.yaml"
        with open(self.config_file, "w") as f:
            f.write(config_str)
        # configuration
        self.pars = Parameters(self.config_file)
        self.pars.set_flux(momenta.utils.flux.FluxFixedPowerLaw(1, 100, 2))
        # source
        self.src1 = PointSource(100, 20, 0, name="GRB")
        self.src1.set_distance(80)
        # detector
        det1_dict = {"name": "Detector1", "samples": ["SampleA"]}
        self.det1 = NuDetector(det1_dict)
        self.det1.set_effective_areas([EffAreaDet()])
        self.det1.set_observations([0], [BackgroundGaussian(0.5, 0.3)])

    def test_repr(self):
        model = ModelOneSource(self.det1, self.src1, self.pars)
        self.assertTrue("flux=FixedPowerLaw" in model.__str__())
        self.assertEqual(model.ndims, 2)
        self.assertEqual(model.param_names[0], "norm0")
        model = ModelOneSource_BkgOnly(self.det1, self.pars)
        self.assertTrue("flux=FixedPowerLaw" in model.__str__())
        self.assertEqual(model.ndims, 0)
        
    def test_prior(self):
        model = ModelOneSource(self.det1, self.src1, self.pars)
        p = model.prior(np.array([[0.5, 0.5]]))
        self.assertAlmostEqual(p[0,0], 1)
        self.assertAlmostEqual(p[0,1], 0)
    
    def test_loglike(self):
        model = ModelOneSource(self.det1, self.src1, self.pars)
        self.assertAlmostEqual(model.loglike(np.array([[0.5, 0.5]]))[0], -0.5008100282670334)
    
    def test_deterministics(self):
        model = ModelOneSource(self.det1, self.src1, self.pars)
        samples = {
            "itoy": np.array([0]),
            "norm0": np.array([1]),
        }
        det = model.calculate_deterministics(samples)
        self.assertAlmostEqual(det["etot"][0], 5.5455346924194226e+51)


class TestRunModels(unittest.TestCase):
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
        det1_dict = {"name": "Detector1", "samples": ["SampleA"], "errors": {"acceptance": 0.02, "acceptance_corr": 1}}
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
        lim = get_limits_with_uncertainties(result)["etot0"]
        self.assertLessEqual(np.abs(lim[0]/8.0e54 - 1), 0.15)
        self.assertLessEqual(np.abs(lim[1]/2.0e53 - 1), 0.4)
        hpd = get_hpd_interval(result["samples"]["etot0"])
        self.assertLessEqual(np.abs(hpd[0][1]/8.0e54 - 1), 0.15)

    def test_limits_onesource_wsyst(self):
        self.pars.apply_det_systematics = True
        self.pars.likelihood_method = "pointsource"
        _, result = run_ultranest(self.det1, self.src1, self.pars)
        lim = get_limits(result)["etot0"]
        self.assertLessEqual(np.abs(lim/8.0e54 - 1), 0.15)
        
    def test_limits_stacked_nosyst(self):
        self.pars.apply_det_systematics = False
        self.pars.likelihood_method = "pointsource"
        stack = Stack()
        stack[self.src1] = self.det1
        stack[self.src2] = self.det2
        _, result = run_ultranest_stack(stack, self.pars)
        lim = get_limits(result)["etot0"]
        self.assertLessEqual(np.abs(lim/7.5e54 - 1), 0.15)
        
    def test_difflimits_onesource_nosyst(self):
        self.pars.apply_det_systematics = False
        self.pars.likelihood_method = "poisson"
        bins = np.logspace(0, 4, 5)
        lims = compute_differential_limits(self.det1, self.src1, self.pars, bins)
        for lim, ref in zip(lims, [2900, 28, 0.35, 0.02]):
            self.assertLessEqual(np.abs(lim/ref - 1), 0.15)

    def test_consistency_tabulated_nosyst(self):
        self.pars.apply_det_systematics = False
        self.pars.likelihood_method = "pointsource"    
        # Analytic run
        self.pars.set_flux(momenta.utils.flux.FluxVariablePowerLaw(1e-1, 1e7, gamma_range=(2, 5, 21)))
        _, result_ana = run_ultranest(self.det2, self.src1, self.pars)
        fit_ana = get_bestfit(result_ana['weighted_samples']['points']['flux0_gamma'])
        # Tabulated run
        spl = self.pars.flux.components[0]
        gammas = spl.shapevar_grid[0]
        e_range = np.logspace(-1, 7, 100)
        df_1d = []
        for g in gammas:
            spl.set_shapevars([g])
            fluxes = spl.evaluate(e_range)
            df_1d.append(pd.DataFrame({'energy': e_range, 'flux': fluxes, 'gamma': [g]*len(e_range)}))
        df_1d = pd.concat(df_1d, ignore_index=True)
        self.pars.set_flux(momenta.utils.flux.FluxVariableTabulated1D(df_1d))
        _, result_tab = run_ultranest(self.det2, self.src1, self.pars)
        fit_tab = get_bestfit(result_tab['weighted_samples']['points']['flux0_gamma'])
        # Compare results
        self.assertLessEqual(np.abs((fit_ana - fit_tab)/fit_ana), 0.15)


if __name__ == "__main__":
    t = TestModels()
    t.setUp()
    t.test_repr()
    t.test_prior()
    t.test_loglike()
    t.test_deterministics()