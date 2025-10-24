import numpy as np
import unittest

import momenta.io.neutrinos as nu
import momenta.io.neutrinos_irfs as irfs
import momenta.utils.flux as flux


class EnergySignal(irfs.EnergySignal):
    def __init__(self):
        super().__init__()

    def __call__(self, evt, fluxcomponent):
        return 1 / (np.log10(fluxcomponent.emax) - np.log10(fluxcomponent.emin)) if fluxcomponent.emin <= evt.energy <= fluxcomponent.emax else 0


class TestIRFs(unittest.TestCase):
    def test_aeffs(self):
        aeff = irfs.EffectiveAreaBase()
        self.assertEqual(aeff.evaluate(0, 0, 0), 0)
        

class TestSample(unittest.TestCase):
    def setUp(self):
        self.s1 = nu.NuSample("sample")
        self.s1.set_observations(0, nu.BackgroundFixed(0.5))
        self.s1.set_pdfs(
            sig_ang=irfs.VonMisesSignal(),
            sig_ene=EnergySignal(),
            bkg_ang=irfs.AngularBackground(lambda ra, dec, energy: 1),
            bkg_ene=irfs.EnergyBackground(lambda ra, dec, energy: 1 / energy),
        )

    def test_members(self):
        self.assertEqual(self.s1.name, "sample")
        self.assertEqual(self.s1.nobserved, 0)
        self.assertEqual(self.s1.background.nominal, 0.5)

    def test_background(self):
        self.assertEqual(nu.BackgroundFixed(1).nominal, 1)
        self.assertEqual(nu.BackgroundGaussian(1, 0.1).nominal, 1)
        self.assertEqual(nu.BackgroundPoisson(10, 10).nominal, 1)

    def test_pdfs(self):
        f = flux.FluxFixedPowerLaw(1, 100, 2)
        ev = nu.NuEvent(ra=0, dec=0, sigma=0.01, energy=10)
        ra_src, dec_src = 0, 0
        #
        p_bkg = self.s1.compute_background_probability(ev)
        self.assertEqual(p_bkg, 1 / ev.energy)
        #
        p_sig = self.s1.compute_signal_probability(ev, f.components[0], ra_src, dec_src)
        self.assertEqual(p_sig, 1 / (4*np.pi) / ev.sigma**2)


class TestDetector(unittest.TestCase):
    def setUp(self):
        self.dict_det1 = {
            "name": "Test",
            "nsamples": 1,
            "samples": ["sample1"],
            "errors": {"acceptance": 0, "acceptance_corr": 0, "background": 0},
        }
        self.dict_det2 = {
            "name": "Test2",
            "nsamples": 4,
            "samples": ["sample1", "sample2", "sample3", "sample4"],
            "errors": {"acceptance": 0.40, "acceptance_corr": 1, "background": 0.40},
        }
        self.d1 = nu.NuDetector(self.dict_det1)
        self.d2 = nu.NuDetector(self.dict_det2)
        self.d2.set_observations(
            [0, 0, 0, 0],
            [
                nu.BackgroundFixed(0.3),
                nu.BackgroundGaussian(0.1, 0.01),
                nu.BackgroundGaussian(0.1, 0.02),
                nu.BackgroundPoisson(3, 10),
            ],
        )

    def test_members(self):
        self.assertEqual(self.d1.nsamples, 1)
        self.assertEqual(self.d2.nsamples, 4)
        self.assertEqual(self.d1.name, "Test")
        self.assertEqual(len(self.d2.samples), 4)
        self.assertEqual(self.d1.samples[0].name, "sample1")

    def test_exceptions(self):
        with self.assertRaises(TypeError):
            nu.NuDetector(0)
        with self.assertRaises(ValueError):
            self.d1.set_observations([0, 0], [nu.BackgroundFixed(0)])
        with self.assertRaises(ValueError):
            self.d1.set_observations([0], [nu.BackgroundFixed(0), nu.BackgroundFixed(0)])

    def test_superdetector(self):
        sd = nu.SuperNuDetector("SD")
        sd.add_detector(self.d1)
        sd.add_detector(self.d2)
        with self.assertLogs(level="ERROR"):
            sd.add_detector(self.d2)
        self.d1.set_observations([0], [nu.BackgroundFixed(0)])
        self.assertEqual(sd.nsamples, 5)
        self.assertEqual(len(list(sd.samples)), 5)


class TestOther(unittest.TestCase):
    def test_infer_uncertainties(self):
        self.assertTrue(np.array_equal(nu.infer_uncertainties(0, 2), np.array([[0, 0], [0, 0]])))
        self.assertTrue(np.array_equal(nu.infer_uncertainties([1, 2], 2), np.array([[1, 0], [0, 4]])))
        self.assertTrue(np.array_equal(nu.infer_uncertainties([1, 2], 2, 1), np.array([[1, 2], [2, 4]])))
        self.assertTrue(np.array_equal(nu.infer_uncertainties([[1, 0], [0, 4]], 2), np.array([[1, 0], [0, 4]])))
        self.assertIsNone(nu.infer_uncertainties(None, 1))
        with self.assertRaises(RuntimeError):
            nu.infer_uncertainties([0, 0, 0], 2)
