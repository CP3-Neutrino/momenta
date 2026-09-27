import healpy as hp
import numpy as np
import tempfile
import unittest
import logging

from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle

from momenta.io import Parameters, GWDatabase, GW, Transient, PointSource
import momenta.utils.conversions
import momenta.stats


class TestJetModels(unittest.TestCase):
    def test_isotropic(self):
        jet = momenta.utils.conversions.JetIsotropic()
        jet.eiso_to_etot(0)
        print(jet, jet.str_filename)

    def test_vonmises(self):
        jet = momenta.utils.conversions.JetVonMises(np.inf)
        jet.eiso_to_etot(0)
        jet.eiso_to_etot(0.5)
        print(jet, jet.str_filename)
        jet = momenta.utils.conversions.JetVonMises(0.1)
        jet.eiso_to_etot(0)
        jet.eiso_to_etot(0.5)
        print(jet, jet.str_filename)
        jet = momenta.utils.conversions.JetVonMises(0.1, with_counter=True)
        jet.eiso_to_etot(0)
        jet.eiso_to_etot(0.5)
        print(jet, jet.str_filename)

    def test_rectangular(self):
        jet = momenta.utils.conversions.JetRectangular(np.inf)
        jet.eiso_to_etot(0)
        jet.eiso_to_etot(0.5)
        print(jet, jet.str_filename)
        jet = momenta.utils.conversions.JetRectangular(0.1)
        jet.eiso_to_etot(0)
        jet.eiso_to_etot(0.5)
        print(jet, jet.str_filename)
        jet = momenta.utils.conversions.JetRectangular(0.1, with_counter=True)
        jet.eiso_to_etot(0)
        jet.eiso_to_etot(0.5)
        print(jet, jet.str_filename)

    def test_list(self):
        momenta.utils.conversions.list_jet_models()


class TestGW(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        config_str = """
            skymap_resolution: 8
            detector_systematics: 0

            analysis:
                likelihood: poisson
                prior_normalisation:
                    variable: flux
                    type: flat-linear
                    range:
                        min: 0.0
                        max: 1.0e+10
        """
        self.config_file = f"{self.tmpdir}/config.yaml"
        with open(self.config_file, "w") as f:
            f.write(config_str)
        self.pars = Parameters(self.config_file)
        self.dbgw = GWDatabase("examples/input_files/gw_catalogs/database_example.csv")
        self.dbgw.set_parameters(self.pars)
        self.gw = self.dbgw.find_gw("GW190412")
        self.tmpdir = tempfile.mkdtemp()

    def test_constructor(self):
        gw = GW()
        gw.set_parameters(self.pars)

    def test_skymap(self):
        self.assertAlmostEqual(np.sum(self.gw.fits.get_skymap()), 1)
        self.assertAlmostEqual(np.sum(self.gw.fits.get_skymap(4)), 1)
        self.assertTrue(np.all(self.gw.fits.get_signal_region(8, None) == np.arange(hp.nside2npix(8))))
        self.assertTrue(np.all(self.gw.fits.get_signal_region(8, 0.90) == [163, 131, 164]))
        self.assertTrue(np.isclose(self.gw.fits.get_ra_dec_bestfit(8)[1], 35.68533471265204))

    def test_database(self):
        emptydb = GWDatabase()
        emptydb.add_entry("ev", "", "")
        with self.assertRaises(RuntimeError):
            emptydb.save()
        emptydb = GWDatabase(f"{self.tmpdir}/db.csv")
        emptydb.add_entry("ev", "", "")
        emptydb.save(f"{self.tmpdir}/db.csv")
        emptydb.save()
        #
        with self.assertRaises(RuntimeError):
            self.dbgw.find_gw("missing_ev")
        self.dbgw.list_all()
        self.dbgw.list("BBH", 0, 1000)
        self.dbgw.list("BNS", 1000, 0)
        self.dbgw.add_entry("ev", "", "")
        self.dbgw.save(f"{self.tmpdir}/db.csv")
        
    def test_samples(self):
        gw = self.dbgw.find_gw("GW190412")
        gw.samples_priorities = None
        gw.samples.find_correct_sample()
        with self.assertRaises(RuntimeError):
            gw.prepare_prior_samples(4)
        gw.samples = None
        gw.prepare_prior_samples(4)        


class TestTransient(unittest.TestCase):
    def setUp(self):
        self.params = {
            "name":"test",
            "utc": Time.now(),
        }
        
    
    def test_constructor(self):
        """Check that parametrers are set correctly"""
        transient = Transient(**self.params)
        for par in self.params:
            with self.subTest(par=par):
                self.assertEqual(getattr(transient, par), self.params[par])
    
    def test_repr(self):
        self.assertNotEqual(repr(Transient("a")), repr(Transient("b")))
    
    def test_log(self):
        transient = Transient(**self.params, logger="atnemom")
        with self.assertLogs('atnemom', level='INFO') as cm:
            transient.log.info('neutrino')
        self.assertEqual(cm.output, ['INFO:atnemom:neutrino'])
        


class TestPointSource(unittest.TestCase):
    def setUp(self):
        self.params = {
            "ra_deg":0,
            "dec_deg":0,
            "name":"test",
            "utc":Time.now(),
        }
    
    def test_constructor(self):
        """check parameters from test are set correctly"""
        ps = PointSource(**self.params)
        for par in self.params:
            with self.subTest(par=par):
                self.assertEqual(getattr(ps, par), self.params[par])
        self.assertEqual(ps.err.value, 0)
    
    def test_set_distance(self):
        """check distance setting and conversion"""
        ps = PointSource(**self.params)
        ps.set_distance(1)
        self.assertEqual(ps.distance, 1)
        self.assertAlmostEqual(ps.redshift, momenta.utils.conversions.lumidistance_to_redshift(1))
    
    def test_set_redshift(self):
        """check redshift setting and conversion"""
        ps = PointSource(**self.params)
        ps.set_redshift(1)
        self.assertEqual(ps.redshift, 1)
        self.assertAlmostEqual(ps.distance, momenta.utils.conversions.redshift_to_lumidistance(1))

    def test_samples(self):
        """check prior samples of positional uncertainty"""
        nside, nsample = 32, 1000_000
        # default no uncertainty, we get a single sample at the source position
        ps = PointSource(**self.params)
        ps.set_redshift(1)
        toys_0 = ps.prepare_prior_samples(nside=nside, size=nsample)
        self.assertEqual(toys_0["ra"][0], ps.ra_deg)
        self.assertEqual(toys_0["dec"][0], ps.dec_deg)
        self.assertIn("distance_scaling", toys_0.dtype.names)

        # other test cases:
        ps_1deg = PointSource(0, 0, 1)
        ps_1deg.set_redshift(1)
        ps_10deg = PointSource(0, 0, 10)
        ps_1deg_pole = PointSource(0, 90, 1)
        ps_10deg_pole = PointSource(0, 90, 10)
        
        toys_1deg = ps_1deg.prepare_prior_samples(nside, size=nsample)
        toys_10deg = ps_10deg.prepare_prior_samples(nside, size=nsample)
        toys_1deg_pole = ps_1deg_pole.prepare_prior_samples(nside, size=nsample)
        toys_10deg_pole = ps_10deg_pole.prepare_prior_samples(nside, size=nsample)

        self.assertIn("distance_scaling", toys_1deg.dtype.names)
        
        # for 10 deg, the small-angle approximation of vMF is still close enough to preserve scaling, test it:
        containment_1 = np.count_nonzero(toys_1deg_pole["dec"] > (90 - 1)) / nsample
        containment_10 = np.count_nonzero(toys_10deg_pole["dec"] > (90 - 10)) / nsample
        self.assertAlmostEqual(containment_1, containment_10, places=2, msg="containment not preserved in scaling")
        containment_expected = 1 - 1 / np.sqrt(np.e) # 39.3%
        self.assertAlmostEqual(containment_1, containment_expected, places=2, msg="containment wrong for vMF with sigma=1")
        self.assertAlmostEqual(containment_10, containment_expected, places=2, msg="containment wrong for vMF with sigma=10")

        # the peak should be near the centroid
        for dec, toys in [(0, toys_1deg), (90, toys_1deg_pole)]:
            with self.subTest("centroid position as expected", dec=dec):
                ipix_max = np.argmax(np.bincount(toys["ipix"]))
                ra_max, dec_max = hp.pix2ang(nside, ipix_max, lonlat=True)
                self.assertAlmostEqual(np.deg2rad(dec_max), np.deg2rad(dec), places=1)
                if dec != 90: # RA is degenerate at poles
                    self.assertAlmostEqual(np.deg2rad(ra_max), 0, places=1)

        # priors at pole and equator are just rotated from each other
        coords_10_equator = SkyCoord(ra=toys_10deg["ra"], dec=toys_10deg["dec"], unit="deg")
        ang_10_equator = ps_10deg.coords.separation(coords_10_equator).deg
        containment_10_equator = np.count_nonzero(ang_10_equator < 10) / nsample
        self.assertAlmostEqual(containment_10, containment_10_equator, places=1, msg="prior shape at pole != at equator")
        

class TestParameters(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        config_str = """
            skymap_resolution: 8
            detector_systematics: 0

            analysis:
                likelihood: poisson
                prior_normalisation:
                    variable: flux
                    type: flat-linear
                    range:
                        min: 0.0
                        max: 1.0e+10
        """
        self.config_file = f"{self.tmpdir}/config.yaml"
        with open(self.config_file, "w") as f:
            f.write(config_str)
        pars = Parameters(self.config_file)
        print(pars)
        print(pars.str_filename)


class TestRadians(unittest.TestCase):
    """test converting angles to radians"""

    def test_none(self):
        self.assertTrue(momenta.utils.conversions.to_radians(None) is None)
    
    def test_from_quantity(self):
        self.assertEqual(momenta.utils.conversions.to_radians(0 * u.deg), 0.)
        self.assertAlmostEqual(momenta.utils.conversions.to_radians(180 * u.deg), np.pi)
    
    def test_from_angle(self):
        self.assertEqual(momenta.utils.conversions.to_radians(Angle(0, unit=u.deg)), 0.)
        self.assertAlmostEqual(momenta.utils.conversions.to_radians(Angle(180, unit=u.deg)), np.pi)

    def test_from_float(self):
        self.assertEqual(momenta.utils.conversions.to_radians(0), 0.)
        self.assertAlmostEqual(momenta.utils.conversions.to_radians(180), np.pi)

    def test_error(self):
        self.assertRaises(TypeError, momenta.utils.conversions.to_radians, "foo")


