import healpy as hp
import numpy as np
import tempfile
import unittest

from momenta.io import Parameters, GWDatabase
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

    def test_skymap(self):
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
