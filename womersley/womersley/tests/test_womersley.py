
from unittest import TestCase

import womersley #.utils
import numpy as np

class TestWomersley(TestCase):
    def test_coef(self):
        s = womersley.utils.get_coefficients()
        self.assertEqual(len(s), 14)
        
#class TestWomersley(TestCase):        
    def test_Waveform_Data(self):
        x = np.array([[1000, 2000], [3000, 4000], [5000, 6000]], np.float64) #(3,2)
        #print(x.shape)
        s = womersley.components.Waveform_Data(dir_path="path.txt", data=x,
                                               radius = 1.2, waveform_ctr_name="waveform_name.txt") #
                                               #period=0.0, output_path='', scale=1000.0, mu=0.0035, rho=1050.0, kind='ctr )
        
        self.assertTrue(s.periodiszero())
        self.assertRaises(ValueError)
        s.set_period(1.0)
        self.assertFalse(s.periodiszero())
        self.assertAlmostEqual(s.get_period(), 1.0)
        
        self.assertAlmostEqual(s.get_radius(), 1.2)
        s.set_radius (1.0)
        self.assertAlmostEqual(s.get_radius(), 1.0)
        
        self.assertEqual(s.output_path, s.dir_path)
        s.set_output_path("hallo")
        self.assertEqual(s.output_path, "hallo")

        self.assertAlmostEqual(s.time[0], 1.0)
        self.assertAlmostEqual(s.flowvelocity[0], 2.0)
        
        self.assertAlmostEqual(s.mu, 0.0035)
        self.assertAlmostEqual(s.rho, 1050.0)
        
        self.assertEqual(s.kind, 'ctr')
        
        
        self.assertEqual(s.waveform_ctr_name, "waveform_name.txt")


    def test_reconstruct(self):
        coef = womersley.utils.get_coefficients()
        q, t = womersley.utils.reconstruct(coef, T=1.0, t_pts=200)
        q2, t2 = womersley.utils.reconstruct2(coef, T=1.0, t_pts=200)
        np.testing.assert_allclose(q, q2)
        np.testing.assert_allclose(t, t2)
        #self.assertEqual(len(s), 14)
        
    def test_reconstruct_pt(self):
        coef = womersley.utils.get_coefficients()
        q = womersley.utils.reconstruct_pt(coef, 0.2, T=1.0)
        q2 = womersley.utils.reconstruct_pt2(coef, 0.2, T=1.0)
        np.testing.assert_allclose(q, q2)
        #self.assertEqual(len(s), 14)
