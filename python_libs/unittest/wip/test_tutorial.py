import unittest # pytest, nose2
import tutorial

class TestTutorial(unittest.TestCase):
    def test_divide(self):
        self.assertEqual(tutorial.divide(5, 1), 5)
        # self.assertEqual(tutorial.divide(4, 0), None)
        self.assertAlmostEqual(tutorial.divide(4, 3), 1.3, places=1)
        self.assertRaises(ValueError, tutorial.divide, 4, 0)

        # with self.assertRaises(ValueError):
        #     tutorial.divide(4, 0)
    
    def test_is_prime(self):
        self.assertTrue(tutorial.is_prime(7))
        self.assertFalse(tutorial.is_prime(8))
        self.assertFalse(tutorial.is_prime(1))
        self.assertFalse(tutorial.is_prime(0)) 

    # def test_generate_random_num(self):
    #     self.assertIn(tutorial.generate_random_num(), range(0, 101)) 
        
if __name__ == '__main__':
    unittest.main()