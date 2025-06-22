import unittest
import tutorial

# https://docs.python.org/3/library/unittest.html#unittest.TestCase.debug
# https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertNotIsInstance
# https://docs.python.org/3/library/unittest.html#unittest.TestCase.assertNoLogs
# https://www.youtube.com/watch?v=6tNS--WetLI&t=1484s

class TestTutorial(unittest.TestCase): # կարանք ինչ ուզում ենք դնենք անունը
    def test_divide(self):
        self.assertEqual(tutorial.divide(4, 2), 2)
        self.assertEqual(tutorial.divide(4, 0), None)
        self.assertEqual(tutorial.divide(4, 1), 4)
        # self.assertAlmostEquals(tutorial.divide(4, 3), 1.3, places=3)
        #      self.assertRaises(ZeroDivisionError, tutorial.divide, 4, 0)

        # with self.assertRaises(ValueError):
        #     tutorial.divide(4, 0)


    def test_is_prime(self):
        self.assertTrue(tutorial.is_prime(7))
        self.assertFalse(tutorial.is_prime(8))
        self.assertFalse(tutorial.is_prime(1))
        self.assertFalse(tutorial.is_prime(0))

    def test_generate_random_num(self):
        self.assertIn(tutorial.generate_random_num(), range(0, 101))
# python -m unittest test_tutorical.py

if __name__ == '__main__':
    unittest.main()


#      self.assertRaises(ZeroDivisionError, tutorial.divide, 4, 0)
# with self.assertRaises(ZeroDivisionError):
#     tutorial.divide(4, 0)


