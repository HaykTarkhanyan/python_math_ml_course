import unittest
import movies 

import json
import os

def test_add(a: bool, b: bool):
    pass


class TestMovies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("setting up class")

    @classmethod
    def tearDownClass(cls):
        print("tearing down class")

    def setUp(self):
        print("\nsetting up")
        PATH = os.path.join("..", "imdb_top_999.json")

        with open(PATH, "r") as f:
            data = json.load(f)

        KEYS_TO_DELETE = ['Meta_score', "No_of_Votes", "Overview"]

        for i in data.keys():
            for k in KEYS_TO_DELETE: 
                del data[i][k]
            
        for i in data.keys():
            data[i]["Name"] = i

        self.kino = movies.Kino(**data["The Shawshank Redemption"])

    def tearDown(self):
        print("tearing down\n")
        del self.kino

    def test_get_actors(self):
        print("test get actors")
        self.assertEqual(self.kino.get_actors(), "Tim Robbins, Morgan Freeman, Bob Gunton, William Sadler")

    def test_get_info(self):
        print("test get info")
        self.assertEqual(self.kino.get_info(), "Was released in 1994 by Frank Darabont")


# test mocking

if __name__ == "__main__":
    unittest.main()
