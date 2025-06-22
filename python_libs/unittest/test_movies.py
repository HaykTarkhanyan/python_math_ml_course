import unittest
import movies 

class TestMovies(unittest.TestCase):
    def test_get_actors(self):
        import json
        PATH = "imdb_top_999.json"

        with open(PATH, "r") as f:
            data = json.load(f)

        KEYS_TO_DELETE = ['Meta_score', "No_of_Votes", "Overview"]

        for i in data.keys():
            for k in KEYS_TO_DELETE: 
                del data[i][k]
            
        for i in data.keys():
            data[i]["Name"] = i

        kino = movies.Kino(**data["The Shawshank Redemption"])
        self.assertEqual(kino.get_actors(), "Tim Robbins, Morgan Freeman, Bob Gunton, William Sadler")

    def test_get_info(self):
        import json
        PATH = "imdb_top_999.json"

        with open(PATH, "r") as f:
            data = json.load(f)

        KEYS_TO_DELETE = ['Meta_score', "No_of_Votes", "Overview"]

        for i in data.keys():
            for k in KEYS_TO_DELETE: 
                del data[i][k]
            
        for i in data.keys():
            data[i]["Name"] = i

        kino = movies.Kino(**data["The Shawshank Redemption"])
        self.assertEqual(kino.get_info(), "Was released in 1994 by Frank Darabont")
    

class TestMovies(unittest.TestCase):
    def setUp(self):
        print("setUp")
        import json
        PATH = "imdb_top_999.json"

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
        print("tearDown\n")
        del self.kino

    def test_get_actors(self):
        print("test_get_actors")
        self.assertEqual(self.kino.get_actors(), "Tim Robbins, Morgan Freeman, Bob Gunton, William Sadler")

    def test_get_info(self):
        print("test_get_info")
        self.assertEqual(self.kino.get_info(), "Was released in 1994 by Frank Darabont")
    
    def test_name(self):
        print("test_name")
        self.assertEqual(self.kino.name, "The Shawshank Redemption")



class TestMovies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("ամենասկիզբ\n")

    @classmethod
    def tearDownClass(cls):
        print("ամենավերջ")
    
    def setUp(self):
        print("setUp")
        import json
        PATH = "imdb_top_999.json"

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
        print("tearDown\n")
        del self.kino

    def test_get_actors(self):
        print("test_get_actors")
        self.assertEqual(self.kino.get_actors(), "Tim Robbins, Morgan Freeman, Bob Gunton, William Sadler")

    def test_get_info(self):
        print("test_get_info")
        self.assertEqual(self.kino.get_info(), "Was released in 1994 by Frank Darabont")
    
    def test_name(self):
        print("test_name")
        self.assertEqual(self.kino.name, "The Shawshank Redemption")



if __name__ == '__main__':
    unittest.main()