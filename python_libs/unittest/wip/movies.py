import json
import os

# glob - for working with paths

PATH = os.path.join("imdb_top_999.json")

with open(PATH, "r") as f:
    data = json.load(f)

KEYS_TO_DELETE = ['Meta_score', "No_of_Votes", "Overview"]

for i in data.keys():
    for k in KEYS_TO_DELETE: 
        del data[i][k]
    
for i in data.keys():
    data[i]["Name"] = i


class Kino:
    def __init__(self, Name, Released_Year, Runtime, Genre, IMDB_Rating, Director,
                Star1, Star2, Star3, Star4, Gross):
        self.name = Name
        self.release_year = Released_Year
        self.runtime = Runtime
        self.genre = Genre
        self.rating = IMDB_Rating
        self.director = Director
        self.star1 = Star1
        self.star2 = Star2
        self.star3 = Star3
        self.star4 = Star4
        self.gross = Gross
    
    def get_actors(self):
        self.star_string = ", ".join([self.star1, self.star2, self.star3, self.star4])
        return self.star_string

    def get_info(self):
        self.info = f"Was released in {self.release_year} by {self.director}"
        return self.info

movies_list = []
for i in data.keys():
    kino = data[i]
    kino_class = Kino(**kino)
    movies_list.append(kino_class)

movies_list = [Kino(**data[i]) for i in data.keys()]

if __name__=="__main__":
    print(movies_list[0].get_info())