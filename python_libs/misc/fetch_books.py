import pandas as pd 

PATH = "https://raw.githubusercontent.com/scostap/goodreads_bbe_dataset/main/Best_Books_Ever_dataset/books_1.Best_Books_Ever.csv"

df = pd.read_csv(PATH, nrows=1000)

print(df.head())