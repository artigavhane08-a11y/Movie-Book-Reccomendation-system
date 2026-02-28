import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv("movies.csv")
books = pd.read_csv("books.csv" ,
                    on_bad_lines='skip',
                    encoding='latin-1')

# ---------- MOVIE RECOMMENDATION ----------

movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)

movie_vectorizer = CountVectorizer(stop_words='english')
movie_matrix = movie_vectorizer.fit_transform(movies['genres'])
movie_similarity = cosine_similarity(movie_matrix)

def recommend_movie(movie_name):
    movie_name = movie_name.lower()
    if movie_name not in movies['title'].str.lower().values:
        print("Movie not found!")
        return

    idx = movies[movies['title'].str.lower() == movie_name].index[0]
    scores = list(enumerate(movie_similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    print("\nRecommended Movies:")
    for i in scores:
        print(movies.iloc[i[0]]['title'])


# ---------- BOOK RECOMMENDATION ----------

book_vectorizer = CountVectorizer(stop_words='english')
book_matrix = book_vectorizer.fit_transform(books['title'])
book_similarity = cosine_similarity(book_matrix)

def recommend_book(book_name):
    book_name = book_name.lower()
    if book_name not in books['title'].str.lower().values:
        print("Book not found!")
        return

    idx = books[books['title'].str.lower() == book_name].index[0]
    scores = list(enumerate(book_similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    print("\nRecommended Books:")
    for i in scores:
        print(books.iloc[i[0]]['title'])


# ---------- USER INPUT ----------

choice = input("Are you interested in movie or book? ").lower()

if choice == "movie":
    name = input("Enter movie name: ")
    recommend_movie(name)

elif choice == "book":
    name = input("Enter book name: ")
    recommend_book(name)

else:
    print("Invalid choice!")
