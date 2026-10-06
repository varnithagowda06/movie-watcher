import pandas as pd

movies = pd.read_csv("movies.csv")
print(movies)
print(movies.columns)

X = movies[["runtime", "imdb_rating"]]
y = movies["liked"]
print(X)
print(y)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=45
)

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
probabilities = model.predict_proba(X_test)
print(probabilities)