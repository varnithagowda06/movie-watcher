import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

movies = pd.read_csv("movies.csv")
print(movies.head())
print(movies.columns) 
X = movies[["runtime", "imdb_rating"]]
y = movies["user_rating"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45
)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Actual ratings:", list(y_test))
print("Predicted ratings:", predictions)
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))
print("Intercept:", model.intercept_)
print("Runtime coefficient:", model.coef_[0])
print("IMDb rating coefficient:", model.coef_[1])

new_movie = pd.DataFrame({
    "runtime": [120],
    "imdb_rating": [8.0]
})
predicted_rating = model.predict(new_movie)
print("Predicted user rating:", predicted_rating[0])