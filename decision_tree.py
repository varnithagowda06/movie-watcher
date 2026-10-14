import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

movies = pd.read_csv("movies.csv")
print(movies.columns)
X = movies[["runtime", "imdb_rating"]]
y = movies["liked"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=45
)
X = movies[["runtime", "imdb_rating"]]
y = movies["liked"]

model = DecisionTreeClassifier(max_depth=3,random_state=45)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nPredicted values:")
print(y_pred)
print("\nActual values:")
print(y_test.to_numpy())
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Training accuracy:", accuracy_score(y_train, model.predict(X_train)))
print("Testing accuracy:", accuracy_score(y_test, y_pred))