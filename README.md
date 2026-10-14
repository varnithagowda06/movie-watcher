# movie watcher
A beginner machine learning project for learning and implementing ML models using movie data

About:
I'm learning ML by implementing models myself and working out how they behave. It starts with a small movie dataset, and I'll grow it as I pick up new concepts.

Where it stands:
The first model is logistic regression. It predicts whether I'll like a movie, using only runtime and IMDb rating.
So far I've used Pandas to load and handle the data, split it into training and test sets, trained the model, made predictions, and looked at the prediction probabilities. On the test data it gets 83.33% accuracy.

I've also implemented Linear Regression to predict user ratings based on runtime and IMDb rating. On my current test set, it achieved an MAE of approximately 0.128 and an R² score of 0.956.
I experimented with Decision Tree Classification to predict whether a movie would be liked or disliked. After limiting the tree's depth, it achieved 83.33% test accuracy, with 89.58% training accuracy.

I've also built a Streamlit app called **MovieMatch**, where users can filter movies by genre, runtime, and IMDb rating. The app displays the evaluation results of my Linear Regression and Logistic Regression models.
I'm continuing to explore machine learning algorithms, compare their performance, and improve the project as I learn.

Dataset:
Each row has the movie, genre, runtime, IMDb rating, my rating, and whether I liked it. The current model only uses runtime and IMDb rating.

Tools:
Python, Pandas, Scikit-learn

Next:
I'll add more models as I learn them, try more features, and compare how the models do against each other.

Status:
Work in progress.