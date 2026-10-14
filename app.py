
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

st.set_page_config(
    page_title="MovieMatch",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
</style>
""", unsafe_allow_html=True)

movies = pd.read_csv("movies.csv")

st.title("🎬 MovieMatch")
st.write("Discover movies you might enjoy, powered by machine learning.")
st.caption("A hands-on project exploring movie ratings and preferences.")

required_columns = {
    "movie", "genre", "runtime", "imdb_rating", "user_rating", "liked"
}

if not required_columns.issubset(movies.columns):
    st.error("Please check that movies.csv contains all the required columns.")
    st.stop()

if movies[list(required_columns)].isnull().any().any():
    st.error("Please handle missing values in the required dataset columns.")
    st.stop()

X = movies[["runtime", "imdb_rating"]]
y_rating = movies["user_rating"]
y_liked = movies["liked"]

X_train, X_test, y_rating_train, y_rating_test, y_liked_train, y_liked_test = (
    train_test_split(
        X,
        y_rating,
        y_liked,
        test_size=0.2,
        random_state=42
    )
)

linear_model = LinearRegression()
linear_model.fit(X_train, y_rating_train)

rating_predictions = linear_model.predict(X_test)

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_liked_train)

like_predictions = logistic_model.predict(X_test)

mae = mean_absolute_error(y_rating_test, rating_predictions)
r2 = r2_score(y_rating_test, rating_predictions)
accuracy = accuracy_score(y_liked_test, like_predictions)

st.divider()
st.subheader("Machine Learning Models")

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader("01 · Linear Regression")
        st.write("Predict numerical user ratings from movie features.")

        metric1, metric2 = st.columns(2)
        metric1.metric("Mean Absolute Error", f"{mae:.3f}")
        metric2.metric("R² Score", f"{r2:.3f}")

        st.caption("Inputs: Runtime + IMDb rating")
        st.caption("Target: User rating")

with col2:
    with st.container(border=True):
        st.subheader("02 · Logistic Regression")
        st.write("Predict whether a movie will be liked or disliked.")

        st.metric("Test Accuracy", f"{accuracy:.2%}")

        st.caption("Inputs: Runtime + IMDb rating")
        st.caption("Target: Liked / Disliked")

st.divider()
st.header("Find Your Movie")

genres = sorted(movies["genre"].unique().tolist())

runtime_min = int(movies["runtime"].min())
runtime_max = int(movies["runtime"].max())

rating_min = float(movies["imdb_rating"].min())
rating_max = float(movies["imdb_rating"].max())

if len(genres) == 0 or runtime_min == runtime_max or rating_min == rating_max:
    st.error("The dataset needs multiple genres and a range of movie features.")
    st.stop()

genre = st.selectbox("Choose your preferred genre", genres)

col3, col4 = st.columns(2)

with col3:
    runtime = st.slider(
        "Maximum runtime (minutes)",
        min_value=runtime_min,
        max_value=runtime_max,
        value=min(140, runtime_max),
        step=1
    )

with col4:
    imdb = st.slider(
        "Minimum IMDb rating",
        min_value=float(rating_min),
        max_value=float(rating_max),
        value=float(min(max(7.5, rating_min), rating_max)),
        step=0.1
    )

if st.button("Find Movies", type="primary"):
    recommendations = movies[
        (movies["genre"] == genre)
        & (movies["runtime"] <= runtime)
        & (movies["imdb_rating"] >= imdb)
    ]

    st.subheader("Your Matches")

    if not recommendations.empty:
        st.success(f"Found {len(recommendations)} matching movies!")

        st.dataframe(
            recommendations[
                ["movie", "genre", "runtime", "imdb_rating", "user_rating"]
            ],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No movies matched your preferences. Try adjusting the filters.")

st.divider()

with st.expander("Explore the test-set predictions"):
    results = pd.DataFrame({
        "Movie": movies.loc[X_test.index, "movie"].to_numpy(),
        "Actual User Rating": y_rating_test.to_numpy(),
        "Predicted User Rating": rating_predictions.round(2),
        "Actual Preference": y_liked_test.to_numpy(),
        "Predicted Preference": like_predictions
    })

    st.dataframe(results, use_container_width=True, hide_index=True)

st.caption("Built with Python · Pandas · Scikit-learn · Streamlit")
