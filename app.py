import streamlit as st
import pandas as pd

st.title("🎬 MovieMatch")
st.write("Find movies you might enjoy!")

movies = pd.read_csv("movies.csv")

genre = st.selectbox(
    "Choose a genre",
    movies["genre"].unique()
)

runtime = st.slider(
    "Maximum runtime (minutes)",
    60,
    200,
    140
)

imdb = st.slider(
    "Minimum IMDb rating",
    5.0,
    9.0,
    7.5
)

if st.button("Find Movies"):
    recommendations = movies[
        (movies["genre"] == genre) &
        (movies["runtime"] <= runtime) &
        (movies["imdb_rating"] >= imdb)
    ]

    st.subheader("🎯 Your Matches")

    if len(recommendations) > 0:
        st.dataframe(recommendations)
    else:
        st.write("No movies matched your preferences.")

st.header("Your Movie Preferences")

genre = st.selectbox(
    "Preferred Genre",
    ["Comedy", "Action", "Romance", "Sci-Fi", "Thriller", "Animation"]
)

runtime = st.slider(
    "Preferred Runtime (minutes)",
    60,
    210,
    140,
    10
)

min_rating = st.slider(
    "Minimum IMDb Rating",
    5.0,
    9.5,
    7.5,
    0.5
)

if st.button("Find Movies"):
    st.write("Your preferences:")
    st.write("Genre:", genre)
    st.write("Runtime:", runtime, "minutes")
    st.write("Minimum IMDb Rating:", min_rating)