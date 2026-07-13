import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Coursera Course Recommendation System")

df = pd.read_csv("cleaned_course_data.csv")

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(df["tags"]).toarray()

similarity = cosine_similarity(vectors)

course_list = df["course_title"].values

selected_course = st.selectbox(
    "Select Course",
    course_list
)

if st.button("Recommend"):

    index = df[df["course_title"] == selected_course].index[0]

    distances = similarity[index]

    recommended = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    st.subheader("Recommended Courses")

    for i in recommended:
        st.write(df.iloc[i[0]].course_title)