import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
df_uncleaned = pd.read_csv('uncleaned.csv')
html_title = """<h1 style="color:red;text-align:center;">Movies Dataset</h1>"""
st.markdown(html_title,unsafe_allow_html=True)
st.image('dataset-cover.jpeg')
columns_description = [
    "id: A unique identifier for each movie.",
    "title: The name/title of the movie.",
    "year: The year the movie was released.",
    "duration: The runtime of the movie (usually in minutes).",
    "MPA: The MPA (Motion Picture Association) rating (PG, PG-13, R).",
    "rating: The average user rating (e.g., IMDb rating) for the movie.",
    "votes: The number of user votes that contributed to the movie rating.",
    "meta_score: The Metacritic score (critic-based rating),usually out of 100.",
    "description: A short plot summary or description of the movie.",
    "Movie_Link: URL link to the movie’s ",
    "writers: Names of people who wrote the movie .",
    "directors: Names of the directors of the movie.",
    "stars: Main cast or lead actors of the movie.",
    "budget: The estimated budget spent to produce the movie (in USD).",
    "opening_weekend_gross: The money the movie made during opening weekend",
    "gross_worldwide: Total worldwide box office earnings (in USD).",
    "gross_us_canada: Total earnings in the US and Canada only (in USD).",
    "release_date: The date the movie was first released in theaters.",
    "countries_origin: The countries that produced or co-produced the movie.",
    "filming_locations: The locations where the movie was filmed.",
    "production_companies: The companies that produced or funded the movie.",
    "awards_content: Awards won or nominated (Oscars, etc.).",
    "genres: Movie genres (e.g., Action, Drama, Comedy).",
    "languages: Languages spoken in the movie."
]

tab1, tab2, tab3, tab4 = st.tabs(["Dataset uncleaned","Dataset cleaned", "Visualizations", "About"])
with tab1:
    st.subheader("Dataset uncleaned")
    st.dataframe(df_uncleaned.head(5))
    st.subheader("Dataset uncleaned descriptive statistics")
    st.write(df_uncleaned.describe())
    st.subheader("Dataset uncleaned missing values")
    st.write(df_uncleaned.isnull().mean().round(4) * 100)
    st.write("Number of rows:", df_uncleaned.shape[0])
    st.write("Number of columns:", df_uncleaned.shape[1])
    st.write("Columns:", columns_description)

with tab2:
    st.subheader("Dataset cleaned")
    df = pd.read_csv('final_dataset_cleaned.csv')
    st.dataframe(df)
    st.subheader("Dataset cleaned descriptive statistics")
    st.write(df.describe())
    st.subheader("Dataset cleaned missing values")
    st.write(df.isnull().mean().round(4) * 100)
    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])
with tab3:
    st.subheader("Visualizations")
    st.write("Distribution of Ratings")
    fig1 = px.histogram(df, x='rating', title='Distribution of Ratings')
    st.plotly_chart(fig1)

    st.write("Box Plot of Ratings by MPA")
    fig2 = px.box(df, y='rating', color='MPA', title='Box Plot of Ratings by MPA')
    st.plotly_chart(fig2)

    st.write("Correlation Heatmap")
    fig3 = px.imshow(df.corr(numeric_only=True).round(2), height=800, width=800, title='Correlation Heatmap')
    st.plotly_chart(fig3)

    st.write("Scatter Plot of Ratings vs Votes")
    fig4 = px.scatter(df, x='rating', y='votes', title='Scatter Plot of Ratings vs Votes')  
    st.plotly_chart(fig4)

    st.write("Scatter Plot of Votes vs Duration")
    fig5 = px.scatter(df, x='votes', y='duration', title='Scatter Plot of Votes vs Duration')   
    st.plotly_chart(fig5)

    st.write("Scatter Matrix of Numeric Features")
    fig6 = px.scatter_matrix(df.select_dtypes(include='number'), height=1400, width=1400, title='Scatter Matrix of Numeric Features')
    st.plotly_chart(fig6)

    st.write("Top 10 Genres Distribution")
    df['genres'] = df['genres'].apply(eval)
    top_genres = df['genres'].explode().value_counts().head(10)
    fig7 = px.bar(top_genres, x=top_genres.index, y=top_genres.values, title='Top 10 Genres Distribution')
    st.plotly_chart(fig7)

    st.write("Top 10 Countries of Origin Distribution")
    df['countries_origin'] = df['countries_origin'].apply(eval)
    df_countries = df.explode('countries_origin')
    country_counts = df_countries['countries_origin'].value_counts().sort_values(ascending=False).head(10)
    fig8 = px.pie(country_counts, names=country_counts.index, values=country_counts.values, title='Countries of Origin Distribution')
    st.plotly_chart(fig8)

    st.write("Average Rating by MPA")
    avg_per_mpa = df.groupby('MPA')['rating'].mean().round(1).sort_values(ascending=False).reset_index()
    fig9 = px.bar(avg_per_mpa, x='MPA', y='rating', title='Average Rating by MPA')
    st.plotly_chart(fig9)

    st.write("Top 10 Directors by Number of Movies")
    directors_movies_count = df.groupby('directors')['title'].count().sort_values(ascending=False).head(10)
    fig10 = px.bar(directors_movies_count, x=directors_movies_count.index, y=directors_movies_count.values, title='Top 10 Directors by Number of Movies')
    st.plotly_chart(fig10)

    st.write("Top Movies by Rating")
    movies_count = df.groupby('title')['id'].count().sort_values(ascending=False).head(10)
    top_movies = movies_count.index.tolist()
    fig11 = px.bar(df[df['title'].isin(top_movies)], x='title', y='rating', color='MPA', title='Top Movies by Rating')
    st.plotly_chart(fig11)

    st.write("Top Directors by Rating")
    top_directors = directors_movies_count.index.tolist()
    fig12 = px.bar(df[df['directors'].isin(top_directors)], x='directors', y='rating', color='MPA', title='Top Directors by Rating')
    st.plotly_chart(fig12)

    st.write("Directors Distribution")
    df['directors'] = df['directors'].apply(eval)
    df_directors = df.explode('directors')
    director_counts = df_directors['directors'].value_counts().sort_values(ascending=False).head(10)
    fig13 = px.pie(director_counts, names=director_counts.index, values=director_counts.values, title='Directors Distribution')
    st.plotly_chart(fig13)

    st.write("Top 10 Movies for Each of the 10 Most Prolific Years")
    top_years = df['year'].value_counts().sort_values(ascending=False).head(10).index
    df_top_years = df[df['year'].isin(top_years)]
    top_10_movies = []
    for year, group in df_top_years.groupby('year'):
        top_movies = group.sort_values(by='rating', ascending=False).head(10)
        top_10_movies.append(top_movies)
    top_10 = pd.concat(top_10_movies).reset_index(drop=True)
    plt.figure(figsize=(20, 20))
    sns.boxplot(data=top_10, x='year', y='rating')
    plt.title('Top 10 Movies for Each of the 10 Most Prolific Years')
    st.pyplot(plt)
    st.write("Top Writers by Rating")
    writers_movies_count = df.groupby('writers')['title'].count().sort_values(ascending=False).head(10)
    top_writers = writers_movies_count.index.tolist()

    fig14 = px.bar(df[df['writers'].isin(top_writers)], x='writers', y='rating', color='MPA', title='Top Writers by Rating')
    st.plotly_chart(fig14)
    st.write("Writers Distribution")
    df['writers'] = df['writers'].apply(eval)
    df_writers = df.explode('writers')
    writer_counts = df_writers['writers'].value_counts().sort_values(ascending=False).head(10)

    fig15 = px.pie(writer_counts, names=writer_counts.index, values=writer_counts.values, title='Writers Distribution')
    st.plotly_chart(fig15)
with tab4:
    st.subheader("About")
    st.write("This project is a comprehensive analysis of a movie dataset, focusing on various aspects such as ratings, directors, genres, and more. The dataset has been cleaned and visualized using Streamlit, Plotly, and Matplotlib.")
    st.write("The goal is to provide insights into the movie industry, including trends in ratings, popular genres, and the impact of directors and writers on movie success.")
    st.write("The dataset includes information on over 10,000 movies from various countries and genres, making it a rich source for analysis.")
