import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

# Load data
full_df = pd.read_csv("full_merged_cleaned.csv")

# Preprocessing
ratings_per_user = full_df.groupby("userId")["rating"].count()
top_tags = full_df['tag'].value_counts().head(20)
full_df['genre_list'] = full_df['genres'].str.split('|')
genre_exploded = full_df.explode('genre_list')
top_genres = genre_exploded['genre_list'].value_counts().head(15)

# User-item matrix
sample_df = full_df[['userId', 'movieId', 'rating']].sample(n=10000, random_state=42)
user_item_matrix = sample_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

# Clustering
svd = TruncatedSVD(n_components=20, random_state=42)
user_features = svd.fit_transform(user_item_matrix)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(user_features)
user_clusters = pd.DataFrame({'userId': user_item_matrix.index, 'cluster': clusters})

# Streamlit Layout
st.set_page_config(page_title="🎬 Movie Insights Dashboard", layout="wide")

st.title("🎬 Interactive Movie Insights Dashboard")

tabs = st.tabs(["📊 Ratings per User", "🏷️ Top Tags", "🎞️ Top Genres", "🧠 User Cluster Distribution", "🎯 Top Genres by Cluster", "👤 User Ratings by Cluster"])

with tabs[0]:
    st.header("Ratings per User")
    fig = px.histogram(ratings_per_user, nbins=30, title='Number of Ratings per User',
                       labels={'value': 'Ratings', 'count': 'Users'},
                       color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig)
    st.info("This histogram shows how many movies each user has rated. Most users rate only a few.")

with tabs[1]:
    st.header("Top 20 Tags Used")
    fig = px.bar(x=top_tags.values, y=top_tags.index, orientation='h',
                 title='Top 20 Tags Used', labels={'x': 'Frequency', 'y': 'Tags'},
                 color=top_tags.values, color_continuous_scale='Blues')
    st.plotly_chart(fig)
    st.info("Tags reflect what users say about movies and are useful for understanding preferences.")

with tabs[2]:
    st.header("Top Genres Overall")
    fig = px.bar(x=top_genres.values, y=top_genres.index, orientation='h',
                 title='Top Genres', labels={'x': 'Count', 'y': 'Genre'},
                 color=top_genres.values, color_continuous_scale='Viridis',
                 text=[f'{(v / top_genres.sum() * 100):.2f}%' for v in top_genres.values])
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig)
    st.info("Shows how frequently each genre appears and its popularity.")

with tabs[3]:
    st.header("User Cluster Distribution")
    fig = px.histogram(user_clusters.astype({'cluster': str}), x='cluster',
                       title='User Distribution by Cluster', labels={'cluster': 'Cluster ID'},
                       color='cluster', color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig)
    st.info("Each cluster groups users with similar rating behaviors.")

with tabs[4]:
    st.header("Top Genres by Cluster")
    selected_cluster = st.selectbox("Choose a Cluster", sorted(user_clusters['cluster'].unique()))
    clustered_ratings = sample_df.merge(user_clusters, on='userId')
    metadata = full_df[['movieId', 'genres']].drop_duplicates()
    clustered_genres = clustered_ratings.merge(metadata, on='movieId')
    clustered_genres['genre_list'] = clustered_genres['genres'].str.split('|')
    exploded = clustered_genres.explode('genre_list')
    filtered = exploded[exploded['cluster'] == selected_cluster]
    genre_counts = filtered['genre_list'].value_counts().head(10)
    fig = px.bar(x=genre_counts.values, y=genre_counts.index, orientation='h',
                 title=f'Top Genres in Cluster {selected_cluster}', labels={'x': 'Count', 'y': 'Genre'},
                 color=genre_counts.values, color_continuous_scale='Sunset',
                 text=[f'{(v / genre_counts.sum() * 100):.2f}%' for v in genre_counts.values])
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig)
    st.success(f"Users in Cluster {selected_cluster} mostly watch these genres. Useful for personalized suggestions.")

with tabs[5]:
    st.header("Movie Ratings by Cluster & User")
    cluster_choice = st.selectbox("Select Cluster", sorted(user_clusters['cluster'].unique()))
    user_options = user_clusters[user_clusters['cluster'] == cluster_choice]['userId'].tolist()
    user_choice = st.selectbox("Select User from Cluster", user_options)
    
    user_data = sample_df[sample_df['userId'] == user_choice].merge(full_df[['movieId', 'title']].drop_duplicates(), on='movieId')
    fig = px.bar(user_data, x='title', y='rating', title=f"Ratings by User {user_choice}", labels={'title': 'Movie', 'rating': 'Rating'},
                 color_discrete_sequence=['lightblue'])
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig)
    st.info("This shows the movies rated by the selected user in the chosen cluster.")