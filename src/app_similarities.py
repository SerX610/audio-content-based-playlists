"""
This module provides a Streamlit-based demo to find the most similar tracks
to a query track using `effnet-discogs` and `msd-musiccnn` embeddings. It
computes cosine similarity between the query track's embeddings and all other
tracks, excluding the query track itself. The top 10 similar tracks for each
embedding type are displayed with audio previews.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

import utils as u

# pylint: disable=missing-function-docstring
# pylint: disable=too-many-locals


# Path to the audio analysis data
AUDIO_ANALYSIS_PATH = "data/musav_analysis.json"

# Path to save the M3U playlists
M3U_DISCOGS_FILEPATHS_FILE = "playlists/similarities_playlist_discogs.m3u8"
M3U_MUSICCNN_FILEPATHS_FILE = "playlists/similarities_playlist_musiccnn.m3u8"


def compute_similarity(query_embedding, embeddings):
    """
    Compute cosine similarity between the query embedding and all other
    embeddings.

    Args:
        query_embedding (np.array): Embedding of the query track.
        embeddings (list): List of embeddings for all tracks.

    Returns:
        np.array: Array of cosine similarity scores.
    """
    return cosine_similarity([query_embedding], embeddings)[0]


def display_track_info(track, similarity_score, embedding_type):
    """
    Display track information and audio preview.

    Args:
        track (pd.Series): The track data (row from DataFrame).
        similarity_score (float): The similarity score for the track.
        embedding_type (str): The type of embedding used
            (e.g., 'effnet-discogs').
    """
    st.write(f"**Track ID:** {track['id']}")
    st.write(
        f"**Similarity Score ({embedding_type}):** {similarity_score:.4f}"
    )
    st.write(f"**Path:** {track['path']}")
    st.audio(track["path"], format="audio/mp3", start_time=0)
    st.write("---")


def get_top_similar_tracks(
    collection_df, query_embedding, embedding_column, top_n=10
):
    """
    Get the top N most similar tracks based on the given embedding type.

    Args:
        collection_df (pd.DataFrame): The DataFrame containing all tracks.
        query_embedding (np.array): The embedding of the query track.
        embedding_column (str): The column name for the embeddings
            (e.g., 'discogs').
        top_n (int): Number of top similar tracks to return.

    Returns:
        pd.DataFrame: DataFrame containing the top N similar tracks.
    """
    embeddings = (
        collection_df["embeddings"]
        .apply(lambda x: np.array(x[embedding_column]))
        .tolist()
    )
    similarity_scores = compute_similarity(query_embedding, embeddings)
    collection_df = collection_df.copy().assign(
        similarity_score=similarity_scores
    )
    return collection_df.sort_values("similarity_score", ascending=False).head(
        top_n
    )


def save_playlist(tracks, filepath):
    """
    Save the list of track paths to an M3U playlist file.

    Args:
        tracks (list): List of track paths to include in the playlist.
        filepath (str): Path to save the M3U playlist file.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(f"../{track}" for track in tracks))
    st.success(f"Playlist saved to `{filepath}`.")


def main():

    st.write("# Track Similarity Demo")
    st.write(
        "Find the most similar tracks to a query track using `effnet-discogs` \
            and `msd-musiccnn` embeddings."
    )

    # Define variables
    audio_analysis_path = AUDIO_ANALYSIS_PATH
    m3u_discogs_filepaths_file = M3U_DISCOGS_FILEPATHS_FILE
    m3u_musiccnn_filepaths_file = M3U_MUSICCNN_FILEPATHS_FILE

    # Load audio analysis data
    collection = u.load_data_from_json(audio_analysis_path)

    # Convert collection data into pandas DataFrame
    collection_df = pd.DataFrame(collection)

    # Clean paths and extract track IDs
    collection_df["path"] = collection_df["path"].apply(
        lambda x: x.lstrip("../")
    )
    collection_df["id"] = collection_df["path"].apply(
        lambda x: x.split("/")[-1].split(".")[0]
    )

    # Display the list of track IDs for selection
    track_ids = collection_df["id"].tolist()
    selected_track_id = st.selectbox("Select a query track by ID:", track_ids)

    # Get the query track's embeddings
    query_track = collection_df[collection_df["id"] == selected_track_id].iloc[
        0
    ]
    query_discogs_embedding = np.array(query_track["embeddings"]["discogs"])
    query_musiccnn_embedding = np.array(query_track["embeddings"]["musiccnn"])

    # Exclude the query track from the dataset
    collection_df_filtered = collection_df[
        collection_df["id"] != selected_track_id
    ]

    # Get the top 10 most similar tracks for each embedding type
    top_discogs_tracks = get_top_similar_tracks(
        collection_df_filtered,
        query_discogs_embedding,
        "discogs",
    )
    top_musiccnn_tracks = get_top_similar_tracks(
        collection_df_filtered,
        query_musiccnn_embedding,
        "musiccnn",
    )

    # Display the query track
    st.write("## Query Track")
    st.write(f"**Track ID:** {query_track['id']}")
    st.write(f"**Path:** {query_track['path']}")
    st.audio(query_track["path"], format="audio/mp3", start_time=0)

    # Display the top 10 similar tracks using effnet-discogs
    st.write("## Top 10 Similar Tracks (effnet-discogs)")
    for _, track in top_discogs_tracks.iterrows():
        display_track_info(track, track["similarity_score"], "effnet-discogs")

    # Display the top 10 similar tracks using msd-musiccnn
    st.write("## Top 10 Similar Tracks (msd-musiccnn)")
    for _, track in top_musiccnn_tracks.iterrows():
        display_track_info(track, track["similarity_score"], "msd-musiccnn")

    # Save effnet-discogs playlist
    discogs_playlist_paths = top_discogs_tracks["path"].tolist()
    save_playlist(discogs_playlist_paths, m3u_discogs_filepaths_file)

    # Save msd-musiccnn playlist
    musiccnn_playlist_paths = top_musiccnn_tracks["path"].tolist()
    save_playlist(musiccnn_playlist_paths, m3u_musiccnn_filepaths_file)


if __name__ == "__main__":
    main()
