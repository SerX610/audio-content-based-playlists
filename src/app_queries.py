"""
This module provides a Streamlit-based demo to filter and rank audio tracks
based on various attributes such as style activations, tempo,
voice/instrumental, danceability, arousal/valence, and key/scale. It allows
users to interactively select filters, rank tracks, and generate playlists.
The results can be saved as an M3U playlist for further use.
"""

import random

import pandas as pd
import streamlit as st

import utils as u

# pylint: disable=missing-function-docstring
# pylint: disable=too-many-statements
# pylint: disable=cell-var-from-loop
# pylint: disable=too-many-locals
# pylint: disable=broad-exception-caught


# Path to the audio analysis data
AUDIO_ANALYSIS_PATH = "data/musav_analysis.json"

# Path to the Discogs metadata file
STYLE_METADATA_FILE_PATH = "data/discogs-effnet-bs64-1.json"

# Path to save the M3U playlist
M3U_FILEPATHS_FILE = "playlists/query_playlist.m3u8"


def filter_by_style(
    collection_df, style_select, style_select_range, music_style_names
):
    """
    Filter the audio analysis DataFrame based on selected styles and their
    activation ranges.

    Args:
        collection_df (pd.DataFrame): The DataFrame containing audio analysis
            data.
        style_select (list): List of selected styles.
        style_select_range (list): Range of activations for the selected
            styles.
        music_style_names (list): List of all available music style names.

    Returns:
        pd.DataFrame: Filtered DataFrame based on style activations.
    """
    if style_select:
        selected_style_indexes = [
            music_style_names.index(style) for style in style_select
        ]
        collection_df = collection_df[
            collection_df["music_styles_activations"].apply(
                lambda x: all(
                    style_select_range[0] <= x[i] <= style_select_range[1]
                    for i in selected_style_indexes
                )
            )
        ]
    return collection_df


def filter_by_tempo(collection_df, tempo_range):
    """
    Filter the audio analysis DataFrame based on tempo range.

    Args:
        collection_df (pd.DataFrame): The DataFrame containing audio analysis
            data.
        tempo_range (tuple): Minimum and maximum tempo values.

    Returns:
        pd.DataFrame: Filtered DataFrame based on tempo range.
    """
    tempo_min, tempo_max = tempo_range
    tempo_activations = collection_df["tempo"]
    return collection_df[
        (tempo_activations >= tempo_min) & (tempo_activations <= tempo_max)
    ]


def filter_by_voice_instrumental(collection_df, voice_instrumental):
    """
    Filter the audio analysis DataFrame based on voice or instrumental
    activations.

    Args:
        collection_df (pd.DataFrame): The DataFrame containing audio analysis
            data.
        voice_instrumental (str): Selected option ('All', 'Voice', or
            'Instrumental').

    Returns:
        pd.DataFrame: Filtered DataFrame based on voice/instrumental
            activations.
    """
    instrumental_activations = collection_df[
        "voice_instrumental_activations"
    ].str[0]
    voice_activations = collection_df["voice_instrumental_activations"].str[1]
    if voice_instrumental == "Instrumental":
        return collection_df[instrumental_activations > 0.5]
    if voice_instrumental == "Voice":
        return collection_df[voice_activations > 0.5]
    return collection_df


def filter_by_danceability(collection_df, danceability_range):
    """
    Filter the audio analysis DataFrame based on danceability range.

    Args:
        collection_df (pd.DataFrame): The DataFrame containing audio analysis
            data.
        danceability_range (tuple): Minimum and maximum danceability values.

    Returns:
        pd.DataFrame: Filtered DataFrame based on danceability range.
    """
    danceability_min, danceability_max = danceability_range
    danceability_activations = collection_df["danceability_activations"].str[0]
    return collection_df[
        (danceability_activations >= danceability_min)
        & (danceability_activations <= danceability_max)
    ]


def filter_by_arousal_valence(collection_df, valence_range, arousal_range):
    """
    Filter the audio analysis DataFrame based on arousal and valence ranges.

    Args:
        collection_df (pd.DataFrame): The DataFrame containing audio analysis
            data.
        valence_range (tuple): Minimum and maximum valence values.
        arousal_range (tuple): Minimum and maximum arousal values.

    Returns:
        pd.DataFrame: Filtered DataFrame based on arousal and valence ranges.
    """
    valence_min, valence_max = valence_range
    arousal_min, arousal_max = arousal_range
    valence_activations = collection_df["arousal_valence_activations"].str[0]
    arousal_activations = collection_df["arousal_valence_activations"].str[1]
    return collection_df[
        (valence_activations >= valence_min)
        & (valence_activations <= valence_max)
        & (arousal_activations >= arousal_min)
        & (arousal_activations <= arousal_max)
    ]


def filter_by_key_scale(
    collection_df, selected_key, selected_scale, key_profile="temperley"
):
    """
    Filter the audio analysis DataFrame based on key and scale.

    Args:
        collection_df (pd.DataFrame): The DataFrame containing audio analysis
            data.
        selected_key (str): Selected key (e.g., 'C', 'D#').
        selected_scale (str): Selected scale ('major', 'minor', or 'Any').
        key_profile (str): Key profile type (default is 'temperley').

    Returns:
        pd.DataFrame: Filtered DataFrame based on key and scale.
    """
    if selected_key != "Any":
        return collection_df[
            collection_df["key"].apply(
                lambda x: any(
                    k["profile_type"] == key_profile
                    and k["key"].startswith(selected_key)
                    for k in x
                )
            )
        ]
    if selected_scale != "Any":
        return collection_df[
            collection_df["key"].apply(
                lambda x: any(
                    k["profile_type"] == key_profile
                    and k["key"].endswith(selected_scale)
                    for k in x
                )
            )
        ]
    return collection_df


def rank_by_style_activations(collection_df, style_rank, music_style_names):
    """
    Rank the audio analysis DataFrame based on selected style activations.

    Args:
        collection_df (pd.DataFrame): The DataFrame containing audio analysis
            data.
        style_rank (list): List of styles to rank by.
        music_style_names (list): List of all available music style names.

    Returns:
        pd.DataFrame: Ranked DataFrame with 'rank' and
            'music_styles_activations' columns.
    """
    if style_rank:
        style_rank_indexes = [
            music_style_names.index(style) for style in style_rank
        ]
        audio_analysis_query = collection_df.copy()
        audio_analysis_query["rank"] = audio_analysis_query[
            "music_styles_activations"
        ].apply(lambda x: x[style_rank_indexes[0]])
        for idx in style_rank_indexes[1:]:
            audio_analysis_query["rank"] *= audio_analysis_query[
                "music_styles_activations"
            ].apply(lambda x: x[idx])
        ranked = audio_analysis_query.sort_values(["rank"], ascending=[False])
        if "path" in ranked.columns:
            ranked = ranked.set_index("path")
        return ranked[["rank", "music_styles_activations"]]
    return pd.DataFrame()


def main():

    # ===========================
    # LOADING
    # ===========================
    st.write("# Audio Analysis Playlists")

    # Define variables
    audio_analysis_path = AUDIO_ANALYSIS_PATH
    style_metadata_file_path = STYLE_METADATA_FILE_PATH
    m3u_filepaths_file = M3U_FILEPATHS_FILE

    # Load audio analysis data
    collection = u.load_data_from_json(audio_analysis_path)

    # Convert collection data into pandas DataFrame
    collection_df = pd.DataFrame(collection)

    # Get music style names from Discogs metadata
    music_style_metadata = u.load_data_from_json(style_metadata_file_path)
    music_style_names = music_style_metadata["classes"]

    st.write(f"Using analysis data from `{audio_analysis_path}`.")
    st.write("Loaded audio analysis for", len(collection_df), "tracks.")

    # ===========================
    # FILTERING AND RANKING
    # ===========================
    st.write("## 🔍 Filter and Rank")

    # Filter by Style
    st.write("### By style")
    style_select = st.multiselect(
        "Select by style activations:", music_style_names
    )
    if style_select:
        style_select_str = ", ".join(style_select)
        style_select_range = st.slider(
            f"Select tracks with `{style_select_str}` \
                activations within range:",
            min_value=0.0,
            max_value=1.0,
            value=[0.5, 1.0],
        )
        collection_df = filter_by_style(
            collection_df, style_select, style_select_range, music_style_names
        )

    # Filter by Tempo
    st.write("### By Tempo")
    tempo_range = st.slider(
        "Select Tempo Range (BPM):",
        min_value=30,
        max_value=300,
        value=(30, 300),
        step=1,
    )
    collection_df = filter_by_tempo(collection_df, tempo_range)

    # Filter by Voice/Instrumental
    st.write("### By Voice/Instrumental")
    voice_instrumental = st.radio("Select:", ["All", "Voice", "Instrumental"])
    collection_df = filter_by_voice_instrumental(
        collection_df, voice_instrumental
    )

    # Filter by Danceability
    st.write("### By Danceability")
    danceability_range = st.slider(
        "Select Danceability Range:",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.01,
    )
    collection_df = filter_by_danceability(collection_df, danceability_range)

    # Filter by Arousal and Valence
    st.write("### By Arousal and Valence")
    valence_range = st.slider(
        "Select Valence Range:",
        min_value=1.0,
        max_value=9.0,
        value=(1.0, 9.0),
        step=0.1,
    )
    arousal_range = st.slider(
        "Select Arousal Range:",
        min_value=1.0,
        max_value=9.0,
        value=(1.0, 9.0),
        step=0.1,
    )
    collection_df = filter_by_arousal_valence(
        collection_df, valence_range, arousal_range
    )

    # Filter by Key/Scale
    st.write("### By Key/Scale")
    keys = [
        "Any",
        "C",
        "C#",
        "D",
        "Eb",
        "E",
        "F",
        "F#",
        "G",
        "Ab",
        "A",
        "Bb",
        "B",
    ]
    scales = ["Any", "major", "minor"]
    selected_key = st.radio("Select Key:", keys)
    selected_scale = st.radio("Select Scale:", scales)
    collection_df = filter_by_key_scale(
        collection_df, selected_key, selected_scale
    )

    # ===========================
    # RANKING BY STYLE ACTIVATIONS
    # ===========================
    st.write("## 🔝 Rank by Style Activations")
    style_rank = st.multiselect(
        "Rank by style activations \
            (multiplies activations for selected styles):",
        music_style_names,
        [],
    )
    ranked = rank_by_style_activations(
        collection_df, style_rank, music_style_names
    )
    if not ranked.empty:
        st.write("Applied ranking by audio style predictions.")
        st.write(ranked)

    # ===========================
    # POST-PROCESSING
    # ===========================
    st.write("## 🔀 Post-process")
    max_tracks = st.number_input(
        "Maximum number of tracks (0 for all):", value=0, min_value=0
    )
    shuffle = st.checkbox("Random shuffle")

    if st.button("RUN"):

        st.write("## 🔊 Results")

        # Get the list of audio paths
        mp3s = (
            list(collection_df["path"].str.lstrip("../"))
            if not collection_df.empty
            else []
        )

        if not mp3s:
            st.warning("No audio tracks found. Try adjusting your filters.")

        else:
            if max_tracks:
                mp3s = mp3s[:max_tracks]
                st.write("Using top", len(mp3s), "tracks from the results.")

            if shuffle:
                random.shuffle(mp3s)
                st.write("Applied random shuffle.")

            try:
                with open(m3u_filepaths_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(f"../{mp3}" for mp3 in mp3s))
                    st.write(f"Stored M3U playlist to `{m3u_filepaths_file}`.")
            except Exception as e:
                st.error(f"Failed to save M3U playlist: {e}")

            st.write("Audio previews for the first 10 results:")
            for mp3 in mp3s[:10]:
                st.audio(mp3, format="audio/mp3", start_time=0)


if __name__ == "__main__":
    main()
