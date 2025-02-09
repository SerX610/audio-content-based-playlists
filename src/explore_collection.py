import pandas as pd

import utils as u


# ===========================
# INPUT PATHS
# ===========================

# Path to the JSON file containing music collection data
COLLECTION_JSON_FILE_PATH = '../data/musav_analysis.json'

# Path to the Discogs metadata JSON file
STYLE_METADATA_FILE_PATH = '../data/discogs-effnet-bs64-1.json'

# ===========================
# OUTPUT PATHS
# ===========================

# Path to save the music style distribution TSV file
STYLE_DISTRIBUTION_TSV_PATH = "../results/music_style_distribution.tsv"

# Path to save the genre distribution plot
GENRE_DISTRIBUTION_PLOT_PATH = "../results/genres_distribution.png" 

# Path to save the tempo distribution plot
TEMPO_DISTRIBUTION_PLOT_PATH = "../results/tempo_distribution.png"

# Path to save the danceability distribution plot
DANCEABILITY_DISTRIBUTION_PLOT_PATH = "../results/danceability_distribution.png"

# Path to save the key distribution plot
KEY_DISTRIBUTION_PLOT_PATH = "../results/key_distribution.png"

# Path to save the loudness distribution plot
LOUDNESS_DISTRIBUTION_PLOT_PATH = "../results/loudness_distribution.png"

# Path to save the valence-arousal distribution plot
VALENCE_AROUSAL_DISTRIBUTION_PLOT_PATH = "../results/valence_arousal_distribution.png"

# Path to save the instrumental-voice distribution plot
INSTRUMENTAL_VOICE_DISTRIBUTION_PLOT_PATH = "../results/instrumental_voice_distribution.png"


def get_music_styles(collection_df, style_metadata_file_path):
    """
    Determine the music style for each track in the collection based on the maximum activation value.
    
    Args:
        collection_df (pd.DataFrame): DataFrame containing the music collection data.
        style_metadata_file_path (str): Path to the JSON file containing Discogs metadata.
    
    Returns:
        pd.Series: A Series containing the music style names for each track.
    """
    # Use the maximum activation to determine the style for each track
    music_style_index = collection_df['music_styles_activations'].apply(lambda x: x.index(max(x)))
    
    # Get music style names from Discogs metadata
    music_style_metadata = u.load_data_from_json(style_metadata_file_path)
    music_style_names = music_style_metadata['classes']

    # Map numerical indices to music style names from Discogs metadata
    return music_style_index.map(lambda x: music_style_names[x])


def get_music_genres(collection_df):
    """
    Extract the genre from the music style string.
    
    Args:
        collection_df (pd.DataFrame): DataFrame containing the music collection data.
    
    Returns:
        pd.Series: A Series containing the genre for each track.
    """
    # Music style format is Genre---Music style
    return collection_df['music_style'].str.split('---').str[0]


def get_activation_value(collection_df, column_name, index):
    """
    Extract the activation value at a specific index from a list in a DataFrame column.
    
    Args:
        collection_df (pd.DataFrame): DataFrame containing the music collection data.
        column_name (str): Name of the column containing the activation values.
        index (int): Index of the activation value to extract.
    
    Returns:
        pd.Series: A Series containing the activation values at the specified index.
    """
    return collection_df[column_name].apply(lambda x: x[index])


def extend_collection_df(collection_df, style_metadata_file_path):
    """
    Extend the collection DataFrame with additional derived columns.
    
    Args:
        collection_df (pd.DataFrame): DataFrame containing the music collection data.
        style_metadata_file_path (str): Path to the JSON file containing Discogs metadata.
    
    Returns:
        pd.DataFrame: Extended DataFrame with additional columns for music style, genre, danceability, valence, arousal, instrumental, and voice.
    """

    # Explore music styles and genres from the collection
    collection_df['music_style'] = get_music_styles(collection_df, style_metadata_file_path)
    collection_df['genre'] = get_music_genres(collection_df)

    # Extract danceability, valence, and arousal values
    collection_df['danceability'] = get_activation_value(collection_df, 'danceability_activations', 0)
    collection_df['valence'] = get_activation_value(collection_df, 'arousal_valence_activations', 0)
    collection_df['arousal'] = get_activation_value(collection_df, 'arousal_valence_activations', 1)

    # Extract instrumental and voice values
    instrumental_values = get_activation_value(collection_df, 'voice_instrumental_activations', 0)
    voice_values = get_activation_value(collection_df, 'voice_instrumental_activations', 1)

    # Determine whether a track is instrumental or voice based on its values
    collection_df['instrumental'] = (instrumental_values > voice_values).astype(int)
    collection_df['voice'] = (voice_values > instrumental_values).astype(int)

    return collection_df


def main():
    
    # Load collection data
    collection_data = u.load_data_from_json(COLLECTION_JSON_FILE_PATH)

    # Convert collection data into pandas DataFrame
    collection_df = pd.DataFrame(collection_data)

    # Extend the DataFrame with additional derived columns
    collection_df = extend_collection_df(collection_df, STYLE_METADATA_FILE_PATH)

    # Save music style distribution
    u.save_music_style_distribution(collection_df, STYLE_DISTRIBUTION_TSV_PATH)

    # Save descriptors distributions plots
    u.save_genre_distribution_plot(collection_df, GENRE_DISTRIBUTION_PLOT_PATH)
    u.save_tempo_distribution_plot(collection_df, TEMPO_DISTRIBUTION_PLOT_PATH)
    u.save_danceability_distribution_plot(collection_df, DANCEABILITY_DISTRIBUTION_PLOT_PATH)
    u.save_key_distribution_plot(collection_df, KEY_DISTRIBUTION_PLOT_PATH)
    u.save_loudness_distribution_plot(collection_df, LOUDNESS_DISTRIBUTION_PLOT_PATH)
    u.save_valence_arousal_distribution_plot(collection_df, VALENCE_AROUSAL_DISTRIBUTION_PLOT_PATH)
    u.save_instrumental_voice_plot(collection_df, INSTRUMENTAL_VOICE_DISTRIBUTION_PLOT_PATH)


if __name__=="__main__":
    main()
