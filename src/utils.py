"""
This module provides utility functions for analyzing and visualizing music
collection data. It includes functions for loading and saving data, generating
distribution plots, and handling music metadata. The module is designed to
work with pandas DataFrames and uses seaborn and matplotlib for visualization.
"""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# pylint: disable=unused-argument


def get_audio_files_paths(audios_folder, files_extension=".mp3"):
    """
    Recursively find all audio files with the given extension in the directory.

    Args:
        audios_folder (str): Path to the folder containing audio files.
        files_extension (str): File extension to search for (default: ".mp3").

    Returns:
        list: List of file paths matching the extension.
    """
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(audios_folder)
        for file in files
        if file.lower().endswith(files_extension)
    ]


def load_data_from_json(json_file_path):
    """
    Load data from a JSON file.

    Args:
        json_file_path (str): Path to the JSON file.

    Returns:
        dict: Data loaded from the JSON file.
    """
    if os.path.exists(json_file_path):
        with open(json_file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_data_to_json(json_file_path, data):
    """
    Save data incrementally to a JSON file.

    Args:
        json_file_path (str): Path to the JSON file.
        data (dict): Data to save to the JSON file.
    """
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_music_style_distribution(collection_df, output_file):
    """
    Calculate and save the distribution of music styles to a TSV file.

    Args:
        collection_df (pd.DataFrame): DataFrame containing music track data.
        output_file (str): Path to save the TSV file.
    """
    # Calculate the music style distribution
    music_style_column = collection_df["music_style"]
    music_style_distribution_df = (
        music_style_column.value_counts().reset_index()
    )

    # Save the music style distribution to a TSV file
    tsv_separation = "\t"
    music_style_distribution_df.to_csv(
        output_file, sep=tsv_separation, index=False
    )


def save_plot(output_file, plot_function, collection_df, **kwargs):
    """
    Generic function to create and save a plot.

    Args:
        output_file (str): Path to save the plot image.
        plot_function (callable): Function that generates the plot.
        collection_df (pd.DataFrame): DataFrame containing the data to plot.
        **kwargs: Additional arguments to pass to the plot function.
    """
    # Create a new figure with a specified size
    plt.figure(figsize=kwargs.get("figsize", (10, 6)))

    # Call the plot function
    plot_function(collection_df, **kwargs)

    # Add grid lines to the plot if the 'grid' argument is True
    if kwargs.get("grid", True):

        # Determine the axis for grid lines (default is 'both')
        axis = kwargs.get("grid_axis", "both")

        # Add grid lines with a dashed style and transparency
        plt.grid(axis=axis, linestyle="--", alpha=0.7)

    # Set the title of the plot using the 'title' argument
    plt.title(kwargs.get("title", ""))

    # Set the x-axis label using the 'xlabel' argument
    plt.xlabel(kwargs.get("xlabel", ""))

    # Set the y-axis label using the 'ylabel' argument
    plt.ylabel(kwargs.get("ylabel", ""))

    # Save the plot to the specified output file
    plt.savefig(output_file, bbox_inches="tight")

    # Close the plot to free up memory
    plt.close()


def save_genre_distribution_plot(collection_df, output_file):
    """
    Generate and save a plot of the distribution of music genres.

    Args:
        collection_df (pd.DataFrame): DataFrame containing music collection
            data with a 'genre' column.
        output_file (str): Path to save the plot image.
    """

    def plot(df, **kwargs):
        sns.countplot(
            y="genre", data=df, order=df["genre"].value_counts().index
        )

    save_plot(
        output_file,
        plot,
        collection_df,
        title="Music Genres Distribution",
        xlabel="Frequency",
        ylabel="Music Genre",
        grid_axis="x",
    )


def save_tempo_distribution_plot(collection_df, output_file):
    """
    Generate and save a plot of the distribution of tempo values.

    Args:
        collection_df (pd.DataFrame): DataFrame containing music collection
            data with a 'tempo' column.
        output_file (str): Path to save the plot image.
    """

    # Define the plot function that will generate the distribution plot
    def plot(df, **kwargs):
        sns.histplot(df["tempo"], bins=40)

    # Call the generic save_plot function to create and save the plot
    save_plot(
        output_file,
        plot,
        collection_df,
        title="Tempo Distribution",
        xlabel="Tempo (BPM)",
        ylabel="Frequency",
        grid_axis="y",
    )


def save_danceability_distribution_plot(collection_df, output_file):
    """
    Generate and save a plot of the distribution of danceability scores.

    Args:
        collection_df (pd.DataFrame): DataFrame containing music collection
            data.
        output_file (str): Path to save the plot image.
    """

    # Define the plot function that will generate the distribution plot
    def plot(df, **kwargs):
        sns.histplot(df["danceability"], bins=40)

    # Call the generic save_plot function to create and save the plot
    save_plot(
        output_file,
        plot,
        collection_df,
        title="Danceability Distribution",
        xlabel="Danceability Score",
        ylabel="Frequency",
        grid_axis="y",
    )


def save_key_distribution_plot(collection_df, output_file):
    """
    Generate and save a plot of the distribution of musical keys,
    grouped by profile type.

    Args:
        collection_df (pd.DataFrame): DataFrame containing music collection
            data with a 'key' column.
        output_file (str): Path to save the plot image.
    """
    # Extract keys and profile types from the 'key' column
    keys = []
    profile_types = []
    for key_list in collection_df["key"]:
        for key_dict in key_list:
            keys.append(key_dict["key"])
            profile_types.append(key_dict["profile_type"])

    # Convert the lists into a DataFrame for easier plotting
    keys_df = pd.DataFrame({"key": keys, "profile_type": profile_types})

    # Define the plot function that will generate the distribution plot
    def plot(df, **kwargs):
        sns.countplot(
            y="key",
            hue="profile_type",
            data=df,
            order=df["key"].value_counts().index,
            palette="viridis",
        )

    # Call the generic save_plot function to create and save the plot
    save_plot(
        output_file,
        plot,
        keys_df,
        title="Distribution of Musical Keys (Grouped by Profile Type)",
        xlabel="Frequency",
        ylabel="Key",
        grid_axis="x",
        figsize=(12, 8),
    )


def save_loudness_distribution_plot(collection_df, output_file):
    """
    Generate and save a plot of the distribution of loudness values.

    Args:
        collection_df (pd.DataFrame): DataFrame containing music collection
            data with a 'loudness' column.
        output_file (str): Path to save the plot image.
    """

    # Define the plot function that will generate the distribution plot
    def plot(df, **kwargs):
        sns.histplot(df["loudness"], bins=40)

    # Call the generic save_plot function to create and save the plot
    save_plot(
        output_file,
        plot,
        collection_df,
        title="Loudness Distribution",
        xlabel="Loudness (dB)",
        ylabel="Frequency",
        grid_axis="y",
    )


def save_valence_arousal_distribution_plot(collection_df, output_file):
    """
    Generate and save a 2D scatter plot with density contours for valence
    and arousal.

    Args:
        collection_df (pd.DataFrame): DataFrame containing music collection
            data with 'valence' and 'arousal' columns.
        output_file (str): Path to save the plot image.
    """

    # Define the plot function that will generate the distribution plot
    def plot(df, **kwargs):
        plt.scatter(df["valence"], df["arousal"], alpha=0.5, edgecolor="black")
        plt.xlim(1, 9)
        plt.ylim(1, 9)

    # Call the generic save_plot function to create and save the plot
    save_plot(
        output_file,
        plot,
        collection_df,
        title="Valence vs Arousal 2D Distribution",
        xlabel="Valence",
        ylabel="Arousal",
        figsize=(10, 8),
    )


def save_instrumental_voice_plot(collection_df, output_file):
    """
    Generate and save a bar plot showing the counts of songs classified
    as instrumental and voice.

    Args:
        collection_df (pd.DataFrame): DataFrame containing music collection
            data with 'instrumental' and 'voice' columns.
        output_file (str): Path to save the plot image.
    """
    # Count the number of instrumental and voice tracks
    instrumental_count = collection_df["instrumental"].sum()
    voice_count = collection_df["voice"].sum()

    # Data for plotting
    labels = ["Instrumental", "Voice"]
    counts = [instrumental_count, voice_count]

    # Define the plot function that will generate the distribution plot
    def plot(df, **kwargs):
        ax = sns.barplot(x=labels, y=counts, edgecolor="black")
        for i, count in enumerate(counts):
            ax.text(
                i, count / 2, f"{count}", ha="center", va="center", fontsize=12
            )

    # Call the generic save_plot function to create and save the plot
    save_plot(
        output_file,
        plot,
        collection_df,
        title="Instrumental vs. Voice Distribution",
        xlabel="Classification",
        ylabel="Frequency",
        grid_axis="y",
        figsize=(8, 6),
    )
