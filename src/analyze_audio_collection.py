"""
This script processes a collection of audio files to extract features such as
tempo, key, loudness, embeddings, and classification activations (e.g., genre,
danceability, arousal-valence). It uses Essentia models for analysis and saves
the results incrementally to a JSON file for further use.
"""

import essentia
import essentia.standard as es
from tqdm import tqdm

import utils as u
from audio_analyzer import AudioAnalyzer

essentia.log.warningActive = False  # Suppress Essentia warnings

# pylint: disable=too-many-locals
# pylint: disable=no-member
# pylint: disable=missing-function-docstring
# pylint: disable=broad-exception-caught


# ===========================
# CONFIGURATION CONSTANTS
# ===========================

PATH_TO_MUSAV = "../data/MusAV/"
AUDIO_FILES_EXTENSION = ".mp3"

# Sample rates for resampling
RESAMPLE_INPUT_SAMPLE_RATE = 44100
RESAMPLE_TARGET_SAMPLE_RATE = 16000

# Key profile types for key extraction
KEY_PROFILE_TYPES = ["temperley", "krumhansl", "edma"]

# ===========================
# MODEL CONSTANTS
# ===========================

# Base path for all models
MODELS_PATH = "../models/"

# Paths to deep learning models
TEMPO_MODEL_PATH = f"{MODELS_PATH}deeptemp-k4-3.pb"
DISCOGS_EMBEDDINGS_MODEL_PATH = f"{MODELS_PATH}discogs-effnet-bs64-1.pb"
MUSICCNN_EMBEDDINGS_MODEL_PATH = f"{MODELS_PATH}msd-musicnn-1.pb"
GENRES_MODEL_PATH = f"{MODELS_PATH}genre_discogs400-discogs-effnet-1.pb"
VOICE_INSTRUMENTAL_MODEL_PATH = (
    f"{MODELS_PATH}voice_instrumental-discogs-effnet-1.pb"
)
DANCEABILITY_MODEL_PATH = f"{MODELS_PATH}danceability-discogs-effnet-1.pb"
AROUSAL_VALENCE_MODEL_PATH = f"{MODELS_PATH}emomusic-msd-musicnn-2.pb"

# ===========================
# OUTPUT CONSTANTS
# ===========================

# Path for the output JSON file
OUTPUT_JSON_FILE_PATH = "../results/musav_analysis.json"

# Maximum number of retries for file analysis
FILE_ANALYSIS_MAX_RETRIES = 3


def main():

    # Load audio file paths
    audio_files_path = u.get_audio_files_paths(
        PATH_TO_MUSAV, AUDIO_FILES_EXTENSION
    )

    # Exit if no files found
    if not audio_files_path:
        print(f"No {AUDIO_FILES_EXTENSION} files found in: {PATH_TO_MUSAV}")
        return

    # Load existing results if the JSON file exists
    collection_features = u.load_data_from_json(OUTPUT_JSON_FILE_PATH)
    processed_files = {result["path"] for result in collection_features}

    # Track retries for each file
    retry_count = {}

    # Initialize audio processing components
    mono_mixer = es.MonoMixer()
    resampler = es.Resample(
        inputSampleRate=RESAMPLE_INPUT_SAMPLE_RATE,
        outputSampleRate=RESAMPLE_TARGET_SAMPLE_RATE,
    )

    # Initialize tempo extractor
    tempo_extractor = es.TempoCNN(
        graphFilename=TEMPO_MODEL_PATH
    )  # alternatively: tempo_extractor = es.RhythmExtractor2013()

    # Initialize audio processing extractors
    loudness_extractor = es.LoudnessEBUR128()
    key_extractors = [
        (profile, es.KeyExtractor(profileType=profile))
        for profile in KEY_PROFILE_TYPES
    ]

    # Load deep learning models
    discogs_embeddings_model = es.TensorflowPredictEffnetDiscogs(
        graphFilename=DISCOGS_EMBEDDINGS_MODEL_PATH, output="PartitionedCall:1"
    )
    musiccnn_embeddings_model = es.TensorflowPredictMusiCNN(
        graphFilename=MUSICCNN_EMBEDDINGS_MODEL_PATH,
        output="model/dense/BiasAdd",
    )
    genres_model = es.TensorflowPredict2D(
        graphFilename=GENRES_MODEL_PATH,
        input="serving_default_model_Placeholder",
        output="PartitionedCall:0",
    )
    voice_instrumental_model = es.TensorflowPredict2D(
        graphFilename=VOICE_INSTRUMENTAL_MODEL_PATH, output="model/Softmax"
    )
    danceability_model = es.TensorflowPredict2D(
        graphFilename=DANCEABILITY_MODEL_PATH, output="model/Softmax"
    )
    arousal_valence_model = es.TensorflowPredict2D(
        graphFilename=AROUSAL_VALENCE_MODEL_PATH, output="model/Identity"
    )

    # Keep running until all files are processed
    while len(processed_files) < len(audio_files_path):

        # Extract features from all audios in the collection
        for audio_file_path in tqdm(
            audio_files_path, desc="Processing audio files"
        ):

            if audio_file_path in processed_files:
                continue  # Skip already processed files

            try:
                # Create an instance of AudioAnalyzer
                analyzer = AudioAnalyzer(
                    audio_file_path,
                    mono_mixer,
                    resampler,
                    tempo_extractor,
                    key_extractors,
                    loudness_extractor,
                    discogs_embeddings_model,
                    musiccnn_embeddings_model,
                    genres_model,
                    voice_instrumental_model,
                    danceability_model,
                    arousal_valence_model,
                )

                # Extract features
                audio_features = analyzer.extract_features()

                # Append the features to the list
                collection_features.append(audio_features)

                # Save results incrementally
                u.save_data_to_json(OUTPUT_JSON_FILE_PATH, collection_features)

                # Mark the file as processed
                processed_files.add(audio_file_path)

            except Exception as e:
                print(f"Error analyzing {audio_file_path}: {e}")
                retry_count[audio_file_path] = (
                    retry_count.get(audio_file_path, 0) + 1
                )

                # Skip the file if max retries reached
                if retry_count[audio_file_path] >= FILE_ANALYSIS_MAX_RETRIES:
                    print(
                        f"Skipping {audio_file_path} after \
                            {FILE_ANALYSIS_MAX_RETRIES} retries"
                    )
                    processed_files.add(audio_file_path)

    print(f"All files processed. Features saved to {OUTPUT_JSON_FILE_PATH}")


if __name__ == "__main__":
    main()
