import json
import os

import essentia
import essentia.standard as es
from tqdm import tqdm

from audio_analyzer import AudioAnalyzer

essentia.log.warningActive = False  # Suppress Essentia warnings


# ===========================
# CONFIGURATION CONSTANTS
# ===========================

PATH_TO_MUSAV = "../data/MusAV/"
AUDIO_FILES_EXTENSION = ".mp3"

# Sample rates for resampling
RESAMPLE_INPUT_SAMPLE_RATE = 44100
RESAMPLE_TARGET_SAMPLE_RATE = 16000

# Key profile types for key extraction
KEY_PROFILE_TYPES = ['temperley', 'krumhansl', 'edma']

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
VOICE_INSTRUMENTAL_MODEL_PATH = f"{MODELS_PATH}voice_instrumental-discogs-effnet-1.pb"
DANCEABILITY_MODEL_PATH = f"{MODELS_PATH}danceability-discogs-effnet-1.pb"
AROUSAL_VALENCE_MODEL_PATH = f"{MODELS_PATH}emomusic-msd-musicnn-2.pb"

# ===========================
# OUTPUT CONSTANTS
# ===========================

# Path for the output JSON file
OUTPUT_JSON_FILE_PATH = "../audio_features.json"


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
        for file in files if file.lower().endswith(files_extension)
    ]


def main():

    # Load audio file paths
    audio_files_path = get_audio_files_paths(PATH_TO_MUSAV, AUDIO_FILES_EXTENSION)
    
    # Exit if no files found
    if not audio_files_path:
        print(f"No {AUDIO_FILES_EXTENSION} files found in: {PATH_TO_MUSAV}")
        return
    
    # Initialize audio processing components
    mono_mixer = es.MonoMixer()
    resampler = es.Resample(inputSampleRate=RESAMPLE_INPUT_SAMPLE_RATE, outputSampleRate=RESAMPLE_TARGET_SAMPLE_RATE)

    # Initialize tempo extractor
    tempo_extractor = es.TempoCNN(graphFilename=TEMPO_MODEL_PATH)  # alternatively: tempo_extractor = es.RhythmExtractor2013()

    # Initialize audio processing extractors
    loudness_extractor = es.LoudnessEBUR128()
    key_extractors = [(profile, es.KeyExtractor(profileType=profile)) for profile in KEY_PROFILE_TYPES]

    # Load deep learning models
    discogs_embeddings_model = es.TensorflowPredictEffnetDiscogs(graphFilename=DISCOGS_EMBEDDINGS_MODEL_PATH, output="PartitionedCall:1")
    musiccnn_embeddings_model = es.TensorflowPredictMusiCNN(graphFilename=MUSICCNN_EMBEDDINGS_MODEL_PATH, output="model/dense/BiasAdd")
    genres_model = es.TensorflowPredict2D(graphFilename=GENRES_MODEL_PATH, input="serving_default_model_Placeholder", output="PartitionedCall:0")
    voice_instrumental_model = es.TensorflowPredict2D(graphFilename=VOICE_INSTRUMENTAL_MODEL_PATH, output="model/Softmax")
    danceability_model = es.TensorflowPredict2D(graphFilename=DANCEABILITY_MODEL_PATH, output="model/Softmax")
    arousal_valence_model = es.TensorflowPredict2D(graphFilename=AROUSAL_VALENCE_MODEL_PATH, output="model/Identity")

    # Extract features from all audios in the collection
    collection_features = []
    for audio_file_path in tqdm(audio_files_path, desc="Processing audio files"):
        analyzer = AudioAnalyzer(audio_file_path, mono_mixer, resampler,
                                 tempo_extractor, key_extractors, loudness_extractor,
                                 discogs_embeddings_model, musiccnn_embeddings_model, genres_model,
                                 voice_instrumental_model, danceability_model, arousal_valence_model)

        collection_features.append(analyzer.extract_features())

    # Save extracted features to JSON
    with open(OUTPUT_JSON_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(collection_features, f, indent=4)
    
    print(f"Features saved to {OUTPUT_JSON_FILE_PATH}")


if __name__=="__main__":
    main()
