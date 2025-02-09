"""
This module provides an AudioAnalyzer class for extracting audio features
using Essentia models. It supports tempo, key, loudness, embeddings, and
various classification activations (e.g., genre, danceability, arousal-
valence). The class processes audio files and returns extracted features
in a structured format for further analysis.
"""

import essentia.standard as es
import numpy as np

# pylint: disable=too-many-instance-attributes
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=no-member


class AudioAnalyzer:
    """
    Extracts audio features using Essentia and deep learning models.
    """

    def __init__(
        self,
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
    ):
        """
        Initializes the AudioAnalyzer and loads the audio file.

        Args:
            audio_file_path: Path to the audio file.
            mono_mixer: Converts stereo audio to mono.
            resampler: Resamples the audio to a target sample rate.
            tempo_extractor: Extracts tempo (BPM).
            key_extractors: List of key extraction models.
            loudness_extractor: Extracts loudness.
            discogs_embeddings_model: Discogs-based embeddings model.
            musiccnn_embeddings_model: MusicCNN embeddings model.
            genres_model: Model for genre classification.
            voice_instrumental_model: Model for voice vs. instrumental
                classification.
            danceability_model: Model for danceability classification.
            arousal_valence_model: Model for arousal-valence classification.
        """
        self.audio_file_path = audio_file_path
        self.mono_mixer = mono_mixer
        self.resampler = resampler
        self.tempo_extractor = tempo_extractor
        self.key_extractors = key_extractors
        self.loudness_extractor = loudness_extractor
        self.discogs_embeddings_model = discogs_embeddings_model
        self.musiccnn_embeddings_model = musiccnn_embeddings_model
        self.genres_model = genres_model
        self.voice_instrumental_model = voice_instrumental_model
        self.danceability_model = danceability_model
        self.arousal_valence_model = arousal_valence_model

        # Load and process the audio
        self.stereo_audio = self._load_stereo_audio(self.audio_file_path)
        self.mono_audio = self._convert_to_mono(
            self.stereo_audio, self.mono_mixer
        )
        self.resampled_audio = self._resample_audio(
            self.mono_audio, self.resampler
        )

        # Compute embeddings
        self.discogs_embeddings = self.get_embeddings(
            self.resampled_audio, self.discogs_embeddings_model
        )
        self.musiccnn_embeddings = self.get_embeddings(
            self.resampled_audio, self.musiccnn_embeddings_model
        )

    def _load_stereo_audio(self, audio_file_path):
        """
        Load the audio file in stereo format using AudioLoader.

        Args:
            audio_file_path (str): Path to the audio file.

        Returns:
            numpy.ndarray: The stereo audio data.
        """
        audio_loader = es.AudioLoader(filename=audio_file_path)
        return audio_loader()[0]  # return stereo audio data

    def _convert_to_mono(self, stereo_audio, mono_mixer):
        """
        Convert stereo audio to mono using MonoMixer.

        Args:
            stereo_audio (numpy.ndarray): Stereo audio data.
            mono_mixer (MonoMixer): Essentia MonoMixer instance.

        Returns:
            numpy.ndarray: Mono audio data.
        """
        stereo_channels = 2  # number of stereo_channels
        return mono_mixer(stereo_audio, stereo_channels)

    def _resample_audio(self, mono_audio, resampler):
        """
        Resample the audio to the target sample rate.

        Args:
            mono_audio (numpy.ndarray): Mono audio data.
            resampler (Resample): Essentia Resample instance.

        Returns:
            numpy.ndarray: Resampled audio data.
        """
        return resampler(mono_audio)

    def _average_frames(self, frames):
        """
        Compute the average of frame-based features.

        Args:
            frames (numpy.ndarray): Array of frame-based activations.

        Returns:
            list: Averaged feature values.
        """
        return np.mean(frames, axis=0).tolist()

    def get_tempo(self, mono_audio, tempo_extractor):
        """
        Extract the estimated tempo (BPM) from mono audio.

        Args:
            mono_audio (numpy.ndarray): Mono audio data.
            tempo_extractor (RhythmExtractor2013 or TempoCNN): Tempo
                extraction model.

        Returns:
            float: Estimated BPM.
        """
        return tempo_extractor(mono_audio)[0]  # return bpm / global tempo

    def get_key(self, mono_audio, key_extractors):
        """
        Extract the musical key using multiple key profiles.

        Args:
            mono_audio (numpy.ndarray): Mono audio data.
            key_extractors (list): List of key extractor instances.

        Returns:
            list: A list of dictionaries containing key and scale information.
        """
        return [
            {"profile_type": profile_type, "key": f"{key} {scale}"}
            for profile_type, key_extractor in key_extractors
            for key, scale, _ in [key_extractor(mono_audio)]
        ]

    def get_loudness(self, stereo_audio, loudness_extractor):
        """
        Extract the integrated loudness of the audio.

        Args:
            stereo_audio (numpy.ndarray): Stereo audio data.
            loudness_extractor (LoudnessEBUR128): Loudness extraction model.

        Returns:
            float: Integrated loudness in LUFS.
        """
        return loudness_extractor(stereo_audio)[2]  # integrated loudness

    def get_embeddings(self, resampled_audio, embeddings_model):
        """
        Extract embeddings using a pre-trained deep learning model.

        Args:
            resampled_audio (numpy.ndarray): Resampled mono audio.
            embeddings_model: Deep learning model for extracting embeddings.

        Returns:
            numpy.ndarray: Extracted embeddings.
        """
        return embeddings_model(resampled_audio)

    def get_activations(self, embeddings, model):
        """
        Extract activations (e.g., genre, danceability) from embeddings.

        Args:
            embeddings (numpy.ndarray): Extracted embeddings.
            model: Pre-trained classification model.

        Returns:
            list: Activation values for the given model.
        """
        return self._average_frames(model(embeddings))

    def extract_features(self):
        """
        Extract all audio features and return them in a dictionary.

        Returns:
            dict: Dictionary containing extracted features.
        """
        return {
            "path": self.audio_file_path,
            "tempo": self.get_tempo(self.mono_audio, self.tempo_extractor),
            "key": self.get_key(self.mono_audio, self.key_extractors),
            "loudness": self.get_loudness(
                self.stereo_audio, self.loudness_extractor
            ),
            "embeddings": {
                "discogs": self._average_frames(self.discogs_embeddings),
                "musiccnn": self._average_frames(self.musiccnn_embeddings),
            },
            "music_styles_activations": self.get_activations(
                self.discogs_embeddings, self.genres_model
            ),
            "voice_instrumental_activations": self.get_activations(
                self.discogs_embeddings, self.voice_instrumental_model
            ),
            "danceability_activations": self.get_activations(
                self.discogs_embeddings, self.danceability_model
            ),
            "arousal_valence_activations": self.get_activations(
                self.musiccnn_embeddings, self.arousal_valence_model
            ),
        }
