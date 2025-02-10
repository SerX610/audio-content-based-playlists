# Audio Content Based Playlists

#### _THIS PROJECT IS STILL UNDER DEVELOPMENT [EXPECTED END DATE: 16/02/2025]_

This project has been developed for the Essentia playlists assignment in the context of the "Audio and Music Processing Lab" course from the Master's in Sound and Music Computing at Universitat Pompeu Fabra. For more information, check the [Assignment.pdf](Assignment.pdf)

The goal of the project is to create tools for analyzing and previewing music in any given collection. These tools should allow us to: 

- Generate automated reports about music collection with some analysis statistics. 
- Generate music playlists based on queries by descriptors (tempo, music style, danceability, voice vs instrumental, emotion, etc.) 
- Generate music playlists based on queries by track example (tracks similar to a given query track). 

For that, we will use the [MusAV](https://repositori.upf.edu/items/ea4c4a4c-958f-4004-bdc2-e1f6ad7e6829) dataset as a small music audio collection, extract music descriptors using Essentia, generate a report about computed descriptors, and create a simple user interface to generate playlists based on these descriptors

For more information about the implementation, design, and obtained results, check the [Report.pdf](Report.pdf).

#### ----- TODO: ADD REPORT -----


## Setup

- Clone the repository.

- Create and activate a **Python 3.10** virtual environment:

```
>>> python3.10 -m venv ess
>>> source ess/bin/activate
```

- Install the project dependencies using the following command:

```bash
>>> pip install -r requirements.txt
```


## Usage

### Audio Analysis

To analyze an audio collection, ensure the parameters in `analyze_audio_collection.py` are correctly defined according to your `data` and `models` paths, navigate to the `src/` folder and run the `analyze_audio_collection.py` script:

```bash
# from the src/ path
>>> python analyze_audio_collection.py
```

This script can take a long time to run, depending on your dataset size. In the end, it will output a JSON analysis file with the extracted descriptors from all tracks.

### Music Collection Overview

To visualize the music collection descriptors using the generated JSON file from the analysis, run the `explore_collection.py` script located in `src/`, ensuring all parameters are correctly defined:

```bash
# from the src/ path
>>> python explore_collection.py
```

This script outputs different plots to the `results/` path with insights into the music collection.

### Playlist Generation

There are two apps for generating playlists:

- **Queries App:** Generates music playlists based on queries by descriptors (tempo, music style, danceability, voice vs instrumental, emotion, etc.)

- **Similarities App:** Generates music playlists based on queries by track example (tracks similar to a given query track).

To run the **Queries App**, navigate to the project root path and run the `run_queries.sh` script with the following command:

```bash
# from the root path
>>> ./scripts/run_queries.sh
```

This opens the **Queries App**, where you can generate playlists by filtering and ranking tracks using the previously extracted descriptors. These playlists are stored in your system as an M3U8 file, which can be opened with an external media player, such as VLC.

To run the **Similarities App**, navigate to the project root path and run the `run_similarities.sh` script with the following command:

```bash
# from the root path
>>> ./scripts/run_similarities.sh
```

This opens the **Similarities App**, where you can generate playlists by retrieving the 10 most similar tracks to a given query track from the collection using two different models. These playlists are stored in your system as an M3U8 file, which can be opened with an external media player, such as VLC.


## Demo

#### ----- TODO: ADD DEMO FOR THE APPS -----
