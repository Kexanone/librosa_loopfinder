# librosa_loopfinder
Python library based on librosa for finding seamless music loops

## Installation
```sh
python3 -m pip install git+https://github.com/Kexanone/librosa_loopfinder.git
```

## Usage Example
```py
import numpy as np
import  librosa
from librosa_loopfinder import find_loop_points

data, sr = librosa.load('input.wav', sr=None)
# Get loop points based on a feature window of 8 seconds and a minimum loop duration of 60 seconds
loop_begin, loop_end, score = find_loop_points(y=data, sr=sr,
    win_length=librosa.time_to_samples(8),
    min_length=librosa.time_to_samples(60)
)[0]
intro = data[:loop_begin]
loop = data[loop_begin:loop_end]
outro = data[loop_end:]
# Loop music two times
new_data = np.hstack([intro] + 3*[loop] + [outro])
```
