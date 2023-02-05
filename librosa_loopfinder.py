import numpy as np
import librosa
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import DistanceMetric
from sklearn.pipeline import Pipeline


__version__ = '0.1.0'
__author__ = 'Kex'


class BeatFeaturesGenerator:
    '''
    Generates features for beats
    '''
    def __init__(self, n_pc=12, n_chroma=12, n_mels=128):
        self.n_pc = n_pc
        self.n_chroma = n_chroma
        self.n_mels = n_mels

    def __call__(self, y=None, sr=None, win_length=None):
        _, beats = librosa.beat.beat_track(y=y, sr=sr, units='samples')
        frame_features = Pipeline([
            ('MinMax1', MinMaxScaler()),
            ('PCA', PCA(n_components=self.n_pc)),
            ('MinMax2', MinMaxScaler())
        ]).fit_transform(get_frame_features(y=y, sr=sr, n_chroma=self.n_chroma, n_mels=self.n_mels))
        return frame_to_beat_features(frame_features=frame_features, beats=beats, sr=sr, win_length=win_length)


def find_loop_points(y=None, sr=None, win_length=None, min_length=None, max_length=np.inf,
    get_beat_features=BeatFeaturesGenerator(),
    distance_metric=DistanceMetric.get_metric('manhattan')
):
    '''
    Returns best loop points as a list of (begin stample, end sample, score)
    The score is based on a distance metric, which means the smaller, the better
    '''
    if min_length is None:
        min_length = win_length

    _, beats = librosa.beat.beat_track(y=y, sr=sr, units='samples')
    beat_fatures = get_beat_features(y=y, sr=sr, win_length=win_length)
    scores = distance_metric.pairwise(beat_fatures)
    results = []
    for (b1, b2), score in np.ndenumerate(scores):
        if b1 > b2:
            continue
        s1 = beats[b1]
        s2 = beats[b2]
        ds = s2 - s1
        if ds < min_length or ds > max_length:
            continue
        results.append((s1, s2, score))
    return sorted(results, key=lambda s: s[2])


def get_frame_features(y=None, sr=None, n_chroma=12, n_mels=128):
    '''
    Generates the frame features:
    - Chroma bins
    - Spectral flatness
    - Spectal contrast
    - Onset positions
    - Beat positions
    - Predominant local pulse (PLP)
    '''
    S = get_power_spectrogram(y=y)
    S_mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels)
    onset_envelope = librosa.onset.onset_strength(S=librosa.power_to_db(S_mel))
    return np.vstack([
        librosa.feature.chroma_stft(S=S, sr=sr, n_chroma=n_chroma),
        librosa.feature.spectral_flatness(S=S),
        librosa.feature.spectral_contrast(S=S),
        get_onset_frame_feature(onset_envelope=onset_envelope),
        get_beat_frame_feature(onset_envelope=onset_envelope),
        librosa.beat.plp(onset_envelope=onset_envelope)
    ]).T


def frame_to_beat_features(frame_features=None, beats=None, sr=None, win_length=None):
    '''
    Converts frame features to beat features
    Features are collected from frames after the beat too based on the given win_length
    win_length is given in number of samples
    '''
    n_frames = frame_features.shape[0]
    frame_window = librosa.samples_to_frames(win_length)
    beat_features = []
    for frame in librosa.samples_to_frames(beats):
        if frame + frame_window > n_frames:
            continue
        beat_features.append(np.hstack(frame_features[frame:frame+frame_window]))
    return np.array(beat_features)


def get_beat_frame_feature(onset_envelope=None):
    '''
    Returns a feature vector in frame space with 1 (beat) and 0 (no beat)
    '''
    _, beats = librosa.beat.beat_track(onset_envelope=onset_envelope)
    out = np.zeros(onset_envelope.shape[0])
    out[beats] = 1
    return out


def get_onset_frame_feature(onset_envelope=None):
    '''
    Returns a feature vector in frame space with 1 (onset) and 0 (no onset)
    '''
    onsets = librosa.onset.onset_detect(onset_envelope=onset_envelope)
    out = np.zeros(onset_envelope.shape[0])
    out[onsets] = 1
    return out


def get_power_spectrogram(y=None, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode='constant'):
    '''
    Used for precomputing the power spectrogram
    '''
    return (
        np.abs(
            librosa.stft(
                y,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                center=center,
                window=window,
                pad_mode=pad_mode,
            )
        )**2
    )
