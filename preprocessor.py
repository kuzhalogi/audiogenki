"""
1.Load a file
2. Pad the signal(if necessary)
3. Extracting log spectogram form signal
4. Normalise spectogram
5. Save the normalised spectogram

PreprocessingPipeline
"""

import librosa
import numpy as np
import os
import pickle

class Loader:
    """Loader is responsible for loading an audio file."""
    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono
    
    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0] #ignoring the sr output from the load function

class Padder:
    """Padder responsible to apply padding to an array."""
    def __init__(self, mode="constant"):
        self.mode = mode
    
    def left_pad(self, array, num_missing_items):
        padded_array = np.pad(array,(num_missing_items,0),mode=self.mode)
        return padded_array
    
    def right_pad(self, array, num_missing_items):
        padded_array = np.pad(array,(0,num_missing_items),mode=self.mode)
        return padded_array

class LogSepctrogramExtractor:
    """LogSepctrogramExtractor extracts log spectograms (in dB) from a time-series signal."""

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length
    
    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)[:-1]
        #(1 + frame_size / 2, numframe) 1024 -> 513 -> 512
    
class MinMaxNormaliser:
    """MinMaxNormaliser applies min max normaliation to an array."""
    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val
        
    def normalise(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array
    
    def denormalise(self, normalised_array, original_min, original_max):
        array = (normalised_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array
    
class Saver:
    """saver is responsible to save features, and the min max values."""
    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir
        
    def save_feature(self, feature, file_path):
        save_path = self.generate_save_path(file_path)
        np.save(save_path, feature)
        
    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir, "min_max_values.pkl")
        self.save(min_max_values, save_path)
        
    def save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        
    def generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feaure_save_dir, file_name + ".npy")
        return save_path
    
class PreprocessingPipeLine:
    """PreprocessingPipeLine process audio file in a directory, applying
    the following steps to each file:
    1. Load a file
    2. Pad the signal(if necessary)
    3. Extracting log spectogram form signal
    4. Normalise spectogram
    5. Save the normalised spectogram
    
    Storing the min max values for all the log spectorgrams. 
    """
    
    def __init__(self):
        self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}
        self._loader = None
        self.num_expected_sample = None
        
    @property
    def loader(self):
        return self._loader
    
    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self.num_expected_sample = int(loader.sample_rate * loader.duration)
    
    def process(self, audio_files_dir):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self.process_file(file_path)
                print(f"Processed file {file_path}")
        self.saver.save_min_max_values(self.min_max_values)
                
    def process_file(self, file_path):
        signal = self.loader.load(file_path)
        if self.is_padding_necessary(signal):
            signal = self.apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self.store_min_max_value(save_path, feature.min(), feature.max())
        
    def is_padding_necessary(self, signal):
        if len(signal) < self.num_expected_samples:
            return True
        return False
    
    def apply_padding(self, signal):
        num_missing_samples = self.num_expected_sample - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal
    
    def store_min_max_values(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {"min":min_val,
                                          "max":max_val}
        
        
if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74 # IN SECONDS
    SAMPLE_RATE = 22050
    MONO = True
    
    SPECTROGRAMS_SAVE_DIR = "/home/kuzhalogi/fsdd/spectograms/"
    MIN_MAX_VALUES_SAVE_DIR = "/home/kuzhalogi/fsdd/"
    FILES_DIR = "/home/kuzhalogi/WorkSpace/audiogen/archive/recordings/"
    
    # instantiate all objects
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()
    log_spectogram_extractor = LogSepctrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0,1)
    saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)
    preprocessing_pipline = PreprocessingPipeLine()
    preprocessing_pipline.loader = loader
    preprocessing_pipline.padder = padder
    preprocessing_pipline.extractor = log_spectogram_extractor
    preprocessing_pipline.normaliser = min_max_normaliser
    preprocessing_pipline.saver = saver
    
    preprocessing_pipline.process(FILES_DIR)