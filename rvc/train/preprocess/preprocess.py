import multiprocessing
import os
import sys
import traceback

import librosa
import numpy as np
from scipy import signal
from scipy.io import wavfile

sys.path.append(os.getcwd())

from rvc.lib.audio import load_audio
from rvc.train.preprocess.slicer import Slicer

# Command line argument parsing
exp_dir = sys.argv[1]  # Directory for saving results
input_root = sys.argv[2]  # Directory with input audio files
percentage = float(sys.argv[3])  # Maximum segment length in seconds / Default = 3.0 (from n to 3sec)
sample_rate = int(sys.argv[4])  # Sample rate to convert data to / 32000, 40000 and 48000
normalize = sys.argv[5] == "True"  # Flag to enable/disable normalization
num_processes = max(1, os.cpu_count() - 1)  # Number of processes


class PreProcess:
    def __init__(self, sample_rate, exp_dir, percentage=3.0, normalize=True):
        # Directories for saving processed audio files
        self.gt_wavs_dir = os.path.join(exp_dir, "data", "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "data", "sliced_audios_16k")

        # Create directories if they don't exist
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

        # Initialize Slicer for audio segmentation
        self.slicer = Slicer(
            sr=sample_rate,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sample_rate = sample_rate  # Sample rate
        self.b_high, self.a_high = signal.butter(N=5, Wn=48, btype="high", fs=self.sample_rate)  # High-pass filter
        self.percentage = percentage  # Segment length
        self.overlap = 0.3  # Overlap between segments
        self.tail = self.percentage + self.overlap  # Tail for processing
        self.normalize = normalize  # Flag to enable/disable normalization

    def norm_write(self, tmp_audio, idx0, idx1):
        # Check for exceeding maximum signal level
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            return

        # Apply normalization to audio and save as WAV
        if self.normalize:
            tmp_audio = (tmp_audio / tmp_max * (0.9 * 0.75)) + (1 - 0.75) * tmp_audio
        wavfile.write(f"{self.gt_wavs_dir}/{idx0}_{idx1}.wav", self.sample_rate, tmp_audio.astype(np.float32))

        # Resample audio to 16 kHz and save as WAV
        tmp_audio_16k = librosa.resample(tmp_audio, orig_sr=self.sample_rate, target_sr=16000, res_type="soxr_vhq")
        wavfile.write(f"{self.wavs16k_dir}/{idx0}_{idx1}.wav", 16000, tmp_audio_16k.astype(np.float32))

    def pipeline(self, path, idx0):
        try:
            # Load audio
            audio = load_audio(path, self.sample_rate)
            # Apply high-pass filter
            audio = signal.lfilter(self.b_high, self.a_high, audio)

            idx1 = 0
            # Slice audio into segments
            for audio in self.slicer.slice(audio):
                i = 0
                while True:
                    # Calculate segment start point
                    start = int(self.sample_rate * (self.percentage - self.overlap) * i)
                    i += 1
                    # Check if audio tail remains
                    if len(audio[start:]) > self.tail * self.sample_rate:
                        tmp_audio = audio[start : start + int(self.percentage * self.sample_rate)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                        break
            print(f"{path}\t-> Done")
        except Exception:
            raise RuntimeError(f"{path}\t-> {traceback.format_exc()}")

    def pipeline_mp(self, infos):
        # Process a list of files
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, input_root, num_processes):
        print("Initializing audio data segmentation process...")
        try:
            # Collect file information from directory
            infos = [(os.path.join(input_root, name), idx) for idx, name in enumerate(sorted(list(os.listdir(input_root))))]

            # Parallel processing
            ps = []
            for i in range(num_processes):
                p = multiprocessing.Process(target=self.pipeline_mp, args=(infos[i::num_processes],))
                ps.append(p)
                p.start()
            for p in ps:
                p.join()
            print(f"✓ Segmentation successfully completed!")
        except Exception:
            raise RuntimeError(f"Error! {traceback.format_exc()}")


def preprocess_trainset(input_root, sample_rate, num_processes, exp_dir, percentage, normalize):
    # Initialize and run processing
    pp = PreProcess(sample_rate, exp_dir, percentage, normalize)
    pp.pipeline_mp_inp_dir(input_root, num_processes)


if __name__ == "__main__":
    # Run preprocessing
    preprocess_trainset(input_root, sample_rate, num_processes, exp_dir, percentage, normalize)
