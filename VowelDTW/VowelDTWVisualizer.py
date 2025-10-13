from __future__ import annotations

import os 
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import librosa
import librosa.display
from scipy.spatial.distance import cdist

class VowelDTWVisualizer:
    def __init__(self, recognizer: VowelRecognitionDTW):
        self.recognizer = recognizer

    # Util: pastikan folder untuk path ada
    def _ensure_dir_for(self, path: str | None):
        if not path:
            return
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    # A. Waveform + VAD 
    def plot_waveform_with_vad(self, audio_path, top_db=20, save_path=None):
        y, sr = librosa.load(audio_path, sr=self.recognizer.sample_rate)
        y = librosa.effects.preemphasis(y)

        intervals = librosa.effects.split(y, top_db=top_db)

        plt.figure(figsize=(12, 3))
        times = np.arange(len(y))/sr
        plt.plot(times, y, linewidth=0.8)
        for (s, e) in intervals:
            plt.axvspan(s/sr, e/sr, alpha=0.2)
        plt.title(f"Waveform + VAD intervals ({os.path.basename(audio_path)})")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        if save_path:
            self._ensure_dir_for(save_path)
            plt.savefig(save_path, dpi=150)
        plt.show()

    # B. MFCC 13 + Delta + Delta2 
    def plot_mfcc39(self, audio_path, save_prefix=None):
        feats = self.recognizer.extract_mfcc_39(audio_path)
        # Split 13-13-13
        mfcc_13   = feats[:, :13].T
        delta_13  = feats[:, 13:26].T
        delta2_13 = feats[:, 26:39].T

        def _show(mat, title):
            plt.figure(figsize=(10, 3.2))
            librosa.display.specshow(mat, x_axis='time', cmap=None)
            plt.title(title)
            plt.colorbar(format="%+0.2f")
            plt.tight_layout()
            if save_prefix:
                # contoh nama: <prefix>_mfcc_(13).png -> kita rapikan tanpa spasi
                filename = f"{save_prefix}_{title.replace(' ', '_').lower()}.png"
                self._ensure_dir_for(filename)
                plt.savefig(filename, dpi=150)
            plt.show()

        _show(mfcc_13,   "MFCC (13)")
        _show(delta_13,  "Delta (13)")
        _show(delta2_13, "Delta-Delta (13)")

    # C. Bar: jarak rata2 per vokal 
    def plot_vowel_distances_bar(self, final_vowel_distances: dict, title="Final distances per vowel", save_path=None):
        vowels = list(final_vowel_distances.keys())
        vals = [final_vowel_distances[v] for v in vowels]
        idx = np.arange(len(vowels))
        plt.figure(figsize=(6, 3.5))
        plt.bar(idx, vals)
        plt.xticks(idx, vowels)
        plt.ylabel("DTW distance (normalized)")
        plt.title(title)
        plt.tight_layout()
        if save_path:
            self._ensure_dir_for(save_path)
            plt.savefig(save_path, dpi=150)
        plt.show()

    # D. DTW alignment (cost matrix + warping path) 
    def plot_dtw_alignment(self, template_features, test_features, title="DTW alignment", save_path=None):

        A = template_features[:, :13]
        B = test_features[:, :13]
        D = cdist(A, B, metric='euclidean')
        C, wp = librosa.sequence.dtw(C=D)  

        plt.figure(figsize=(6.8, 6))
        plt.imshow(D.T, origin='lower', aspect='auto')
        path_i = [p[0] for p in wp]
        path_j = [p[1] for p in wp]
        plt.plot(path_i, path_j, linewidth=1.0)
        plt.xlabel("Template frames")
        plt.ylabel("Test frames")
        plt.title(f"{title}\n(cost colormap + warping path)")
        plt.tight_layout()
        if save_path:
            self._ensure_dir_for(save_path)
            plt.savefig(save_path, dpi=150)
        plt.show()

    # E. Confusion Matrix Heatmap
    def plot_confusion_heatmap(self, results, vowels=('a','i','u','e','o'), title="Confusion Matrix", save_path=None):
        idx = {v:i for i,v in enumerate(vowels)}
        cm = np.zeros((len(vowels), len(vowels)), dtype=int)
        for r in results:
            if r['true'] in idx and r['predicted'] in idx:
                cm[idx[r['true']], idx[r['predicted']]] += 1

        plt.figure(figsize=(5.2, 4.7))
        plt.imshow(cm, origin='upper', aspect='equal')
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(range(len(vowels)), vowels)
        plt.yticks(range(len(vowels)), vowels)

        for i in range(len(vowels)):
            for j in range(len(vowels)):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center')

        plt.colorbar(ticks=np.linspace(cm.min(), cm.max() if cm.max()>0 else 1, 5))
        plt.tight_layout()
        if save_path:
            self._ensure_dir_for(save_path)
            plt.savefig(save_path, dpi=150)
        plt.show()
