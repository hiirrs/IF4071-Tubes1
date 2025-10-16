from __future__ import annotations

import os 
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import librosa
import librosa.display
from scipy.spatial.distance import cdist
from scipy.linalg import inv
import seaborn as sns

class VowelDTWVisualizer:
    def __init__(self, recognizer):
        self.recognizer = recognizer

    def _ensure_dir_for(self, path: str | None):
        if not path:
            return
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    def plot_waveform_with_vad(self, audio_path, top_db=20, save_path=None):
        y, sr = librosa.load(audio_path, sr=self.recognizer.sample_rate)
        y = librosa.effects.preemphasis(y)
        intervals = librosa.effects.split(y, top_db=top_db)

        plt.figure(figsize=(12, 3))
        times = np.arange(len(y))/sr
        plt.plot(times, y, linewidth=0.8, label='Sinyal audio')
        for i, (s, e) in enumerate(intervals):
            label = 'Aktivitas suara' if i == 0 else ''
            plt.axvspan(s/sr, e/sr, alpha=0.2, color='green', label=label)
        plt.title(f"Gelombang + interval VAD ({os.path.basename(audio_path)})")
        plt.xlabel("Waktu (detik)")
        plt.ylabel("Amplitudo")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            self._ensure_dir_for(save_path)
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_mfcc39_with_segments(self, audio_path, save_prefix=None):
        feats = self.recognizer.extract_mfcc_39(audio_path)
        segments = self.recognizer.segment_features(feats)
        
        mfcc_13   = feats[:, :13].T
        delta_13  = feats[:, 13:26].T
        delta2_13 = feats[:, 26:39].T

        def _show_with_segments(mat, title, segments_data):
            plt.figure(figsize=(12, 4))
            librosa.display.specshow(mat, x_axis='time', cmap='viridis')
            plt.title(f"{title} - {os.path.basename(audio_path)}")
            plt.colorbar(format="%+0.2f")
            
            n_frames = mat.shape[1]
            segment_size = n_frames // self.recognizer.n_segments
            for i in range(1, self.recognizer.n_segments):
                boundary = i * segment_size
                plt.axvline(x=boundary, color='red', linestyle='--', alpha=0.7, linewidth=2)
            
            for i in range(self.recognizer.n_segments):
                start_x = i * segment_size
                end_x = (i + 1) * segment_size if i < self.recognizer.n_segments - 1 else n_frames
                center_x = (start_x + end_x) / 2
                plt.text(center_x, mat.shape[0] - 1, f'Seg {i+1}', 
                        ha='center', va='top', color='white', fontweight='bold')
            
            plt.tight_layout()
            if save_prefix:
                filename = f"{save_prefix}_{title.replace(' ', '_').lower()}.png"
                self._ensure_dir_for(filename)
                plt.savefig(filename, dpi=150)
            plt.show()

        _show_with_segments(mfcc_13, "MFCC (13)", segments)
        _show_with_segments(delta_13, "Delta (13)", segments)
        _show_with_segments(delta2_13, "Delta-Delta (13)", segments)

    def plot_mfcc39(self, audio_path, save_prefix=None):
        self.plot_mfcc39_with_segments(audio_path, save_prefix)

    def plot_vowel_distances_bar(self, final_vowel_distances: dict, true_vowel=None, 
                                title="Jarak final per vokal", save_path=None):
        vowels = list(final_vowel_distances.keys())
        vals = [final_vowel_distances[v] for v in vowels]
        colors = ['green' if v == true_vowel else 'lightblue' for v in vowels]
        
        idx = np.arange(len(vowels))
        plt.figure(figsize=(8, 4))
        bars = plt.bar(idx, vals, color=colors, edgecolor='black', linewidth=0.5)
        plt.xticks(idx, vowels)
        plt.ylabel("Jarak DTW (ternormalisasi)")
        plt.title(title)
        
        for bar, val in zip(bars, vals):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom')
        
        if true_vowel:
            plt.text(0.02, 0.95, f'Vokal benar: {true_vowel}', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        if save_path:
            self._ensure_dir_for(save_path)
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_generalized_templates(self, vowel, save_path=None):
        if vowel not in self.recognizer.generalized_templates:
            print(f"Template umum tidak ditemukan untuk vokal '{vowel}'")
            return
        
        template = self.recognizer.generalized_templates[vowel]
        means = template['means']
        covariances = template['covariances']
        
        n_segments = len(means)
        fig, axes = plt.subplots(2, n_segments, figsize=(4*n_segments, 6))
        
        if n_segments == 1:
            axes = axes.reshape(2, 1)
        
        for i, (mean, cov) in enumerate(zip(means, covariances)):
            axes[0, i].bar(range(len(mean)), mean)
            axes[0, i].set_title(f'Segmen {i+1} - Rata-rata')
            axes[0, i].set_xlabel('Fitur MFCC')
            axes[0, i].set_ylabel('Nilai')
            axes[0, i].grid(True, alpha=0.3)
            
            im = axes[1, i].imshow(cov, cmap='coolwarm', aspect='auto')
            axes[1, i].set_title(f'Segmen {i+1} - Kovarians')
            axes[1, i].set_xlabel('Fitur')
            axes[1, i].set_ylabel('Fitur')
            plt.colorbar(im, ax=axes[1, i])
        
        plt.suptitle(f'Template Umum untuk Vokal: {vowel.upper()}')
        plt.tight_layout()
        if save_path:
            self._ensure_dir_for(save_path)
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_segment_distances(self, audio_path, vowel, save_path=None):
        if vowel not in self.recognizer.generalized_templates:
            print(f"Template umum tidak ditemukan untuk vokal '{vowel}'")
            return
        
        test_features = self.recognizer.extract_mfcc_39(audio_path)
        test_segments = self.recognizer.segment_features(test_features)
        
        template = self.recognizer.generalized_templates[vowel]
        means = template['means']
        covariances = template['covariances']
        
        segment_distances = []
        for i, (test_segment, mean, cov) in enumerate(zip(test_segments, means, covariances)):
            distances = []
            for test_vector in test_segment:
                dist = self.recognizer.gaussian_log_likelihood(test_vector, mean, cov)
                distances.append(dist)
            segment_distances.append(distances)
        
        plt.figure(figsize=(12, 4))
        for i, distances in enumerate(segment_distances):
            plt.subplot(1, len(segment_distances), i+1)
            plt.plot(distances, marker='o', markersize=3)
            plt.title(f'Segmen {i+1}')
            plt.xlabel('Frame')
            plt.ylabel('Log-likelihood')
            plt.grid(True, alpha=0.3)
            plt.text(0.5, 0.95, f'Rata-rata: {np.mean(distances):.3f}', 
                    transform=plt.gca().transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(f'Jarak per segmen: {os.path.basename(audio_path)} vs {vowel.upper()}')
        plt.tight_layout()
        if save_path:
            self._ensure_dir_for(save_path)
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_dtw_alignment(self, template_features, test_features, title="Penyelarasan DTW", save_path=None):
        A = template_features[:, :13]
        B = test_features[:, :13]
        D = cdist(A, B, metric='euclidean')
        C, wp = librosa.sequence.dtw(C=D)  

        plt.figure(figsize=(8, 6))
        plt.imshow(D.T, origin='lower', aspect='auto', cmap='hot')
        path_i = [p[0] for p in wp]
        path_j = [p[1] for p in wp]
        plt.plot(path_i, path_j, color='cyan', linewidth=2, label='Jalur warping')
        plt.xlabel("Frame template")
        plt.ylabel("Frame uji")
        plt.title(f"{title}\n(Matriks biaya + jalur warping)")
        plt.colorbar(label='Jarak')
        plt.legend()
        plt.tight_layout()
        if save_path:
            self._ensure_dir_for(save_path)
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_confusion_heatmap(self, results, vowels=('a','i','u','e','o'), 
                              title="Matriks Konfusi", save_path=None):
        idx = {v:i for i,v in enumerate(vowels)}
        cm = np.zeros((len(vowels), len(vowels)), dtype=int)
        for r in results:
            if r['true'] in idx and r['predicted'] in idx:
                cm[idx[r['true']], idx[r['predicted']]] += 1

        class_acc = []
        for i in range(len(vowels)):
            if cm[i].sum() > 0:
                acc = cm[i, i] / cm[i].sum() * 100
            else:
                acc = 0
            class_acc.append(acc)

        plt.figure(figsize=(7, 5.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=vowels, yticklabels=vowels,
                   square=True, cbar_kws={'label': 'Jumlah'})
        plt.title(f"{title}\nAkurasi Keseluruhan: {np.trace(cm)/np.sum(cm)*100:.1f}%")
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        
        for i, acc in enumerate(class_acc):
            plt.text(len(vowels) + 0.1, i + 0.5, f'{acc:.1f}%', 
                    ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            self._ensure_dir_for(save_path)
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_all_vowel_distances(self, audio_path, save_path=None):
        """Plot jarak ke semua template vokal untuk audio uji"""
        _, _, _, all_distances = self.recognizer.recognize(audio_path)
        
        vowels = list(all_distances.keys())
        distances = [all_distances[v] for v in vowels]
        
        plt.figure(figsize=(10, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, len(vowels)))
        bars = plt.bar(vowels, distances, color=colors, edgecolor='black', linewidth=0.5)
        
        min_idx = np.argmin(distances)
        bars[min_idx].set_color('red')
        bars[min_idx].set_edgecolor('darkred')
        bars[min_idx].set_linewidth(2)
        
        plt.ylabel('Jarak DTW (Log-likelihood)')
        plt.title(f'Jarak ke Semua Template Vokal\n{os.path.basename(audio_path)}')
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, distances):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.text(0.02, 0.95, f'Prediksi: {vowels[min_idx].upper()}', 
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.8, edgecolor='darkred'),
                color='white', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            self._ensure_dir_for(save_path)
            plt.savefig(save_path, dpi=150)
        plt.show()

    def plot_template_distribution(self, save_path=None):
        """Visualisasi distribusi template antar vokal dan orang"""
        if not self.recognizer.templates:
            print("Tidak ada template yang tersedia untuk divisualisasikan")
            return
        
        vowels = []
        persons = []
        counts = []
        
        for vowel, person_dict in self.recognizer.templates.items():
            for person, template_list in person_dict.items():
                vowels.append(vowel)
                persons.append(person)
                counts.append(len(template_list))
        
        unique_vowels = sorted(list(set(vowels)))
        unique_persons = sorted(list(set(persons)))
        
        matrix = np.zeros((len(unique_persons), len(unique_vowels)))
        for v, p, c in zip(vowels, persons, counts):
            v_idx = unique_vowels.index(v)
            p_idx = unique_persons.index(p)
            matrix[p_idx, v_idx] = c
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='g', cmap='YlOrRd',
                   xticklabels=unique_vowels, yticklabels=unique_persons,
                   cbar_kws={'label': 'Jumlah template'})
        plt.title('Distribusi Template Antar Vokal dan Orang')
        plt.xlabel('Vokal')
        plt.ylabel('Orang')
        plt.tight_layout()
        if save_path:
            self._ensure_dir_for(save_path)
            plt.savefig(save_path, dpi=150)
        plt.show()