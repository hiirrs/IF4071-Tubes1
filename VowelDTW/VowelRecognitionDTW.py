import numpy as np
import librosa
from python_speech_features import mfcc, delta
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import os
from collections import defaultdict
from scipy.signal import medfilt
from scipy.linalg import inv, det

class VowelRecognitionDTW:
    def __init__(self, sample_rate=16000, use_vad=True, normalize=True, n_segments=3):
        """
        Sistem pengenalan vokal menggunakan DTW dan MFCC 39 dimensi dengan Generalized Template
        
        Parameters:
        - sample_rate: sampling rate audio (default 16000 Hz)
        - use_vad: gunakan Voice Activity Detection untuk membuang silence
        - normalize: normalisasi fitur MFCC
        - n_segments: jumlah segment untuk generalized template
        """
        self.sample_rate = sample_rate
        self.use_vad = use_vad
        self.normalize = normalize
        self.n_segments = n_segments
        self.templates = {}  # {vowel: {person_id: [list of mfcc_features]}}
        self.generalized_templates = {}  # {vowel: {'means': [], 'covariances': []}}
        self.vowels = ['a', 'i', 'u', 'e', 'o']
        
    def voice_activity_detection(self, audio, top_db=20):
        """
        Deteksi bagian audio yang mengandung suara (bukan silence)
        Membuang bagian awal dan akhir yang hening
        """
        intervals = librosa.effects.split(audio, top_db=top_db)
        
        if len(intervals) == 0:
            return audio
        
        start = intervals[0][0]
        end = intervals[-1][1]
        
        return audio[start:end]
    
    def extract_mfcc_39(self, audio_path):
        """
        Ekstraksi fitur MFCC 39 dimensi dengan preprocessing yang lebih baik
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        audio = librosa.effects.preemphasis(audio)
        
        if self.use_vad:
            audio = self.voice_activity_detection(audio, top_db=20)
        
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        
        mfcc_features = mfcc(
            audio, 
            samplerate=sr, 
            numcep=13,
            nfilt=26, 
            nfft=512,
            winstep=0.01,
            winlen=0.025,
            winfunc=np.hamming,
            preemph=0,  
            ceplifter=22,
            appendEnergy=True
        )
        
        for i in range(mfcc_features.shape[1]):
            mfcc_features[:, i] = medfilt(mfcc_features[:, i], kernel_size=3)
        
        delta_features = delta(mfcc_features, 2)
        delta2_features = delta(delta_features, 2)
        
        features_39 = np.hstack([mfcc_features, delta_features, delta2_features])
        
        if self.normalize:
            features_39 = (features_39 - np.mean(features_39, axis=0)) / (np.std(features_39, axis=0) + 1e-6)
        
        return features_39
    
    def segment_features(self, features):
        """
        Membagi fitur menjadi n_segments menggunakan uniform segmentation
        """
        n_frames = features.shape[0]
        segment_size = n_frames // self.n_segments
        segments = []
        
        for i in range(self.n_segments):
            start_idx = i * segment_size
            if i == self.n_segments - 1:
                end_idx = n_frames
            else:
                end_idx = (i + 1) * segment_size
            
            segment = features[start_idx:end_idx]
            segments.append(segment)
        
        return segments
    
    def compute_segment_statistics(self, segments_list):
        """
        Menghitung mean dan covariance untuk setiap segment dari multiple templates
        segments_list: list of segments from different templates
        """
        means = []
        covariances = []
        
        for segment_idx in range(self.n_segments):
            all_vectors = []
            
            for segments in segments_list:
                if segment_idx < len(segments):
                    all_vectors.extend(segments[segment_idx])
            
            if len(all_vectors) > 0:
                all_vectors = np.array(all_vectors)
                
                mean_vector = np.mean(all_vectors, axis=0)
                
                if all_vectors.shape[0] > 1:
                    cov_matrix = np.cov(all_vectors.T)
                    if cov_matrix.ndim == 0:
                        cov_matrix = np.eye(all_vectors.shape[1]) * cov_matrix
                    elif cov_matrix.shape[0] == 1:
                        cov_matrix = np.eye(all_vectors.shape[1]) * cov_matrix[0, 0]
                else:
                    cov_matrix = np.eye(all_vectors.shape[1]) * 0.01
                
                cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
                
                means.append(mean_vector)
                covariances.append(cov_matrix)
        
        return means, covariances
    
    def mahalanobis_distance(self, x, mean, cov):
        """
        Menghitung Mahalanobis distance
        """
        try:
            cov_inv = inv(cov)
            diff = x - mean
            distance = np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))
            return distance
        except:
            return euclidean(x, mean)
    
    def gaussian_log_likelihood(self, x, mean, cov):
        """
        Menghitung negative Gaussian log likelihood
        """
        try:
            cov_inv = inv(cov)
            det_cov = det(cov)
            
            if det_cov <= 0:
                det_cov = 1e-6
            
            diff = x - mean
            mahal_dist = np.dot(np.dot(diff, cov_inv), diff.T)
            
            log_likelihood = 0.5 * (len(mean) * np.log(2 * np.pi) + np.log(det_cov) + mahal_dist)
            
            return log_likelihood
        except:
            return euclidean(x, mean)
    
    def dtw_distance_generalized(self, test_features, means, covariances):
        """
        Menghitung jarak DTW menggunakan generalized template dengan Gaussian distribution
        """
        test_segments = self.segment_features(test_features)
        
        if len(means) != len(test_segments):
            min_segments = min(len(means), len(test_segments))
            means = means[:min_segments]
            covariances = covariances[:min_segments]
            test_segments = test_segments[:min_segments]
        
        total_distance = 0
        path_length = 0
        
        for i, (test_segment, mean, cov) in enumerate(zip(test_segments, means, covariances)):
            segment_distances = []
            
            for test_vector in test_segment:
                distance = self.gaussian_log_likelihood(test_vector, mean, cov)
                segment_distances.append(distance)
            
            if len(segment_distances) > 0:
                avg_segment_distance = np.mean(segment_distances)
                total_distance += avg_segment_distance
                path_length += 1
        
        if path_length > 0:
            normalized_distance = total_distance / path_length
        else:
            normalized_distance = float('inf')
        
        return normalized_distance
    
    def add_template(self, vowel, person_id, audio_path):
        """
        Menambahkan template ke dictionary
        """
        if vowel not in self.vowels:
            raise ValueError(f"Vokal harus salah satu dari {self.vowels}")
        
        features = self.extract_mfcc_39(audio_path)
        
        if vowel not in self.templates:
            self.templates[vowel] = {}
        
        if person_id not in self.templates[vowel]:
            self.templates[vowel][person_id] = []
        
        self.templates[vowel][person_id].append(features)
        print(f"Template ditambahkan: {vowel} dari {person_id} ({len(self.templates[vowel][person_id])} file)")
    
    def build_generalized_templates(self):
        """
        Membangun generalized templates dari semua templates yang ada
        """
        print("\n=== BUILDING GENERALIZED TEMPLATES ===")
        
        for vowel in self.vowels:
            if vowel not in self.templates:
                continue
            
            print(f"Building generalized template for vowel: {vowel}")
            
            all_segments = []
            
            for person_id, template_features_list in self.templates[vowel].items():
                for template_features in template_features_list:
                    segments = self.segment_features(template_features)
                    all_segments.append(segments)
            
            if len(all_segments) > 0:
                means, covariances = self.compute_segment_statistics(all_segments)
                
                self.generalized_templates[vowel] = {
                    'means': means,
                    'covariances': covariances
                }
                
                print(f"  - Created {len(means)} segments with mean and covariance parameters")
            else:
                print(f"  - No templates found for vowel {vowel}")
        
        print("Generalized templates built successfully!")
    
    def recognize(self, audio_path, template_person_id=None):
        """
        Mengenali vokal dari file audio menggunakan generalized templates
        """
        if not self.generalized_templates:
            print("Building generalized templates...")
            self.build_generalized_templates()
        
        test_features = self.extract_mfcc_39(audio_path)
        
        min_distance = float('inf')
        recognized_vowel = None
        all_distances = {}
        
        for vowel in self.vowels:
            if vowel not in self.generalized_templates:
                continue
            
            means = self.generalized_templates[vowel]['means']
            covariances = self.generalized_templates[vowel]['covariances']
            
            if len(means) > 0 and len(covariances) > 0:
                distance = self.dtw_distance_generalized(test_features, means, covariances)
                all_distances[vowel] = distance
                
                if distance < min_distance:
                    min_distance = distance
                    recognized_vowel = vowel
        
        return recognized_vowel, min_distance, all_distances, all_distances

    def print_detailed_prediction(self, audio_path, true_vowel, test_person_id, recognized_vowel, distance, final_vowel_distances, is_correct, scenario=""):
        """
        Print detailed prediction information
        """
        filename = os.path.basename(audio_path)
        status = "CORRECT" if is_correct else "WRONG"
        
        print(f"\n  Test: {filename}")
        print(f"    Person: {test_person_id}")
        print(f"    Actual: {true_vowel} | Predicted: {recognized_vowel} | {status}")
        print(f"    Best Distance: {distance:.4f}")
        
        print(f"    All distances:")
        sorted_distances = sorted(final_vowel_distances.items(), key=lambda x: x[1])
        for vowel, dist in sorted_distances:
            marker = " <- CHOSEN" if vowel == recognized_vowel else ""
            correct_marker = " (CORRECT)" if vowel == true_vowel else ""
            print(f"      {vowel}: {dist:.4f}{marker}{correct_marker}")
    
    def test_closed_scenario(self, test_data):
        """
        Skenario Closed
        """
        correct = 0
        total = 0
        results = []
        
        print(f"\n--- DETAILED CLOSED SCENARIO RESULTS ---")
        
        for audio_path, true_vowel, test_person_id in test_data:
            recognized_vowel, distance, all_distances, final_vowel_distances = self.recognize(audio_path)
            
            is_correct = (recognized_vowel == true_vowel)
            if is_correct:
                correct += 1
            total += 1
            
            self.print_detailed_prediction(
                audio_path, true_vowel, test_person_id, 
                recognized_vowel, distance, final_vowel_distances, 
                is_correct, "CLOSED"
            )
            
            results.append({
                'audio': audio_path,
                'true': true_vowel,
                'predicted': recognized_vowel,
                'person': test_person_id,
                'correct': is_correct,
                'distance': distance,
                'final_vowel_distances': final_vowel_distances
            })
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"\n  CLOSED SCENARIO SUMMARY: {correct}/{total} correct = {accuracy:.2f}%")
        return accuracy, results
    
    def test_open_scenario(self, test_data, template_person_ids):
        """
        Skenario Open
        """
        correct = 0
        total = 0
        results = []
        
        original_templates = self.templates.copy()
        original_generalized = self.generalized_templates.copy()
        
        filtered_templates = {}
        for vowel in self.templates:
            filtered_templates[vowel] = {
                pid: feat for pid, feat in self.templates[vowel].items() 
                if pid in template_person_ids
            }
        
        self.templates = filtered_templates
        self.generalized_templates = {}
        self.build_generalized_templates()
        
        print(f"\n--- DETAILED OPEN SCENARIO RESULTS ---")
        print(f"Using templates from: {template_person_ids}")
        
        for audio_path, true_vowel, test_person_id in test_data:
            recognized_vowel, distance, all_distances, final_vowel_distances = self.recognize(audio_path)
            
            is_correct = (recognized_vowel == true_vowel)
            if is_correct:
                correct += 1
            total += 1
            
            self.print_detailed_prediction(
                audio_path, true_vowel, test_person_id, 
                recognized_vowel, distance, final_vowel_distances, 
                is_correct, "OPEN"
            )
            
            results.append({
                'audio': audio_path,
                'true': true_vowel,
                'predicted': recognized_vowel,
                'person': test_person_id,
                'correct': is_correct,
                'distance': distance,
                'final_vowel_distances': final_vowel_distances
            })
        
        self.templates = original_templates
        self.generalized_templates = original_generalized
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"\n  OPEN SCENARIO SUMMARY: {correct}/{total} correct = {accuracy:.2f}%")
        return accuracy, results
    
    def evaluate_all_scenarios(self, test_data_us, test_data_other):
        """
        Evaluasi lengkap: closed, open, dan rata-rata
        """
        all_persons = list(set([person_id for _, _, person_id in test_data_us]))
        results_summary = {}
        
        for template_person in all_persons:
            print(f"\n{'='*60}")
            print(f"=== EVALUATION WITH TEMPLATES FROM {template_person.upper()} ===")
            print(f"{'='*60}")
            
            print(f"\nTotal test files (templates_us): {len(test_data_us)}")
            print(f"Total test files (templates_other): {len(test_data_other)}")
            
            print(f"\n[CLOSED SCENARIO] (all templates vs test from templates_us)")
            closed_acc, closed_results = self.test_closed_scenario(test_data_us)
            
            open_test_data = test_data_other
            
            print(f"\n[OPEN SCENARIO] (templates from {template_person} vs test from templates_other)")
            print(f"Test files: {len(open_test_data)}")
            
            open_acc, open_results = self.test_open_scenario(
                open_test_data, 
                [template_person]
            )
            
            avg_acc = (closed_acc + open_acc) / 2
            
            results_summary[template_person] = {
                'closed_accuracy': closed_acc,
                'open_accuracy': open_acc,
                'average_accuracy': avg_acc,
                'closed_results': closed_results,
                'open_results': open_results
            }
            
            print(f"\n--- SUMMARY FOR {template_person.upper()} ---")
            print(f"  Closed Accuracy: {closed_acc:.2f}%")
            print(f"  Open Accuracy: {open_acc:.2f}%")
            print(f"  Average Accuracy: {avg_acc:.2f}%")
        
        overall_closed = np.mean([r['closed_accuracy'] for r in results_summary.values()])
        overall_open = np.mean([r['open_accuracy'] for r in results_summary.values()])
        overall_avg = np.mean([r['average_accuracy'] for r in results_summary.values()])
        
        print(f"\n{'='*60}")
        print(f"=== OVERALL RESULTS ACROSS ALL TEMPLATE PERSONS ===")
        print(f"{'='*60}")
        print(f"Overall Closed Accuracy: {overall_closed:.2f}%")
        print(f"Overall Open Accuracy: {overall_open:.2f}%")
        print(f"Overall Average Accuracy: {overall_avg:.2f}%")
        
        results_summary['overall'] = {
            'closed_accuracy': overall_closed,
            'open_accuracy': overall_open,
            'average_accuracy': overall_avg
        }
        
        return results_summary
    
    def save_predictions_to_csv(self, results, filename):
        """
        Menyimpan hasil prediksi ke file CSV
        """
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['file', 'path', 'person', 'true_vowel', 'pred_vowel', 
                         'dist_a', 'dist_e', 'dist_i', 'dist_o', 'dist_u']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                row = {
                    'file': os.path.basename(result['audio']),
                    'path': result['audio'],
                    'person': result['person'],
                    'true_vowel': result['true'],
                    'pred_vowel': result['predicted'],
                    'dist_a': result['final_vowel_distances'].get('a', 0),
                    'dist_e': result['final_vowel_distances'].get('e', 0),
                    'dist_i': result['final_vowel_distances'].get('i', 0),
                    'dist_o': result['final_vowel_distances'].get('o', 0),
                    'dist_u': result['final_vowel_distances'].get('u', 0)
                }
                writer.writerow(row)
        
        print(f"Predictions saved to: {filename}")

    def print_confusion_matrix(self, results):
        """
        Print confusion matrix untuk analisis error
        """
        from collections import Counter
        
        confusion = defaultdict(lambda: defaultdict(int))
        
        for result in results:
            true_label = result['true']
            pred_label = result['predicted']
            confusion[true_label][pred_label] += 1
        
        print("\n=== CONFUSION MATRIX ===")
        print(f"{'':5}", end='')
        for vowel in self.vowels:
            print(f"{vowel:5}", end='')
        print()
        
        for true_vowel in self.vowels:
            print(f"{true_vowel:5}", end='')
            for pred_vowel in self.vowels:
                count = confusion[true_vowel][pred_vowel]
                print(f"{count:5}", end='')
            print()