import numpy as np
import librosa
from python_speech_features import mfcc, delta
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import os
from collections import defaultdict
from scipy.signal import medfilt

class VowelRecognitionDTW:
    def __init__(self, sample_rate=16000, use_vad=True, normalize=True):
        """
        Sistem pengenalan vokal menggunakan DTW dan MFCC 39 dimensi
        dengan Generalized Template sesuai PPT
        
        Parameters:
        - sample_rate: sampling rate audio (default 16000 Hz)
        - use_vad: gunakan Voice Activity Detection untuk membuang silence
        - normalize: normalisasi fitur MFCC
        """
        self.sample_rate = sample_rate
        self.use_vad = use_vad
        self.normalize = normalize
        
        # Storage untuk raw templates sebelum digeneralisasi
        self.raw_templates = {}  # {vowel: {person_id: [list of mfcc_features]}}
        
        # Storage untuk generalized templates (hasil akhir untuk recognition)
        self.generalized_templates = {}  # {vowel: generalized_mfcc_features}
        
        self.vowels = ['a', 'i', 'u', 'e', 'o']
        
    def voice_activity_detection(self, audio, top_db=20):
        """
        Deteksi bagian audio yang mengandung suara (bukan silence)
        Membuang bagian awal dan akhir yang hening
        """
        # Deteksi non-silent intervals
        intervals = librosa.effects.split(audio, top_db=top_db)
        
        if len(intervals) == 0:
            return audio
        
        # Ambil dari awal suara pertama sampai akhir suara terakhir
        start = intervals[0][0]
        end = intervals[-1][1]
        
        return audio[start:end]
    
    def extract_mfcc_39(self, audio_path):
        """
        Ekstraksi fitur MFCC 39 dimensi dengan preprocessing yang lebih baik
        """
        # Load audio file
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Pre-emphasis filter (meningkatkan frekuensi tinggi)
        audio = librosa.effects.preemphasis(audio)
        
        # Voice Activity Detection - buang silence
        if self.use_vad:
            audio = self.voice_activity_detection(audio, top_db=20)
        
        # Normalisasi amplitudo
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        
        # Ekstraksi 13 MFCC koefisien dengan parameter yang lebih baik
        mfcc_features = mfcc(
            audio, 
            samplerate=sr, 
            numcep=13,
            nfilt=26, 
            nfft=512,
            winstep=0.01,
            winlen=0.025,
            preemph=0,  
            ceplifter=22,
            appendEnergy=True
        )
        
        # Median filtering untuk mengurangi noise
        for i in range(mfcc_features.shape[1]):
            mfcc_features[:, i] = medfilt(mfcc_features[:, i], kernel_size=3)
        
        # Hitung delta (turunan pertama)
        delta_features = delta(mfcc_features, 2)
        
        # Hitung delta-delta (turunan kedua)
        delta2_features = delta(delta_features, 2)
        
        # Gabungkan menjadi 39 dimensi
        features_39 = np.hstack([mfcc_features, delta_features, delta2_features])
        
        # Normalisasi per fitur (Cepstral Mean and Variance Normalization)
        if self.normalize:
            features_39 = (features_39 - np.mean(features_39, axis=0)) / (np.std(features_39, axis=0) + 1e-6)
        
        return features_39
    
    def dtw_distance(self, template, test):
        """
        Menghitung jarak DTW dengan normalisasi path length
        """
        distance, path = fastdtw(template, test, dist=euclidean)
        
        # Normalisasi berdasarkan panjang path untuk fairness
        normalized_distance = distance / len(path)
        
        return normalized_distance
    
    def segment_template(self, template_features, num_segments=5):
        """
        Membagi template menjadi beberapa segmen
        
        Parameters:
        - template_features: MFCC features (time_frames x 39)
        - num_segments: jumlah segmen yang diinginkan
        
        Returns:
        - List of segments, each segment is a numpy array
        """
        total_frames = template_features.shape[0]
        segment_size = total_frames // num_segments
        
        segments = []
        for i in range(num_segments):
            start_idx = i * segment_size
            if i == num_segments - 1:  # segmen terakhir ambil sisa
                end_idx = total_frames
            else:
                end_idx = (i + 1) * segment_size
            
            segment = template_features[start_idx:end_idx]
            segments.append(segment)
        
        return segments
    
    def create_generalized_template(self, all_segments):
        """
        Membuat generalized template dari kumpulan segmen
        
        Parameters:
        - all_segments: List of segments dari berbagai template
        
        Returns:
        - generalized_template: numpy array (combined_frames x 39)
        """
        if len(all_segments) == 0:
            return np.array([])
        
        # Metode konkatenasi sederhana 
        generalized = np.vstack(all_segments)
        
        return generalized
    
    def build_generalized_templates(self, target_person_ids=None, num_segments=5):
        """
        Membangun generalized templates dari raw templates
        
        Parameters:
        - target_person_ids: List of person IDs to use (None = all persons)
        - num_segments: jumlah segmen per template
        """
        print(f"\n=== BUILDING GENERALIZED TEMPLATES ===")
        
        self.generalized_templates = {}
        
        for vowel in self.vowels:
            if vowel not in self.raw_templates:
                continue
            
            print(f"\nProcessing vowel: {vowel}")
            all_segments = []
            template_count = 0
            
            for person_id, template_list in self.raw_templates[vowel].items():
                # Filter berdasarkan target_person_ids jika ada
                if target_person_ids is not None and person_id not in target_person_ids:
                    continue
                
                print(f"  Person {person_id}: {len(template_list)} templates")
                
                for template_features in template_list:
                    # Bagi template menjadi segmen
                    segments = self.segment_template(template_features, num_segments)
                    all_segments.extend(segments)
                    template_count += 1
            
            if len(all_segments) > 0:
                # Buat generalized template
                generalized = self.create_generalized_template(all_segments)
                self.generalized_templates[vowel] = generalized
                
                print(f"  → Created generalized template: {generalized.shape[0]} frames from {len(all_segments)} segments ({template_count} templates)")
            else:
                print(f"  → No templates available for vowel {vowel}")
    
    def add_template(self, vowel, person_id, audio_path):
        """
        Menambahkan raw template ke storage
        Generalized template akan dibuat kemudian dengan build_generalized_templates()
        """
        if vowel not in self.vowels:
            raise ValueError(f"Vokal harus salah satu dari {self.vowels}")
        
        features = self.extract_mfcc_39(audio_path)
        
        if vowel not in self.raw_templates:
            self.raw_templates[vowel] = {}
        
        if person_id not in self.raw_templates[vowel]:
            self.raw_templates[vowel][person_id] = []
        
        self.raw_templates[vowel][person_id].append(features)
        print(f"Raw template ditambahkan: {vowel} dari {person_id} ({len(self.raw_templates[vowel][person_id])} file)")
    
    def recognize(self, audio_path, use_generalized=True):
        """
        Mengenali vokal dari file audio menggunakan generalized templates
        
        Parameters:
        - use_generalized: jika True, gunakan generalized templates
        """
        test_features = self.extract_mfcc_39(audio_path)
        
        if use_generalized and len(self.generalized_templates) == 0:
            raise ValueError("Generalized templates belum dibuat. Panggil build_generalized_templates() dulu.")
        
        min_distance = float('inf')
        recognized_vowel = None
        all_distances = {}
        
        # Gunakan generalized templates untuk recognition
        for vowel in self.vowels:
            if vowel not in self.generalized_templates:
                continue
            
            template_features = self.generalized_templates[vowel]
            distance = self.dtw_distance(template_features, test_features)
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
        
        # Show distances to all vowels (sorted)
        print(f"    All distances:")
        sorted_distances = sorted(final_vowel_distances.items(), key=lambda x: x[1])
        for vowel, dist in sorted_distances:
            marker = " <- CHOSEN" if vowel == recognized_vowel else ""
            correct_marker = " (CORRECT)" if vowel == true_vowel else ""
            print(f"      {vowel}: {dist:.4f}{marker}{correct_marker}")
    
    def test_closed_scenario(self, test_data):
        """
        Skenario Closed - menggunakan semua raw templates untuk generalized template
        """
        print(f"\n--- BUILDING GENERALIZED TEMPLATES FOR CLOSED SCENARIO ---")
        self.build_generalized_templates(target_person_ids=None)  # Gunakan semua person
        
        correct = 0
        total = 0
        results = []
        
        print(f"\n--- DETAILED CLOSED SCENARIO RESULTS ---")
        
        for audio_path, true_vowel, test_person_id in test_data:
            recognized_vowel, distance, all_distances, final_vowel_distances = self.recognize(audio_path, use_generalized=True)
            
            is_correct = (recognized_vowel == true_vowel)
            if is_correct:
                correct += 1
            total += 1
            
            # Print detailed prediction info
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
        Skenario Open - menggunakan hanya template dari person tertentu
        """
        print(f"\n--- BUILDING GENERALIZED TEMPLATES FOR OPEN SCENARIO ---")
        print(f"Using templates from: {template_person_ids}")
        self.build_generalized_templates(target_person_ids=template_person_ids)
        
        correct = 0
        total = 0
        results = []
        
        print(f"\n--- DETAILED OPEN SCENARIO RESULTS ---")
        
        for audio_path, true_vowel, test_person_id in test_data:
            recognized_vowel, distance, all_distances, final_vowel_distances = self.recognize(audio_path, use_generalized=True)
            
            is_correct = (recognized_vowel == true_vowel)
            if is_correct:
                correct += 1
            total += 1
            
            # Print detailed prediction info
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
        
        accuracy = (correct / total) * 100 if total > 0 else 0
        print(f"\n  OPEN SCENARIO SUMMARY: {correct}/{total} correct = {accuracy:.2f}%")
        return accuracy, results
    
    def evaluate_all_scenarios(self, test_data_us, test_data_other):
        """
        Evaluasi lengkap: closed, open, dan rata-rata
        """
        # Get all persons from templates_us test data
        all_persons = list(set([person_id for _, _, person_id in test_data_us]))
        results_summary = {}
        
        for template_person in all_persons:
            print(f"\n{'='*60}")
            print(f"=== EVALUATION WITH TEMPLATES FROM {template_person.upper()} ===")
            print(f"{'='*60}")
            
            print(f"\nTotal test files (templates_us): {len(test_data_us)}")
            print(f"Total test files (templates_other): {len(test_data_other)}")
            
            # Closed scenario testing (all templates vs test from templates_us only)
            print(f"\n[CLOSED SCENARIO] (all raw templates → generalized templates vs test from templates_us)")
            closed_acc, closed_results = self.test_closed_scenario(test_data_us)
            
            # Open scenario testing (templates from template_person vs test from templates_other)
            open_test_data = test_data_other
            
            print(f"\n[OPEN SCENARIO] (raw templates from {template_person} → generalized templates vs test from templates_other)")
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
        
        # Calculate overall statistics
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

    # Properti untuk kompatibilitas dengan kode lama
    @property
    def templates(self):
        """Kompatibilitas dengan kode lama yang mengakses self.templates"""
        return self.raw_templates
    
    def get_generalized_template_info(self):
        """
        Mendapatkan informasi tentang generalized templates
        """
        info = {}
        for vowel, template in self.generalized_templates.items():
            info[vowel] = {
                'shape': template.shape,
                'total_frames': template.shape[0] if len(template.shape) > 0 else 0
            }
        return info