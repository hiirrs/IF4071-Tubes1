import os
import json
from typing import List, Tuple
from datetime import datetime
import sys

from VowelDTW.VowelRecognitionDTW import VowelRecognitionDTW
from VowelDTW.VowelDTWVisualizer import VowelDTWVisualizer

RESULTS_DIR = "results"
IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

class OutputLogger:
    """Class untuk menangkap dan menyimpan semua output"""
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.log_entries = []
        
    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level}] {message}"
        self.log_entries.append(entry)
        print(message)  # Tetap tampilkan di console
        
    def save_log(self):
        with open(self.log_file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.log_entries))

def scan_person_folder(folder: str, vowels: List[str], logger: OutputLogger):
    result = {}
    if not os.path.exists(folder):
        logger.log(f"Folder tidak ditemukan: {folder}", "WARNING")
        return result

    logger.log(f"Scanning folder: {folder}")
    file_count = 0
    
    for filename in os.listdir(folder):
        if not (filename.endswith(".wav") or filename.endswith(".m4a")):
            continue
            
        file_count += 1
        parts = filename.replace('.wav', '').replace('.m4a', '').split(' - ')
        if len(parts) < 2:
            logger.log(f"Skipping file with invalid format: {filename}", "WARNING")
            continue
            
        vowel_num = parts[1].strip().split()
        if len(vowel_num) < 2:
            logger.log(f"Skipping file with invalid vowel format: {filename}", "WARNING")
            continue

        vowel_part = vowel_num[0].lower()
        try:
            file_num = int(vowel_num[1])
        except ValueError:
            logger.log(f"Warning: Could not parse number from {filename}", "WARNING")
            continue

        if vowel_part in vowels:
            result.setdefault(vowel_part, []).append((os.path.join(folder, filename), file_num))
            
    logger.log(f"Found {file_count} audio files in {folder}")
    return result

def save_detailed_results(results, test_data_us, test_data_other, output_path):
    """Menyimpan hasil detail ke file JSON"""
    detailed_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_test_files_us": len(test_data_us),
            "total_test_files_other": len(test_data_other)
        },
        "test_data": {
            "us_files": [
                {
                    "file_path": path,
                    "true_vowel": vowel,
                    "person": person
                } for path, vowel, person in test_data_us
            ],
            "other_files": [
                {
                    "file_path": path,
                    "true_vowel": vowel,
                    "person": person
                } for path, vowel, person in test_data_other
            ]
        },
        "detailed_results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

def save_template_info(recognizer, output_path, logger):
    """Menyimpan informasi template yang dimuat"""
    template_info = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_vowels": len(recognizer.vowels)
        },
        "templates": {}
    }
    
    total_templates = 0
    for vowel in recognizer.vowels:
        if vowel in recognizer.templates:
            template_info["templates"][vowel] = {}
            for person, templates in recognizer.templates[vowel].items():
                template_info["templates"][vowel][person] = {
                    "count": len(templates),
                    "feature_shapes": [list(t.shape) for t in templates]
                }
                total_templates += len(templates)
    
    template_info["metadata"]["total_templates"] = total_templates
    logger.log(f"Total templates loaded: {total_templates}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template_info, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"execution_log_{timestamp}.txt")
    detailed_results_file = os.path.join(RESULTS_DIR, f"detailed_results_{timestamp}.json")
    template_info_file = os.path.join(RESULTS_DIR, f"template_info_{timestamp}.json")
    
    logger = OutputLogger(log_file)
    
    persons = ['densu', 'hira', 'naufal', 'wiga', 'zya']
    vowels = ['a', 'i', 'u', 'e', 'o']
    base_path = ""  

    logger.log("=== MEMULAI EKSPERIMEN VOWEL RECOGNITION ===")
    logger.log(f"Persons: {persons}")
    logger.log(f"Vowels: {vowels}")

    recognizer = VowelRecognitionDTW(sample_rate=16000, use_vad=True, normalize=True)
    viz = VowelDTWVisualizer(recognizer)

    # Scan templates_us 
    logger.log("=== SCANNING FILES (templates_us) ===")
    all_files = {}
    for person in persons:
        person_folder = os.path.join(f"{base_path}templates_us", person)
        all_files[person] = scan_person_folder(person_folder, vowels, logger)
        
        # Log statistik per person
        total_files = sum(len(files) for files in all_files[person].values())
        logger.log(f"Person {person}: {total_files} total files")
        for vowel, files in all_files[person].items():
            logger.log(f"  - Vowel {vowel}: {len(files)} files")

    # Load templates & build test set (US) 
    logger.log("\n=== LOADING TEMPLATES (using files with smaller numbers) ===")
    test_data_us: List[Tuple[str, str, str]] = []
    templates_loaded = 0
    
    for person in persons:
        logger.log(f"Processing person: {person}")
        for vowel in vowels:
            files = all_files.get(person, {}).get(vowel, [])
            files.sort(key=lambda x: x[1])
            if len(files) == 0:
                logger.log(f"  No files found for vowel {vowel}", "WARNING")
                continue

            # semua kecuali terakhir -> template, terakhir -> test
            template_files = files[:-1] if len(files) > 1 else []
            test_file = files[-1]

            logger.log(f"  Vowel {vowel}: {len(template_files)} templates, 1 test file")
            
            for file_path, file_num in template_files:
                try:
                    recognizer.add_template(vowel, person, file_path)
                    templates_loaded += 1
                    logger.log(f"    Loaded template: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.log(f"    Error loading {file_path}: {e}", "ERROR")

            test_data_us.append((test_file[0], vowel, person))
            logger.log(f"    Test file: {os.path.basename(test_file[0])}")

    logger.log(f"\nTotal templates loaded: {templates_loaded}")
    logger.log(f"Total US test files: {len(test_data_us)}")
    
    # Simpan informasi template
    save_template_info(recognizer, template_info_file, logger)

    # Scan templates_other 
    logger.log("\n=== SCANNING FILES (templates_other) ===")
    all_files_other = {}
    folder_other = f"{base_path}templates_other"
    if os.path.exists(folder_other):
        for person in os.listdir(folder_other):
            person_folder = os.path.join(folder_other, person)
            if not os.path.isdir(person_folder):
                continue
            logger.log(f"Processing other person: {person}")
            all_files_other[person] = scan_person_folder(person_folder, vowels, logger)
            
            # Log statistik per person
            total_files = sum(len(files) for files in all_files_other[person].values())
            logger.log(f"Person {person}: {total_files} total files")
    else:
        logger.log(f"Folder tidak ditemukan: {folder_other}", "WARNING")

    # Build test_data_other dari file terakhir tiap vowel
    test_data_other: List[Tuple[str, str, str]] = []
    logger.log("\n=== TEST DATA (templates_other - using files with largest numbers) ===")
    for person in all_files_other:
        logger.log(f"Processing other person test files: {person}")
        for vowel in vowels:
            files = all_files_other[person].get(vowel, [])
            files.sort(key=lambda x: x[1])
            if len(files) == 0:
                continue
            test_file = files[-1]
            test_data_other.append((test_file[0], vowel, person))
            logger.log(f"  Test file {vowel}: {os.path.basename(test_file[0])}")
            
    logger.log(f"Total other test files: {len(test_data_other)}")

    # Contoh Visualisasi untuk satu sampel 
    logger.log("\n=== GENERATING SAMPLE VISUALIZATIONS ===")
    if len(test_data_us) > 0:
        sample_path, sample_true, sample_person = test_data_us[0]
        sample_base = os.path.splitext(os.path.basename(sample_path))[0]
        
        logger.log(f"Creating visualizations for sample: {sample_base}")

        # Waveform + VAD & MFCC-39
        try:
            viz.plot_waveform_with_vad(
                sample_path,
                top_db=20,
                save_path=os.path.join(IMAGES_DIR, f"{sample_base}_waveform_vad.png")
            )
            logger.log("  Waveform + VAD visualization created")
        except Exception as e:
            logger.log(f"  Error creating waveform visualization: {e}", "ERROR")
            
        try:
            viz.plot_mfcc39(
                sample_path,
                save_prefix=os.path.join(IMAGES_DIR, f"{sample_base}_mfcc39")
            )
            logger.log("  MFCC-39 visualization created")
        except Exception as e:
            logger.log(f"  Error creating MFCC visualization: {e}", "ERROR")

        # DTW alignment terhadap salah satu template dari vokal yang sama
        if sample_true in recognizer.templates and len(recognizer.templates[sample_true]) > 0:
            try:
                first_pid, templ_list = next(iter(recognizer.templates[sample_true].items()))
                if len(templ_list) > 0:
                    template_feats = templ_list[0]
                    test_feats = recognizer.extract_mfcc_39(sample_path)
                    viz.plot_dtw_alignment(
                        template_feats, test_feats,
                        title=f"DTW {sample_true.upper()} | template:{first_pid} vs test:{sample_person}",
                        save_path=os.path.join(IMAGES_DIR, f"{sample_base}_dtw_alignment.png")
                    )
                    logger.log("  DTW alignment visualization created")
            except Exception as e:
                logger.log(f"  Error creating DTW visualization: {e}", "ERROR")

        # Bar distances untuk sampel ini
        try:
            pred, dist, _, final_d = recognizer.recognize(sample_path, use_averaging=True)
            viz.plot_vowel_distances_bar(
                final_d,
                title=f"Distances for {os.path.basename(sample_path)} (true={sample_true}, pred={pred})",
                save_path=os.path.join(IMAGES_DIR, f"{sample_base}_vowel_distances.png")
            )
            logger.log(f"  Distance bar chart created (true={sample_true}, pred={pred})")
        except Exception as e:
            logger.log(f"  Error creating distance visualization: {e}", "ERROR")

    # Evaluasi lengkap 
    logger.log("\n=== STARTING EVALUATION ===")
    logger.log("This may take several minutes...")
    
    try:
        results = recognizer.evaluate_all_scenarios(test_data_us, test_data_other)
        logger.log("Evaluation completed successfully")
        
        # Log summary results
        logger.log("\n=== EVALUATION SUMMARY ===")
        for person, person_results in results.items():
            if person != 'overall':
                logger.log(f"Person {person}:")
                logger.log(f"  Closed accuracy: {person_results['closed_accuracy']:.3f}")
                logger.log(f"  Open accuracy: {person_results['open_accuracy']:.3f}")
                logger.log(f"  Average accuracy: {person_results['average_accuracy']:.3f}")
        
        if 'overall' in results:
            logger.log("Overall results:")
            for key, value in results['overall'].items():
                logger.log(f"  {key}: {value:.3f}")
                
    except Exception as e:
        logger.log(f"Error during evaluation: {e}", "ERROR")
        results = {}

    # Confusion matrix (closed) sebagai heatmap
    logger.log("\n=== GENERATING CONFUSION MATRIX ===")
    try:
        all_closed_results = []
        for person_results in results.values():
            if isinstance(person_results, dict) and 'closed_results' in person_results:
                all_closed_results.extend(person_results['closed_results'])
                
        if len(all_closed_results) > 0:
            viz.plot_confusion_heatmap(
                all_closed_results,
                vowels=recognizer.vowels,
                title="Confusion Matrix (Closed Scenario)",
                save_path=os.path.join(IMAGES_DIR, "confusion_matrix_closed.png")
            )
            logger.log("Confusion matrix heatmap created")
        else:
            logger.log("No closed results available for confusion matrix", "WARNING")
    except Exception as e:
        logger.log(f"Error creating confusion matrix: {e}", "ERROR")

    # Simpan hasil detail
    logger.log("\n=== SAVING RESULTS ===")
    try:
        save_detailed_results(results, test_data_us, test_data_other, detailed_results_file)
        logger.log(f"Detailed results saved to {detailed_results_file}")
    except Exception as e:
        logger.log(f"Error saving detailed results: {e}", "ERROR")

    # Simpan ringkasan angka ke JSON (ke folder results/)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, 'results.json')
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json_results = {}
            for k, v in results.items():
                if k != 'overall':
                    json_results[k] = {
                        'closed_accuracy': v['closed_accuracy'],
                        'open_accuracy': v['open_accuracy'],
                        'average_accuracy': v['average_accuracy']
                    }
                else:
                    json_results[k] = v
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        logger.log(f"Summary results saved to {json_path}")
    except Exception as e:
        logger.log(f"Error saving summary results: {e}", "ERROR")

    # Simpan log eksekusi
    logger.log(f"\n=== EXPERIMENT COMPLETED ===")
    logger.log(f"Execution log: {log_file}")
    logger.log(f"Detailed results: {detailed_results_file}")
    logger.log(f"Template info: {template_info_file}")
    logger.log(f"Summary results: {json_path}")
    logger.log(f"Images directory: {IMAGES_DIR}")
    
    try:
        logger.save_log()
        print(f"\nAll outputs saved successfully!")
    except Exception as e:
        print(f"Error saving execution log: {e}")