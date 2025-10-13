import os
import json
from typing import List, Tuple

from VowelDTW.VowelRecognitionDTW import VowelRecognitionDTW
from VowelDTW.VowelDTWVisualizer import VowelDTWVisualizer

RESULTS_DIR = "results"
IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

def scan_person_folder(folder: str, vowels: List[str]):
    result = {}
    if not os.path.exists(folder):
        print(f"Folder tidak ditemukan: {folder}")
        return result

    for filename in os.listdir(folder):
        if not (filename.endswith(".wav") or filename.endswith(".m4a")):
            continue
        parts = filename.replace('.wav', '').replace('.m4a', '').split(' - ')
        if len(parts) < 2:
            continue
        vowel_num = parts[1].strip().split()
        if len(vowel_num) < 2:
            continue

        vowel_part = vowel_num[0].lower()
        try:
            file_num = int(vowel_num[1])
        except ValueError:
            print(f"Warning: Could not parse number from {filename}")
            continue

        if vowel_part in vowels:
            result.setdefault(vowel_part, []).append((os.path.join(folder, filename), file_num))
    return result


if __name__ == "__main__":
    persons = ['densu', 'hira', 'naufal', 'wiga', 'zya']
    vowels = ['a', 'i', 'u', 'e', 'o']
    base_path = ""  

    recognizer = VowelRecognitionDTW(sample_rate=16000, use_vad=True, normalize=True)
    viz = VowelDTWVisualizer(recognizer)

    # Scan templates_us 
    print("=== SCANNING FILES (templates_us) ===")
    all_files = {}
    for person in persons:
        person_folder = os.path.join(f"{base_path}templates_us", person)
        all_files[person] = scan_person_folder(person_folder, vowels)

    # Load templates & build test set (US) 
    print("\n=== LOADING TEMPLATES (using files with smaller numbers) ===")
    test_data_us: List[Tuple[str, str, str]] = []
    for person in persons:
        for vowel in vowels:
            files = all_files.get(person, {}).get(vowel, [])
            files.sort(key=lambda x: x[1])
            if len(files) == 0:
                continue

            # semua kecuali terakhir -> template, terakhir -> test
            template_files = files[:-1] if len(files) > 1 else []
            test_file = files[-1]

            for file_path, _ in template_files:
                try:
                    recognizer.add_template(vowel, person, file_path)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

            test_data_us.append((test_file[0], vowel, person))

    print(f"\n=== TEST DATA (templates_us - using files with largest numbers) ===")
    print(f"Total: {len(test_data_us)} test files")

    # Scan templates_other 
    print("\n=== SCANNING FILES (templates_other) ===")
    all_files_other = {}
    folder_other = f"{base_path}templates_other"
    if os.path.exists(folder_other):
        for person in os.listdir(folder_other):
            person_folder = os.path.join(folder_other, person)
            if not os.path.isdir(person_folder):
                continue
            all_files_other[person] = scan_person_folder(person_folder, vowels)
    else:
        print(f"Folder tidak ditemukan: {folder_other}")

    # Build test_data_other dari file terakhir tiap vowel
    test_data_other: List[Tuple[str, str, str]] = []
    print("\n=== TEST DATA (templates_other - using files with largest numbers) ===")
    for person in all_files_other:
        for vowel in vowels:
            files = all_files_other[person].get(vowel, [])
            files.sort(key=lambda x: x[1])
            if len(files) == 0:
                continue
            test_file = files[-1]
            test_data_other.append((test_file[0], vowel, person))
    print(f"Total: {len(test_data_other)} test files")

    # Contoh Visualisasi untuk satu sampel 
    if len(test_data_us) > 0:
        sample_path, sample_true, sample_person = test_data_us[0]
        sample_base = os.path.splitext(os.path.basename(sample_path))[0]

        # Waveform + VAD & MFCC-39
        viz.plot_waveform_with_vad(
            sample_path,
            top_db=20,
            save_path=os.path.join(IMAGES_DIR, f"{sample_base}_waveform_vad.png")
        )
        viz.plot_mfcc39(
            sample_path,
            save_prefix=os.path.join(IMAGES_DIR, f"{sample_base}_mfcc39")
        )

        # DTW alignment terhadap salah satu template dari vokal yang sama
        if sample_true in recognizer.templates and len(recognizer.templates[sample_true]) > 0:
            first_pid, templ_list = next(iter(recognizer.templates[sample_true].items()))
            if len(templ_list) > 0:
                template_feats = templ_list[0]
                test_feats = recognizer.extract_mfcc_39(sample_path)
                viz.plot_dtw_alignment(
                    template_feats, test_feats,
                    title=f"DTW {sample_true.upper()} | template:{first_pid} vs test:{sample_person}",
                    save_path=os.path.join(IMAGES_DIR, f"{sample_base}_dtw_alignment.png")
                )

        # Bar distances untuk sampel ini
        pred, dist, _, final_d = recognizer.recognize(sample_path, use_averaging=True)
        viz.plot_vowel_distances_bar(
            final_d,
            title=f"Distances for {os.path.basename(sample_path)} (true={sample_true}, pred={pred})",
            save_path=os.path.join(IMAGES_DIR, f"{sample_base}_vowel_distances.png")
        )

    # Evaluasi lengkap 
    print("\n=== STARTING EVALUATION ===")
    results = recognizer.evaluate_all_scenarios(test_data_us, test_data_other)

    # Confusion matrix (closed) sebagai heatmap
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

    # Simpan ringkasan angka ke JSON (ke folder results/)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, 'results.json')
    with open(json_path, 'w') as f:
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
        json.dump(json_results, f, indent=2)
    print(f"\nHasil disimpan ke {json_path}")
