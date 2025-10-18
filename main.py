import csv
import os
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

from VowelDTW.BuildRecognizer import build_recognizer_from_folders
from VowelDTW.SafeVisualizer import SafeVisualizer, create_comprehensive_visualizations

def scan_person_folder(folder: str, vowels: List[str]) -> Dict[str, List[Tuple[str, int]]]:
    result: Dict[str, List[Tuple[str, int]]] = {}
    if not os.path.exists(folder):
        print(f"Folder tidak ditemukan: {folder}")
        return result

    for filename in os.listdir(folder):
        if not (filename.lower().endswith(".wav") or filename.lower().endswith(".m4a")):
            continue
        name_wo_ext = filename.rsplit(".", 1)[0]
        parts = name_wo_ext.split(" - ")
        if len(parts) < 2:
            continue

        # ekspektasi: "<person> - <vowel> <num>"
        vowel_num = parts[1].strip().split()
        if len(vowel_num) < 2:
            continue

        vowel_part = vowel_num[0].lower()
        try:
            file_num = int(re.sub(r'\D+', '', vowel_num[1]))
        except ValueError:
            print(f"Warning: Could not parse number from {filename}")
            continue

        if vowel_part in vowels:
            result.setdefault(vowel_part, []).append((os.path.join(folder, filename), file_num))

    # sort each list by num asc
    for v in list(result.keys()):
        result[v].sort(key=lambda x: x[1])
    return result


def scan_person_folder_recursive(root: str, vowels: List[str]) -> Dict[str, List[Tuple[str, int]]]:
    out: Dict[str, List[Tuple[str, int]]] = {}
    rootp = Path(root)
    if not rootp.exists():
        print(f"[WARN] Folder not found: {root}")
        return out

    for f in rootp.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in (".wav", ".m4a"):
            continue

        base = f.stem
        if " - " not in base:
            continue

        right = base.split(" - ", 1)[1].strip()
        parts = right.split()
        if not parts:
            continue

        vtok = parts[0].lower()
        if vtok not in vowels:
            continue

        num = 0
        if len(parts) > 1:
            try:
                num = int(re.sub(r"\D+", "", parts[1]))
            except Exception:
                num = 0

        out.setdefault(vtok, []).append((str(f), num))

    for v in list(out.keys()):
        out[v].sort(key=lambda x: x[1])
    return out


def person_id_from_name(filepath: str) -> str:
    base = Path(filepath).stem
    parts = base.split(' - ')
    return parts[0].strip().lower() if parts else "unknown"


def diagnose_templates_dir(root: str, exts=(".wav", ".m4a")):
    p = Path(root)
    print("Exists:", p.exists(), "| Dir:", p.is_dir(), "| Path:", p.resolve())
    if not p.exists():
        return
    print("Immediate entries:", [x.name for x in p.iterdir()])
    files = [str(f) for f in p.rglob("*") if f.suffix.lower() in exts]
    print(f"Found {len(files)} audio file(s) under {root} (recursive).")
    for s in files[:20]:
        print("  -", s)


def export_predictions_csv(recognizer, test_data: List[Tuple[str, str, str]], out_csv_path: str):
    rows = []
    for wav_path, true_vowel, person in test_data:
        try:
            pred, _, _, final_d = recognizer.recognize(wav_path)
        except Exception as e:
            print(f"[WARN] recognize failed for {wav_path}: {e}")
            pred, final_d = None, {}

        row = {
            "file": os.path.basename(wav_path),
            "path": wav_path,
            "person": person,
            "true_vowel": true_vowel,
            "pred_vowel": pred,
        }
        for k, v in final_d.items():
            try:
                row[f"dist_{k}"] = float(v)
            except Exception:
                row[f"dist_{k}"] = None
        rows.append(row)

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    fieldnames = ["file", "path", "person", "true_vowel", "pred_vowel"]
    extra_cols = sorted({k for r in rows for k in r.keys() if k.startswith("dist_")})
    fieldnames.extend(extra_cols)

    with open(out_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_test_data_from_files(persons: List[str],
                              all_files: Dict[str, Dict[str, List[Tuple[str, int]]]],
                              vowels: List[str],
                              use_last_file: bool = True) -> List[Tuple[str, str, str]]:
    test_data: List[Tuple[str, str, str]] = []
    for person in persons:
        for vowel in vowels:
            files = (all_files.get(person, {}) or {}).get(vowel, [])
            if not files:
                continue
            test_file = files[-1] if use_last_file else files[0]
            test_data.append((test_file[0], vowel, person))
    return test_data


def add_templates_from_files(recognizer,
                            persons: List[str],
                            all_files: Dict[str, Dict[str, List[Tuple[str, int]]]],
                            max_templates_per_speaker: Optional[int] = None,
                            exclude_last: bool = True):
    for person in persons:
        for vowel, files in (all_files.get(person, {}) or {}).items():
            template_files = files[:-1] if exclude_last and len(files) > 1 else files
            if max_templates_per_speaker:
                template_files = template_files[:max_templates_per_speaker]
            for file_path, _ in template_files:
                try:
                    recognizer.add_template(vowel, person, file_path)
                except Exception as e:
                    print(f"[WARN] Error loading template {file_path}: {e}")


def create_si_closed_recognizer(base_recognizer, exclude_person: str):
    from VowelDTW.VowelRecognitionDTW import VowelRecognitionDTW
    
    si_recognizer = VowelRecognitionDTW(
        sample_rate=base_recognizer.sample_rate,
        use_vad=base_recognizer.use_vad,
        normalize=base_recognizer.normalize,
        n_segments=base_recognizer.n_segments
    )
    
    templates_added = 0
    for vowel, person_dict in base_recognizer.templates.items():
        for person, template_list in person_dict.items():
            if person != exclude_person:  
                for template_features in template_list:
                    if vowel not in si_recognizer.templates:
                        si_recognizer.templates[vowel] = {}
                    if person not in si_recognizer.templates[vowel]:
                        si_recognizer.templates[vowel][person] = []
                    
                    si_recognizer.templates[vowel][person].append(template_features)
                    templates_added += 1
    
    if templates_added > 0:
        si_recognizer.build_generalized_templates()
        print(f"SI-Closed: Built generalized templates excluding {exclude_person} ({templates_added} templates)")
    else:
        print(f"SI-Closed: No templates available after excluding {exclude_person}")
    
    return si_recognizer


def predict_excluding_same_speaker(recognizer, wav_path: str, test_person: str):
    try:
        si_recognizer = create_si_closed_recognizer(recognizer, test_person)
        
        if not si_recognizer.generalized_templates:
            print(f"[WARN] No generalized templates available after excluding {test_person}")
            return None, {}
        
        pred, min_dist, all_distances, final_vowel_distances = si_recognizer.recognize(wav_path)
        return pred, final_vowel_distances
        
    except Exception as e:
        print(f"[ERROR] SI-Closed prediction failed for {wav_path}: {e}")
        pred, min_dist, all_distances, final_vowel_distances = recognizer.recognize(wav_path)
        return pred, final_vowel_distances


def evaluate_si_closed(recognizer, test_data_us: List[Tuple[str, str, str]]):
    total, correct = 0, 0
    results_dicts: List[Dict[str, str]] = []
    
    print(f"\n=== Evaluasi SI-Closed dengan Generalized Templates ===")
    print(f"Total test files: {len(test_data_us)}")
    
    for i, (wav_path, true_vowel, person) in enumerate(test_data_us):
        print(f"Processing {i+1}/{len(test_data_us)}: {os.path.basename(wav_path)} (person: {person})")
        
        pred, final_distances = predict_excluding_same_speaker(recognizer, wav_path, person)
        
        if pred is not None:
            results_dicts.append({"true": true_vowel, "predicted": pred})
            total += 1
            if pred == true_vowel:
                correct += 1
                print(f"Correct: {true_vowel} -> {pred}")
            else:
                print(f"Wrong: {true_vowel} -> {pred}")
        else:
            print(f"Skipped: No templates available after excluding {person}")
    
    acc = (correct / total) * 100.0 if total > 0 else 0.0
    print(f"\nSI-Closed Results: {correct}/{total} = {acc:.2f}%")
    
    return acc, results_dicts

# Configuration
class Config:    
    # Paths
    BASE_PATH = ""
    TEMPLATES_US_PATH = "templates_us"
    TEMPLATES_OTHER_PATH = "templates_other"
    RESULTS_DIR = "results"
    IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
    
    # Experiment settings
    PERSONS = ['densu', 'hira', 'naufal', 'wiga', 'zya']
    VOWELS = ['a', 'i', 'u', 'e', 'o']
    
    # Recognizer parameters
    SAMPLE_RATE = 16000
    USE_VAD = True
    NORMALIZE = True
    N_SEGMENTS = 3
    MAX_TEMPLATES_PER_SPEAKER: Optional[int] = None  # None = use all available
    
    # Visualization settings
    MAX_SAMPLES_PER_SCENARIO = 10
    
    def __init__(self):
        # Ensure directories exist
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        os.makedirs(self.IMAGES_DIR, exist_ok=True)
    
    @property
    def templates_us_full_path(self):
        return os.path.join(self.BASE_PATH, self.TEMPLATES_US_PATH)
    
    @property
    def templates_other_full_path(self):
        return os.path.join(self.BASE_PATH, self.TEMPLATES_OTHER_PATH)


def run_diagnostics(config: Config):
    """Run diagnostic checks on the directory structure"""
    print("=" * 60)
    print("=== RUNNING DIAGNOSTICS ===")
    print("=" * 60)
    
    print(f"\n--- Checking templates_us directory ---")
    diagnose_templates_dir(config.templates_us_full_path)
    
    if os.path.exists(config.templates_other_full_path):
        print(f"\n--- Checking templates_other directory ---")
        diagnose_templates_dir(config.templates_other_full_path)
    else:
        print(f"\n--- templates_other directory not found: {config.templates_other_full_path} ---")


def build_recognizer(config: Config):
    """Build the recognizer with templates"""
    print("\n" + "=" * 60)
    print("=== BUILDING RECOGNIZER ===")
    print("=" * 60)
    
    # Use the builder to create recognizer
    recognizer, test_data_templates, test_data_others = build_recognizer_from_folders(
        template_base_path=config.templates_us_full_path,
        template_persons=config.PERSONS,
        test_base_path=config.templates_other_full_path,
        sample_rate=config.SAMPLE_RATE,
        use_vad=config.USE_VAD,
        normalize=config.NORMALIZE,
        n_segments=config.N_SEGMENTS,
        max_templates_per_speaker=config.MAX_TEMPLATES_PER_SPEAKER
    )
    
    return recognizer, test_data_templates, test_data_others


def run_evaluation(recognizer, test_data_templates, test_data_others, config: Config):
    """Run comprehensive evaluation"""
    print("\n" + "=" * 60)
    print("=== RUNNING EVALUATION ===")
    print("=" * 60)
    
    # Main evaluation using recognizer's built-in method
    results_summary = recognizer.evaluate_all_scenarios(test_data_templates, test_data_others)
    
    # Additional SI-Closed evaluation
    print(f"\n--- SI-Closed Evaluation ---")
    si_closed_acc, si_closed_results = evaluate_si_closed(recognizer, test_data_templates)
    print(f"SI-Closed Accuracy: {si_closed_acc:.2f}%")
    
    # Add SI-Closed results to summary
    results_summary['si_closed'] = {
        'si_closed_accuracy': si_closed_acc,
        'si_closed_results': si_closed_results
    }
    
    return results_summary


def export_results(recognizer, test_data_templates, test_data_others, results_summary, config: Config):
    """Export results to CSV and JSON"""
    print("\n" + "=" * 60)
    print("=== EXPORTING RESULTS ===")
    print("=" * 60)
    
    # Export prediction CSVs
    if test_data_templates:
        csv_path_closed = os.path.join(config.RESULTS_DIR, "predictions_closed.csv")
        export_predictions_csv(recognizer, test_data_templates, csv_path_closed)
        print(f"Closed predictions saved to: {csv_path_closed}")
    
    if test_data_others:
        csv_path_open = os.path.join(config.RESULTS_DIR, "predictions_open.csv")
        export_predictions_csv(recognizer, test_data_others, csv_path_open)
        print(f"Open predictions saved to: {csv_path_open}")
    
    # Export JSON summary
    json_path = os.path.join(config.RESULTS_DIR, 'results.json')
    json_results = {}
    
    for k, v in results_summary.items():
        if k == 'overall':
            json_results[k] = v
        elif k == 'si_closed':
            json_results[k] = {'si_closed_accuracy': v.get('si_closed_accuracy')}
        else:
            if isinstance(v, dict):
                json_results[k] = {
                    'closed_accuracy': v.get('closed_accuracy'),
                    'open_accuracy': v.get('open_accuracy'),
                    'average_accuracy': v.get('average_accuracy')
                }
    
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results summary saved to: {json_path}")


def create_visualizations(recognizer, test_data_templates, test_data_others, results_summary, config: Config):
    """Create comprehensive visualizations"""
    print("\n" + "=" * 60)
    print("=== CREATING VISUALIZATIONS ===")
    print("=" * 60)
    
    visualization_summary = create_comprehensive_visualizations(
        recognizer=recognizer,
        test_data_templates=test_data_templates,
        test_data_others=test_data_others,
        results_summary=results_summary,
        images_dir=config.IMAGES_DIR,
        max_samples_per_scenario=config.MAX_SAMPLES_PER_SCENARIO
    )
    
    # Additional confusion matrix for SI-Closed if available
    if 'si_closed' in results_summary and 'si_closed_results' in results_summary['si_closed']:
        visualizer = SafeVisualizer(recognizer, config.IMAGES_DIR)
        si_results = results_summary['si_closed']['si_closed_results']
        if si_results:
            visualizer.safe_plot_confusion_heatmap(
                si_results,
                title="Confusion Matrix (SI-Closed: exclude same speaker)",
                save_path=os.path.join(config.IMAGES_DIR, "confusion_matrix_si_closed.png")
            )
    
    return visualization_summary


def print_final_summary(results_summary, config: Config):
    """Print final summary of results"""
    print("\n" + "=" * 80)
    print("=== FINAL SUMMARY ===")
    print("=" * 80)
    
    if 'overall' in results_summary:
        overall = results_summary['overall']
        print(f"\nOverall Results:")
        print(f"  Closed Accuracy:  {overall.get('closed_accuracy', 0):.2f}%")
        print(f"  Open Accuracy:    {overall.get('open_accuracy', 0):.2f}%")
        print(f"  Average Accuracy: {overall.get('average_accuracy', 0):.2f}%")
    
    if 'si_closed' in results_summary:
        si_acc = results_summary['si_closed'].get('si_closed_accuracy', 0)
        print(f"  SI-Closed Accuracy: {si_acc:.2f}%")
    
    print(f"\nResults saved to: {config.RESULTS_DIR}/")
    print(f"  - CSV predictions: predictions_closed.csv, predictions_open.csv")
    print(f"  - JSON summary: results.json")
    print(f"  - Visualizations: {config.IMAGES_DIR}/")
    
    print(f"\nExperiment completed successfully!")


def main():
    """Main execution function"""
    print("Starting Vowel Recognition System Evaluation")
    print("=" * 80)
    
    # Initialize configuration
    config = Config()
    
    try:
        # Step 1: Run diagnostics
        run_diagnostics(config)
        
        # Step 2: Build recognizer
        recognizer, test_data_templates, test_data_others = build_recognizer(config)
        
        # Step 3: Run evaluation
        results_summary = run_evaluation(recognizer, test_data_templates, test_data_others, config)
        
        # Step 4: Export results
        export_results(recognizer, test_data_templates, test_data_others, results_summary, config)
        
        # Step 5: Create visualizations
        visualization_summary = create_visualizations(
            recognizer, test_data_templates, test_data_others, results_summary, config
        )
        
        # Step 6: Print final summary
        print_final_summary(results_summary, config)
        
    except Exception as e:
        print(f"\nERROR: Experiment failed with exception:")
        print(f"  {type(e).__name__}: {e}")
        print(f"\nCheck your data paths and file structure.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)