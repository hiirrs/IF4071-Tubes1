class SafeVisualizer:    
    def __init__(self, recognizer, images_dir: str = "results/images"):
        self.viz = VowelDTWVisualizer(recognizer)
        self.recognizer = recognizer
        self.images_dir = images_dir
        os.makedirs(images_dir, exist_ok=True)
    
    def safe_plot_waveform_vad(self, wav_path: str, save_path: Optional[str] = None, top_db: int = 20):
        try:
            if save_path is None:
                base = os.path.splitext(os.path.basename(wav_path))[0]
                save_path = os.path.join(self.images_dir, f"{base}_waveform_vad.png")
            self.viz.plot_waveform_with_vad(wav_path, top_db=top_db, save_path=save_path)
            return save_path
        except Exception as e:
            print(f"[WARN] waveform/VAD failed for {wav_path}: {e}")
            return None
    
    def safe_plot_mfcc39(self, wav_path: str, save_prefix: Optional[str] = None):
        try:
            if save_prefix is None:
                base = os.path.splitext(os.path.basename(wav_path))[0]
                save_prefix = os.path.join(self.images_dir, f"{base}_mfcc39")
            self.viz.plot_mfcc39_with_segments(wav_path, save_prefix=save_prefix)
            return save_prefix
        except Exception as e:
            print(f"[WARN] mfcc39 failed for {wav_path}: {e}")
            return None
    
    def safe_plot_dtw_alignment(self, wav_path: str, vowel: str, person: str, 
                               save_path: Optional[str] = None, title_prefix: str = "DTW"):
        try:
            if save_path is None:
                base = os.path.splitext(os.path.basename(wav_path))[0]
                save_path = os.path.join(self.images_dir, f"{base}_dtw_alignment.png")
            
            if vowel in self.recognizer.templates and len(self.recognizer.templates[vowel]) > 0:
                first_pid, templ_list = next(iter(self.recognizer.templates[vowel].items()))
                if len(templ_list) > 0:
                    template_feats = templ_list[0]
                    test_feats = self.recognizer.extract_mfcc_39(wav_path)
                    self.viz.plot_dtw_alignment(
                        template_feats, test_feats,
                        title=f"{title_prefix} {vowel.upper()} | template:{first_pid} vs test:{person}",
                        save_path=save_path
                    )
                    return save_path
                else:
                    print(f"[INFO] No template list for vowel '{vowel}' -> skip DTW.")
            else:
                print(f"[INFO] No individual templates for vowel '{vowel}' -> skip DTW.")
            return None
        except Exception as e:
            print(f"[WARN] dtw alignment failed for {wav_path}: {e}")
            return None
    
    def safe_plot_vowel_bars(self, wav_path: str, true_vowel: str, 
                            save_path: Optional[str] = None, suffix_title: str = ""):
        try:
            if save_path is None:
                base = os.path.splitext(os.path.basename(wav_path))[0]
                save_path = os.path.join(self.images_dir, f"{base}_vowel_distances.png")
            
            pred, _, _, final_d = self.recognizer.recognize(wav_path)
            self.viz.plot_vowel_distances_bar(
                final_d,
                true_vowel=true_vowel,
                title=f"Distances for {os.path.basename(wav_path)}{suffix_title} (true={true_vowel}, pred={pred})",
                save_path=save_path
            )
            return save_path
        except Exception as e:
            print(f"[WARN] bar distances failed for {wav_path}: {e}")
            return None
    
    def safe_plot_all_vowel_distances(self, wav_path: str, save_path: Optional[str] = None):
        try:
            if save_path is None:
                base = os.path.splitext(os.path.basename(wav_path))[0]
                save_path = os.path.join(self.images_dir, f"{base}_all_vowel_distances.png")
            
            self.viz.plot_all_vowel_distances(wav_path, save_path=save_path)
            return save_path
        except Exception as e:
            print(f"[WARN] all vowel distances plot failed for {wav_path}: {e}")
            return None
    
    def safe_plot_generalized_templates(self, vowel: str, save_path: Optional[str] = None):
        try:
            if save_path is None:
                save_path = os.path.join(self.images_dir, f"generalized_template_{vowel}.png")
            
            if vowel in self.recognizer.generalized_templates:
                self.viz.plot_generalized_templates(vowel, save_path=save_path)
                return save_path
            else:
                print(f"[INFO] No generalized template for vowel '{vowel}'")
                return None
        except Exception as e:
            print(f"[WARN] generalized template plot failed for {vowel}: {e}")
            return None
    
    def safe_plot_segment_distances(self, wav_path: str, vowel: str, save_path: Optional[str] = None):
        try:
            if save_path is None:
                base = os.path.splitext(os.path.basename(wav_path))[0]
                save_path = os.path.join(self.images_dir, f"{base}_segment_distances_{vowel}.png")
            
            self.viz.plot_segment_distances(wav_path, vowel, save_path=save_path)
            return save_path
        except Exception as e:
            print(f"[WARN] segment distances plot failed for {wav_path}: {e}")
            return None
    
    def safe_plot_template_distribution(self, save_path: Optional[str] = None):
        try:
            if save_path is None:
                save_path = os.path.join(self.images_dir, "template_distribution.png")
            
            self.viz.plot_template_distribution(save_path=save_path)
            return save_path
        except Exception as e:
            print(f"[WARN] template distribution plot failed: {e}")
            return None
    
    def safe_plot_confusion_heatmap(self, results: List[dict], title: str = "Confusion Matrix", 
                                   save_path: Optional[str] = None):
        try:
            if save_path is None:
                save_path = os.path.join(self.images_dir, f"confusion_matrix_{title.lower().replace(' ', '_')}.png")
            
            self.viz.plot_confusion_heatmap(
                results,
                vowels=self.recognizer.vowels,
                title=title,
                save_path=save_path
            )
            return save_path
        except Exception as e:
            print(f"[WARN] confusion matrix plot failed: {e}")
            return None


def visualize_sample_analysis(visualizer: SafeVisualizer,
                             sample_path: str,
                             true_vowel: str,
                             sample_person: str,
                             scenario: str = ""):
    base = os.path.splitext(os.path.basename(sample_path))[0]
    scenario_suffix = f"_{scenario}" if scenario else ""
    
    print(f"\n--- Visualizing sample: {base} ({scenario}) ---")
    
    plots = {}
    
    # Waveform + VAD
    plots['waveform'] = visualizer.safe_plot_waveform_vad(
        sample_path, 
        save_path=os.path.join(visualizer.images_dir, f"{base}{scenario_suffix}_waveform_vad.png")
    )
    
    # MFCC 39 with segments
    plots['mfcc'] = visualizer.safe_plot_mfcc39(
        sample_path, 
        save_prefix=os.path.join(visualizer.images_dir, f"{base}{scenario_suffix}_mfcc39")
    )
    
    # DTW alignment
    plots['dtw'] = visualizer.safe_plot_dtw_alignment(
        sample_path, true_vowel, sample_person,
        save_path=os.path.join(visualizer.images_dir, f"{base}{scenario_suffix}_dtw_alignment.png")
    )
    
    # Vowel distance bars
    plots['distances'] = visualizer.safe_plot_vowel_bars(
        sample_path, true_vowel,
        save_path=os.path.join(visualizer.images_dir, f"{base}{scenario_suffix}_vowel_distances.png"),
        suffix_title=f" [{scenario.upper()}]" if scenario else ""
    )
    
    # All vowel distances
    plots['all_distances'] = visualizer.safe_plot_all_vowel_distances(
        sample_path,
        save_path=os.path.join(visualizer.images_dir, f"{base}{scenario_suffix}_all_vowel_distances.png")
    )
    
    # Segment distances for the true vowel
    plots['segments'] = visualizer.safe_plot_segment_distances(
        sample_path, true_vowel,
        save_path=os.path.join(visualizer.images_dir, f"{base}{scenario_suffix}_segment_distances.png")
    )
    
    return plots


def visualize_multiple_samples(visualizer: SafeVisualizer,
                              test_data: List[Tuple[str, str, str]],
                              max_samples: int = 3,
                              scenario: str = ""):
    results = []
    
    for i, (sample_path, true_vowel, person) in enumerate(test_data[:max_samples]):
        print(f"\n=== Sample {i+1}/{min(max_samples, len(test_data))} ===")
        plots = visualize_sample_analysis(visualizer, sample_path, true_vowel, person, scenario)
        results.append({
            'sample_info': (sample_path, true_vowel, person),
            'plots': plots
        })
    
    return results


def visualize_generalized_templates(visualizer: SafeVisualizer, vowels: List[str]):
    print("\n=== VISUALIZING GENERALIZED TEMPLATES ===")
    
    template_plots = {}
    for vowel in vowels:
        print(f"Generating template plot for vowel: {vowel}")
        template_plots[vowel] = visualizer.safe_plot_generalized_templates(vowel)
    
    return template_plots


def visualize_system_overview(visualizer: SafeVisualizer):
    print("\n=== GENERATING SYSTEM OVERVIEW ===")
    
    overview_plots = {}
    
    # Template distribution
    overview_plots['distribution'] = visualizer.safe_plot_template_distribution()
    
    return overview_plots


def visualize_evaluation_results(visualizer: SafeVisualizer, results_summary: dict):
    print("\n=== VISUALIZING EVALUATION RESULTS ===")
    
    eval_plots = {}
    
    # Confusion matrix for closed scenario
    all_closed_results = []
    for person_results in results_summary.values():
        if isinstance(person_results, dict) and 'closed_results' in person_results:
            all_closed_results.extend(person_results['closed_results'])
    
    if all_closed_results:
        eval_plots['closed_confusion'] = visualizer.safe_plot_confusion_heatmap(
            all_closed_results,
            title="Confusion Matrix (Closed Scenario)",
            save_path=os.path.join(visualizer.images_dir, "confusion_matrix_closed.png")
        )
    
    # Confusion matrix for open scenario
    all_open_results = []
    for person_results in results_summary.values():
        if isinstance(person_results, dict) and 'open_results' in person_results:
            all_open_results.extend(person_results['open_results'])
    
    if all_open_results:
        eval_plots['open_confusion'] = visualizer.safe_plot_confusion_heatmap(
            all_open_results,
            title="Confusion Matrix (Open Scenario)",
            save_path=os.path.join(visualizer.images_dir, "confusion_matrix_open.png")
        )
    
    return eval_plots


def create_comprehensive_visualizations(recognizer,
                                      test_data_templates: List[Tuple[str, str, str]],
                                      test_data_others: List[Tuple[str, str, str]],
                                      results_summary: dict,
                                      images_dir: str = "results/images",
                                      max_samples_per_scenario: int = 3):
    visualizer = SafeVisualizer(recognizer, images_dir)
    
    visualization_summary = {}
    
    # System overview
    visualization_summary['overview'] = visualize_system_overview(visualizer)
    
    # Generalized templates
    visualization_summary['templates'] = visualize_generalized_templates(
        visualizer, recognizer.vowels
    )
    
    # Sample analyses
    if test_data_templates:
        visualization_summary['closed_samples'] = visualize_multiple_samples(
            visualizer, test_data_templates, max_samples_per_scenario, "closed"
        )
    
    if test_data_others:
        visualization_summary['open_samples'] = visualize_multiple_samples(
            visualizer, test_data_others, max_samples_per_scenario, "open"
        )
    
    # Evaluation results
    visualization_summary['evaluation'] = visualize_evaluation_results(
        visualizer, results_summary
    )
    
    print(f"\n=== VISUALIZATION COMPLETE ===")
    print(f"All plots saved to: {images_dir}")
    
    return visualization_summary