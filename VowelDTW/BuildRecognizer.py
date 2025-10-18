import os
from typing import List, Dict, Tuple, Optional
from VowelDTW.VowelRecognitionDTW import VowelRecognitionDTW


class RecognizerBuilder:    
    def __init__(self, 
                 sample_rate: int = 16000,
                 use_vad: bool = True,
                 normalize: bool = True,
                 n_segments: int = 3,
                 max_templates_per_speaker: Optional[int] = None):
        
        self.recognizer = VowelRecognitionDTW(
            sample_rate=sample_rate,
            use_vad=use_vad,
            normalize=normalize,
            n_segments=n_segments
        )
        self.max_templates_per_speaker = max_templates_per_speaker
        self.vowels = ['a', 'i', 'u', 'e', 'o']
        self.persons = []
        self.template_files = {}  
        self.test_files = {}      
        
    def scan_template_folders(self, base_path: str, persons: List[str], recursive: bool = False):
        self.persons = persons
        self.template_files = {}
        
        print(f"=== SCANNING TEMPLATE FILES ({base_path}) ===")
        for person in persons:
            person_folder = os.path.join(base_path, person)
            if recursive:
                self.template_files[person] = scan_person_folder_recursive(person_folder, self.vowels)
            else:
                self.template_files[person] = scan_person_folder(person_folder, self.vowels)
            
            total_files = sum(len(files) for files in self.template_files[person].values())
            print(f"  {person}: {total_files} files found")
            for vowel, files in self.template_files[person].items():
                print(f"    {vowel}: {len(files)} files")
        
        return self.template_files
    
    def scan_test_folders(self, base_path: str, persons: Optional[List[str]] = None, recursive: bool = False):
        if persons is None:
            if os.path.exists(base_path):
                persons = [name for name in os.listdir(base_path) 
                          if os.path.isdir(os.path.join(base_path, name))]
            else:
                print(f"Test folder not found: {base_path}")
                return {}
        
        self.test_files = {}
        print(f"=== SCANNING TEST FILES ({base_path}) ===")
        
        for person in persons:
            person_folder = os.path.join(base_path, person)
            if recursive:
                self.test_files[person] = scan_person_folder_recursive(person_folder, self.vowels)
            else:
                self.test_files[person] = scan_person_folder(person_folder, self.vowels)
            
            total_files = sum(len(files) for files in self.test_files[person].values())
            if total_files > 0:
                print(f"  {person}: {total_files} files found")
                for vowel, files in self.test_files[person].items():
                    if files:
                        print(f"    {vowel}: {len(files)} files")
        
        return self.test_files
    
    def load_templates(self, exclude_last_for_test: bool = True):
        print("\n=== LOADING TEMPLATES ===")
        
        templates_added = 0
        per_vowel_counts = {v: 0 for v in self.vowels}
        
        add_templates_from_files(
            self.recognizer,
            self.persons,
            self.template_files,
            self.max_templates_per_speaker,
            exclude_last=exclude_last_for_test
        )
        
        for vowel, person_dict in self.recognizer.templates.items():
            for person, template_list in person_dict.items():
                count = len(template_list)
                per_vowel_counts[vowel] += count
                templates_added += count
        
        print(f"Templates loaded: {templates_added} total")
        for vowel, count in per_vowel_counts.items():
            print(f"  {vowel}: {count} templates")
        
        return templates_added
    
    def build_generalized_templates(self):
        print("\n=== BUILDING GENERALIZED TEMPLATES ===")
        self.recognizer.build_generalized_templates()
        
        for vowel in self.vowels:
            if vowel in self.recognizer.generalized_templates:
                n_segments = len(self.recognizer.generalized_templates[vowel]['means'])
                print(f"  {vowel}: {n_segments} segments")
            else:
                print(f"  {vowel}: No templates")
    
    def build_test_datasets(self):
        test_data_templates = build_test_data_from_files(
            self.persons, 
            self.template_files, 
            self.vowels, 
            use_last_file=True
        )
        
        test_persons_other = list(self.test_files.keys()) if self.test_files else []
        test_data_others = build_test_data_from_files(
            test_persons_other,
            self.test_files,
            self.vowels,
            use_last_file=True
        )
        
        print(f"\n=== TEST DATASETS ===")
        print(f"Test data (from templates): {len(test_data_templates)} files")
        print(f"Test data (from others): {len(test_data_others)} files")
        
        return test_data_templates, test_data_others
    
    def build_complete_recognizer(self, 
                                 template_base_path: str,
                                 template_persons: List[str],
                                 test_base_path: Optional[str] = None,
                                 recursive_scan: bool = False):
        self.scan_template_folders(template_base_path, template_persons, recursive_scan)
        
        if test_base_path and os.path.exists(test_base_path):
            self.scan_test_folders(test_base_path, persons=None, recursive=recursive_scan)
        
        self.load_templates(exclude_last_for_test=True)
        
        self.build_generalized_templates()
        
        test_data_templates, test_data_others = self.build_test_datasets()
        
        return self.recognizer, test_data_templates, test_data_others
    
    def print_summary(self):
        print("\n=== RECOGNIZER SUMMARY ===")
        
        if hasattr(self.recognizer, 'templates'):
            total_templates = 0
            print("Templates by vowel and person:")
            for vowel in self.vowels:
                if vowel in self.recognizer.templates:
                    print(f"  {vowel}:")
                    for person, template_list in self.recognizer.templates[vowel].items():
                        count = len(template_list)
                        total_templates += count
                        print(f"    {person}: {count} templates")
                else:
                    print(f"  {vowel}: No templates")
            print(f"Total templates: {total_templates}")
        
        if hasattr(self.recognizer, 'generalized_templates'):
            print("\nGeneralized templates:")
            for vowel in self.vowels:
                if vowel in self.recognizer.generalized_templates:
                    n_segments = len(self.recognizer.generalized_templates[vowel]['means'])
                    print(f"  {vowel}: {n_segments} segments")
                else:
                    print(f"  {vowel}: Not available")
        
        template_test_count = sum(len(files) for files in self.template_files.values() 
                                 for files in files.values())
        other_test_count = sum(len(files) for files in self.test_files.values() 
                              for files in files.values()) if self.test_files else 0
        
        print(f"\nTest data available:")
        print(f"  From template folders: {template_test_count} files")
        print(f"  From separate test folders: {other_test_count} files")


def build_recognizer_from_folders(template_base_path: str,
                                 template_persons: List[str],
                                 test_base_path: Optional[str] = None,
                                 **recognizer_kwargs):
    builder = RecognizerBuilder(**recognizer_kwargs)
    recognizer, test_templates, test_others = builder.build_complete_recognizer(
        template_base_path, template_persons, test_base_path
    )
    builder.print_summary()
    return recognizer, test_templates, test_others


def build_recognizer_from_files(template_files: List[str],
                               vowel_labels: List[str],
                               person_labels: List[str],
                               **recognizer_kwargs):
    recognizer = VowelRecognitionDTW(**recognizer_kwargs)
    
    for file_path, vowel, person in zip(template_files, vowel_labels, person_labels):
        try:
            recognizer.add_template(vowel, person, file_path)
        except Exception as e:
            print(f"[WARN] Error loading template {file_path}: {e}")
    
    recognizer.build_generalized_templates()
    return recognizer