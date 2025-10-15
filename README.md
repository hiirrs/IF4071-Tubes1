# Vowel Recognition using Dynamic Time Warping (DTW)

Sistem pengenalan vokal Bahasa Indonesia (a, i, u, e, o) menggunakan metode Dynamic Time Warping (DTW) dan fitur MFCC 39 dimensi.

## 📋 Deskripsi

Proyek ini mengimplementasikan sistem pengenalan vokal otomatis yang dapat:
- Mengenali 5 vokal dasar Bahasa Indonesia: **a, i, u, e, o**
- Menggunakan DTW untuk mencocokkan pola audio dengan template
- Ekstraksi fitur MFCC 39 dimensi (13 MFCC + 13 Delta + 13 Delta-Delta)
- Preprocessing audio dengan Voice Activity Detection (VAD) dan pre-emphasis
- Evaluasi dengan dua skenario: **Closed** dan **Open**

## 🗂️ Struktur Proyek

```
IF4071-TUBES1/
├── templates_us/          # Dataset training dari 5 orang (densu, hira, naufal, wiga, zya)
│   ├── densu/
│   │   ├── densu - a 1.wav
│   │   ├── densu - a 2.wav
│   │   ├── densu - a 3.wav
│   │   ├── densu - e 1.wav
│   │   └── ...
│   ├── hira/
│   ├── naufal/
│   ├── wiga/
│   └── zya/
├── templates_other/       # Dataset testing dari orang lain (Akbar, Evelyn, Fed, Justin, Ucup)
│   ├── Akbar/
│   ├── Evelyn/
│   ├── Fed/
│   ├── Justin/
│   └── Ucup/
├── VowelDTW/
│   ├── __init__.py
│   ├── VowelRecognitionDTW.py      # Core recognition engine
│   └── VowelDTWVisualizer.py       # Visualization tools
├── results/
│   ├── images/            # Generated visualization images
│   └── results.json       # Evaluation results
├── main.py                # Main execution script
└── README.md
```

## 🚀 Instalasi

### Prerequisites

- Python 3.7+
- pip

### Install Dependencies

```bash
pip install numpy scipy librosa matplotlib python_speech_features fastdtw
```

Atau gunakan requirements.txt:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.19.0
scipy>=1.5.0
librosa>=0.8.0
matplotlib>=3.3.0
python_speech_features>=0.6
fastdtw>=0.3.4
```

## 📊 Dataset

### Format File Audio
- **Format**: `.wav` atau `.m4a`
- **Naming Convention**: `{person} - {vowel} {number}.wav`
  - Contoh: `densu - a 1.wav`, `hira - i 2.wav`

### Pembagian Data

**templates_us** (Training & Testing Closed):
- 5 orang: densu, hira, naufal, wiga, zya
- Setiap orang merekam 3 file per vokal
- File 1-2: Template training
- File 3: Testing untuk closed scenario

**templates_other** (Testing Open):
- 5 orang berbeda: Akbar, Evelyn, Fed, Justin, Ucup
- File terakhir dari setiap vokal digunakan untuk testing open scenario

## 🔧 Cara Menggunakan

### Running the Main Program

```bash
python main.py
```

Program akan:
1. Scan semua file audio dari folder `templates_us` dan `templates_other`
2. Load template dari file dengan nomor terkecil (1-2)
3. Menggunakan file dengan nomor terbesar (3) untuk testing
4. Generate visualisasi (waveform, MFCC, DTW alignment, confusion matrix)
5. Evaluasi dengan closed dan open scenario
6. Simpan hasil ke `results/results.json` dan gambar ke `results/images/`

### Custom Usage

```python
from VowelDTW.VowelRecognitionDTW import VowelRecognitionDTW
from VowelDTW.VowelDTWVisualizer import VowelDTWVisualizer

# Initialize recognizer
recognizer = VowelRecognitionDTW(
    sample_rate=16000,
    use_vad=True,
    normalize=True
)

# Add templates
recognizer.add_template('a', 'person1', 'path/to/audio_a.wav')
recognizer.add_template('i', 'person1', 'path/to/audio_i.wav')

# Recognize vowel
vowel, distance, all_distances, final_distances = recognizer.recognize('test_audio.wav')
print(f"Recognized: {vowel} with distance: {distance}")

# Visualization
viz = VowelDTWVisualizer(recognizer)
viz.plot_waveform_with_vad('test_audio.wav', save_path='waveform.png')
viz.plot_mfcc39('test_audio.wav', save_prefix='mfcc')
```

## 📈 Fitur Ekstraksi

### MFCC 39 Dimensi

1. **13 MFCC Coefficients**
   - Mel-Frequency Cepstral Coefficients
   - Parameter: numcep=13, nfilt=26, nfft=512
   
2. **13 Delta (Δ)**
   - Turunan pertama dari MFCC
   - Menangkap perubahan temporal
   
3. **13 Delta-Delta (ΔΔ)**
   - Turunan kedua dari MFCC
   - Menangkap akselerasi perubahan

### Preprocessing

- **Pre-emphasis Filter**: Meningkatkan frekuensi tinggi
- **Voice Activity Detection (VAD)**: Membuang silence (top_db=20)
- **Amplitude Normalization**: Normalisasi amplitudo audio
- **Median Filtering**: Mengurangi noise pada MFCC
- **CMVN**: Cepstral Mean and Variance Normalization

## 🎯 Evaluasi

### Skenario Testing

**1. Closed Scenario**
- Template: Semua orang dari `templates_us` (file 1-2)
- Test: File 3 dari `templates_us`
- Mengukur akurasi untuk speaker yang sudah dikenal

**2. Open Scenario**
- Template: Satu orang tertentu dari `templates_us`
- Test: Semua orang dari `templates_other`
- Mengukur generalisasi untuk speaker baru

### Metrics

- **Closed Accuracy**: Akurasi pada closed scenario
- **Open Accuracy**: Akurasi pada open scenario
- **Average Accuracy**: Rata-rata dari kedua scenario
- **Confusion Matrix**: Analisis kesalahan prediksi

## 📊 Visualisasi

Program menghasilkan visualisasi berikut di folder `results/images/`:

1. **Waveform + VAD**: Sinyal audio dengan interval voice activity
2. **MFCC Features**: Heatmap untuk MFCC, Delta, dan Delta-Delta
3. **DTW Alignment**: Cost matrix dan warping path
4. **Vowel Distances Bar**: Jarak DTW ke setiap vokal
5. **Confusion Matrix**: Heatmap confusion matrix

## 🔍 Algoritma DTW

Dynamic Time Warping digunakan untuk mencocokkan dua sequence dengan panjang berbeda:

```
distance, path = fastdtw(template, test, dist=euclidean)
normalized_distance = distance / len(path)
```

**Normalisasi**: Jarak dibagi panjang path untuk fairness antar sequence berbeda panjang

## 📝 Output

### Console Output
- Detail prediksi setiap file test
- Jarak ke semua vokal
- Confusion matrix
- Summary akurasi per person dan overall

### results.json
```json
{
  "densu": {
    "closed_accuracy": 95.50,
    "open_accuracy": 88.30,
    "average_accuracy": 91.90
  },
  "overall": {
    "closed_accuracy": 93.20,
    "open_accuracy": 85.60,
    "average_accuracy": 89.40
  }
}
```

## 🛠️ Troubleshooting

**Audio tidak terdeteksi:**
- Pastikan format file `.wav` atau `.m4a`
- Cek naming convention: `{person} - {vowel} {number}.wav`

**Error saat load audio:**
- Install ulang librosa: `pip install --upgrade librosa`
- Pastikan ffmpeg terinstall untuk format non-wav

**Akurasi rendah:**
- Tingkatkan jumlah template per vokal
- Adjust parameter VAD (top_db)
- Coba dengan/tanpa normalisasi

## 👥 Contributors

- 13522013 - Denise Felicia Tiowanni
- 13522053 - Erdianti Wiga Putri Andini
- 13522063 - Shazya Audrea Taufik
- 13522074 - Muhammad Naufal Aulia
- 13522085 - Zahira Dina Amalia

## 📄 License

This project is for educational purposes (IF4071 - Pattern Recognition).

## 📚 References

- Rabiner, L., & Juang, B. H. (1993). Fundamentals of speech recognition.
- Müller, M. (2007). Information retrieval for music and motion.
- Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition.

---

**Note**: Proyek ini dibuat untuk tugas mata kuliah IF4071 Pattern Recognition, Institut Teknologi Bandung.
