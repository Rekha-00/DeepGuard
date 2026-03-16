<<<<<<< HEAD
# 🎙️ DeepGuard — Deepfake Audio Detector

> Final Year CS Project | AI/ML | Audio Deep Learning
Detects AI-generated (deepfake) voice clones using spectral feature analysis and machine learning.
> [Try DeepGuard Live](https://huggingface.co/spaces/Rekha-23/deepfake-audio-detector)

---

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your Dataset
```
data/
  real/   ← Put genuine human voice .wav files here
  fake/   ← Put AI-generated voice .wav files here
```
**Recommended Dataset:** ASVspoof 2021  
Download: https://www.asvspoof.org/index2021.html

**Alternative datasets:**
- FakeAVCeleb: https://github.com/DASH-Lab/FakeAVCeleb
- WaveFake: https://github.com/RUB-SysSec/WaveFake

### 3. Train the Model
```bash
python model.py
```
This will:
- Extract features from all audio files
- Train a Random Forest classifier
- Save model to `models/rf_model.pkl`
- Generate confusion matrix + ROC curve

### 4. Launch Web App
```bash
python app.py
```
Open `http://localhost:5000` in your browser.

---

## 📁 Project Structure

```
deepfake_audio_detector/
│
├── model.py          # Feature extraction + model training
├── inference.py      # Prediction engine
├── app.py            # Flask web server
├── requirements.txt  # Python dependencies
│
├── data/
│   ├── real/         # Genuine audio files (.wav)
│   └── fake/         # AI-generated audio files (.wav)
│
├── models/
│   ├── rf_model.pkl  # Trained Random Forest model
│   └── scaler.pkl    # Feature scaler
│
├── static/
│   └── index.html    # Frontend dashboard
│
└── outputs/          # Evaluation plots (auto-created)
```

---

## 🧠 How It Works

### Feature Extraction Pipeline
Each audio file is converted into a rich feature vector:

| Feature | Description | Dim |
|---------|-------------|-----|
| MFCCs (mean + std) | Timbral texture | 80 |
| Delta MFCCs | Temporal dynamics | 40 |
| Mel-Spectrogram (mean + std) | Frequency profile | 256 |
| Chroma | Harmonic content | 12 |
| Spectral Contrast | Peak vs valley energy | 7 |
| ZCR + RMS | Energy characteristics | 4 |

**Total: ~400-dimensional feature vector**

### Models Compared
1. **Random Forest** (baseline) — Fast, interpretable
2. **CNN on Mel-Spectrogram** — Visual pattern detection  
3. **Wav2Vec2** — State-of-art pretrained audio transformer

### Why Deepfakes Are Detectable
AI-generated voices have subtle artifacts in:
- **Unnatural harmonics** — Spectral irregularities
- **Missing breathiness** — Human breath is hard to synthesize
- **Temporal inconsistencies** — Unnatural pausing/rhythm
- **Frequency artifacts** — Vocoders leave spectral fingerprints

---

## 📊 Expected Performance (ASVspoof 2021)

| Model | Accuracy | AUC |
|-------|----------|-----|
| Random Forest | ~85–91% | ~0.90 |
| CNN | ~88–94% | ~0.93 |
| Wav2Vec2 | ~92–97% | ~0.97 |

---

## 🔬 Novel Contributions

1. **Multi-feature fusion** — Combines 6 different audio features
2. **Explainable output** — Spectrogram visualization shows WHY
3. **Modern AI voice coverage** — Tested on ElevenLabs, Suno outputs
4. **Real-time ready** — Processes 4-second clips in <2 seconds

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Audio Processing | `librosa`, `soundfile` |
| Machine Learning | `scikit-learn`, `PyTorch` |
| Deep Learning | `transformers` (Wav2Vec2) |
| Web Backend | `Flask` |
| Visualization | `matplotlib` |
| Frontend | HTML/CSS/JS |

---

## 📹 Demo Flow (for YouTube video)

1. Show the web interface
2. Upload a real human voice → show REAL result
3. Upload an ElevenLabs-generated voice → show FAKE result
4. Show the spectrogram visualization
5. Explain the probability bars

---

## 📝 Report Sections

1. Abstract
2. Introduction & Motivation
3. Literature Review (ASVspoof, Wav2Vec2 papers)
4. System Architecture
5. Feature Engineering
6. Model Training & Results
7. Web Application
8. Conclusion & Future Work

---

## 📚 References

1. Todisco et al. "ASVspoof 2019: A large-scale public database..." (2019)
2. Baevski et al. "wav2vec 2.0: A Framework for Self-Supervised Learning" (2020)
3. Yi et al. "Half-truth: A partially fake audio detection dataset" (2022)

---

*Built for Final Year CS Project — AI/ML Track*
=======
---
title: DeepGuard
emoji: 👀
colorFrom: indigo
colorTo: pink
sdk: docker
pinned: false
license: mit
short_description: AI-Powered Deepfake Audio Detection System
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 930444b1eb1d5ac3408ceac6332bbff264da300b
