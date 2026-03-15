"""
Deepfake Audio Detector - Inference Engine
============================================
Load trained model and predict on a single audio file.

Usage:
    from inference import predict_audio
    result = predict_audio("path/to/audio.wav")
    print(result)
"""

import os
import numpy as np
import librosa
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────
SAMPLE_RATE  = 16000
DURATION     = 4
N_MFCC       = 40
N_MELS       = 128
HOP_LENGTH   = 512
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "rf_model.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "models", "scaler.pkl")


# ─── Feature Extraction (must match model.py) ─────
def extract_features(file_path: str) -> np.ndarray:
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        target_len = SAMPLE_RATE * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        features = []

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        delta_mfcc = librosa.feature.delta(mfcc)
        features.extend(np.mean(delta_mfcc, axis=1))

        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features.extend(np.mean(mel_db, axis=1))
        features.extend(np.std(mel_db, axis=1))

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))

        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend(np.mean(contrast, axis=1))

        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.std(rms))

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(np.mean(rolloff))

        return np.array(features)
    except Exception as e:
        raise ValueError(f"Feature extraction failed: {e}")


# ─── Spectrogram Generator ────────────────────────
def generate_spectrogram(file_path: str, save_path: str) -> str:
    """Generate mel-spectrogram image and save it."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.patch.set_facecolor('#0d0d1a')

    # Mel Spectrogram
    axes[0].set_facecolor('#0d0d1a')
    im1 = axes[0].imshow(mel_db, aspect='auto', origin='lower',
                         cmap='magma', interpolation='bilinear')
    axes[0].set_title('Mel-Spectrogram', color='white', fontsize=13, pad=10)
    axes[0].set_xlabel('Time Frames', color='#aaaaaa', fontsize=10)
    axes[0].set_ylabel('Mel Frequency Bins', color='#aaaaaa', fontsize=10)
    axes[0].tick_params(colors='#888888')
    plt.colorbar(im1, ax=axes[0], format='%+2.0f dB').ax.yaxis.set_tick_params(color='white')

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    axes[1].set_facecolor('#0d0d1a')
    im2 = axes[1].imshow(mfcc, aspect='auto', origin='lower',
                         cmap='coolwarm', interpolation='bilinear')
    axes[1].set_title('MFCC Features', color='white', fontsize=13, pad=10)
    axes[1].set_xlabel('Time Frames', color='#aaaaaa', fontsize=10)
    axes[1].set_ylabel('MFCC Coefficients', color='#aaaaaa', fontsize=10)
    axes[1].tick_params(colors='#888888')
    plt.colorbar(im2, ax=axes[1]).ax.yaxis.set_tick_params(color='white')

    plt.tight_layout(pad=2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0d0d1a')
    plt.close()
    return save_path


# ─── Main Prediction Function ─────────────────────
def predict_audio(file_path: str) -> dict:
    """
    Predict whether an audio file is real or fake.

    Returns:
        {
          "label":       "FAKE" | "REAL",
          "confidence":  float (0–100),
          "fake_prob":   float (0–1),
          "real_prob":   float (0–1),
          "risk_level":  "HIGH" | "MEDIUM" | "LOW",
          "verdict":     str
        }
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            "Model not found. Please run `python model.py` first to train the model."
        )

    # Load model
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Extract features
    features = extract_features(file_path)
    features_scaled = scaler.transform([features])

    # Predict
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    fake_prob = float(probabilities[1])
    real_prob = float(probabilities[0])
    label     = "FAKE" if prediction == 1 else "REAL"
    confidence = (fake_prob if label == "FAKE" else real_prob) * 100

    # Risk level
    if fake_prob >= 0.75:
        risk_level = "HIGH"
    elif fake_prob >= 0.45:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # Human-readable verdict
    if label == "FAKE":
        verdict = (
            f"⚠️ This audio is likely AI-generated with {confidence:.1f}% confidence. "
            f"Patterns consistent with voice cloning or synthesis were detected."
        )
    else:
        verdict = (
            f"✅ This audio appears to be genuine human speech with {confidence:.1f}% confidence. "
            f"No significant deepfake patterns were detected."
        )

    return {
        "label":      label,
        "confidence": round(confidence, 2),
        "fake_prob":  round(fake_prob, 4),
        "real_prob":  round(real_prob, 4),
        "risk_level": risk_level,
        "verdict":    verdict,
        "file":       os.path.basename(file_path)
    }


# ─── CLI Usage ────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <audio_file.wav>")
        sys.exit(1)

    audio_file = sys.argv[1]
    print(f"\n🎙️  Analyzing: {audio_file}")
    print("-" * 45)

    result = predict_audio(audio_file)

    print(f"  Label      : {result['label']}")
    print(f"  Confidence : {result['confidence']}%")
    print(f"  Risk Level : {result['risk_level']}")
    print(f"  Fake Prob  : {result['fake_prob']}")
    print(f"  Real Prob  : {result['real_prob']}")
    print(f"\n  {result['verdict']}")
