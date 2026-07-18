import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
import io
import os
import cv2
import soundfile as sf
from huggingface_hub import hf_hub_download

# Safe import flags for Hugging Face components
try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Multimodal Age & Gender Predictor",
    page_icon="🧬",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS  (must match training notebooks exactly)
# ─────────────────────────────────────────────────────────────────────────────

class ImprovedFaceModel(nn.Module):
    """
    Multi-task CNN for age regression + gender classification from face images.
    Input: 3 × 128 × 128 RGB image (ImageNet-normalised)
    Outputs: (age_scalar, gender_logits[2])
    """
    def __init__(self, dropout_rate=0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 — 128×128 → 64×64
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            # Block 2 — 64×64 → 32×32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.3),
            # Block 3 — 32×32 → 16×16
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.4),
            # Block 4 — 16×16 → 8×8
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.4),
        )

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
        )

        self.age_branch = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(256, 128), nn.ReLU(inplace=True),     
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 1),
        )

        self.gender_branch = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        f = self.features(x)
        s = self.shared(f)
        return self.age_branch(s).squeeze(), self.gender_branch(s)


class VoiceAgeEstimator(nn.Module):
    """
    CNN for age regression from mel-spectrograms of speech.
    Input: 1 × 128 × 94 mel-spectrogram (min-max normalised to [0,1])
    Output: age_scalar (non-negative via ReLU)
    """
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.3),
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(10240, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1), nn.ReLU(),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

FACE_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

SAMPLE_RATE  = 16_000
DURATION_SEC = 3.0
N_SAMPLES    = int(SAMPLE_RATE * DURATION_SEC)   # 48,000
N_MELS       = 128
N_FFT        = 2048
HOP_LENGTH   = 512
F_MAX        = 8_000

MEL_TRANSFORM = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_max=F_MAX,
)
AMPLITUDE_TO_DB = T.AmplitudeToDB()

YUNET_MODEL_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")

def detect_and_crop_face(pil_img: Image.Image) -> tuple[torch.Tensor, Image.Image, bool]:
    """
    Detects faces using OpenCV's YuNet framework, crops with margin, and prepares tensor.
    """
    img_bgr = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
    height, width, _ = img_bgr.shape
    
    annotated_img = pil_img.copy()
    draw = ImageDraw.Draw(annotated_img)
    face_found = False

    if not os.path.exists(YUNET_MODEL_PATH):
        cropped_tensor = FACE_TRANSFORM(pil_img.convert("RGB")).unsqueeze(0)
        return cropped_tensor, annotated_img, False

    try:
        detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL_PATH, "", (width, height), 0.6, 0.3, 5000, 0, 0
        )
        _, faces = detector.detect(img_bgr)
    except Exception:
        faces = None

    if faces is not None and len(faces) > 0:
        face_found = True
        x, y, w, h = map(int, faces[0][0:4])
        
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)
        
        crop_xmin = max(0, x - pad_x)
        crop_ymin = max(0, y - pad_y)
        crop_xmax = min(width, x + w + pad_x)
        crop_ymax = min(height, y + h + pad_y)
        
        face_crop = pil_img.crop((crop_xmin, crop_ymin, crop_xmax, crop_ymax))
        cropped_tensor = FACE_TRANSFORM(face_crop.convert("RGB")).unsqueeze(0)
        draw.rectangle([x, y, x + w, y + h], outline="#00FFCC", width=4)
    else:
        cropped_tensor = FACE_TRANSFORM(pil_img.convert("RGB")).unsqueeze(0)
        
    return cropped_tensor, annotated_img, face_found


def preprocess_audio(audio_bytes: bytes) -> torch.Tensor:
    buf = io.BytesIO(audio_bytes)
    data, sr = sf.read(buf, dtype='float32')

    if len(data.shape) == 1:
        waveform = torch.tensor(data).unsqueeze(0)
    else:
        waveform = torch.tensor(data).t()

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    if waveform.shape[1] < N_SAMPLES:
        pad = N_SAMPLES - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :N_SAMPLES]

    mel = MEL_TRANSFORM(waveform)
    mel = AMPLITUDE_TO_DB(mel)
    mel_min, mel_max = mel.min(), mel.max()
    if mel_max > mel_min:
        mel = (mel - mel_min) / (mel_max - mel_min)

    return mel.unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

HF_REPO_ID = "St0Lexy/Multimodal"
FACE_MODEL_FILENAME = "best_face_model.pth"
VOICE_MODEL_FILENAME = "best_voice_model.pth"
GENDER_MODEL_ID  = "dima806/man_woman_face_image_detection"

FACE_WEIGHT  = 0.6
VOICE_WEIGHT = 0.4

@st.cache_resource
def load_face_model():
    model = ImprovedFaceModel(dropout_rate=0.5)
    try:
        with st.spinner("Downloading face model weights from Hugging Face..."):
            resolved_face_model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=FACE_MODEL_FILENAME)
        state = torch.load(resolved_face_model_path, map_location=DEVICE)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        return model, True
    except Exception as e:
        st.error(f"Failed to load face model: {e}")
        return model.to(DEVICE).eval(), False

@st.cache_resource
def load_voice_model():
    model = VoiceAgeEstimator()
    try:
        with st.spinner("Downloading voice model weights from Hugging Face..."):
            resolved_path = hf_hub_download(repo_id=HF_REPO_ID, filename=VOICE_MODEL_FILENAME)
        state = torch.load(resolved_path, map_location=DEVICE)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        return model, True
    except Exception as e:
        st.error(f"Failed to load voice model: {e}")
        return model.to(DEVICE).eval(), False

@st.cache_resource
def load_gender_model():
    """Dynamically initializes the pre-trained transformers ViT pipeline."""
    if not TRANSFORMERS_AVAILABLE:
        st.error("The 'transformers' library is not available. Gender prediction will fallback to Unknown.")
        return None, False
    try:
        with st.spinner("Loading Vision Transformer gender classification pipeline..."):
            # Set pipeline device map based on torch hardware availability
            device_id = 0 if torch.cuda.is_available() else -1
            pipe = hf_pipeline("image-classification", model=GENDER_MODEL_ID, device=device_id)
        return pipe, True
    except Exception as e:
        st.error(f"Failed to initialize validation pipeline: {e}")
        return None, False


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def predict_face(model, tensor: torch.Tensor):
    """
    Accepts preprocessed image tensor and extracts custom scalar face age.
    """
    with torch.no_grad():
        tensor = tensor.to(DEVICE)
        age_raw, _ = model(tensor) # Safely handle and discard the built-in head's logit outputs
        age = float(age_raw.item())
    return age

def predict_gender(gender_classifier, pil_img: Image.Image):
    """
    Use the pre-trained ViT classifier for gender scoring.
    """
    if gender_classifier is None:
        return "Unknown", 0.0
    try:
        results = gender_classifier(pil_img.convert("RGB"))
        top = results[0]
        label_raw = top["label"].lower()
        confidence = top["score"] * 100
        label = "Male" if "man" in label_raw else "Female"
        return label, confidence
    except Exception:
        return "Unknown", 0.0

def predict_voice(model, tensor: torch.Tensor):
    with torch.no_grad():
        out = model(tensor.to(DEVICE))
        return float(out.squeeze().item())

def fuse_ages(face_age: float, voice_age: float) -> float:
    return FACE_WEIGHT * face_age + VOICE_WEIGHT * voice_age


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def age_bar(label: str, age: float, max_age: int = 100, color: str = "#4F8BF9"):
    pct = min(age / max_age, 1.0) * 100
    st.markdown(f"""
        <p style="margin-bottom:2px;font-size:13px;color:#888">{label}</p>
        <div style="background:#e9ecef;border-radius:6px;height:20px;width:100%">
          <div style="background:{color};border-radius:6px;height:20px;width:{pct:.1f}%;
                      display:flex;align-items:center;padding-left:8px">
            <span style="color:white;font-weight:700;font-size:13px">{age:.1f} yrs</span>
          </div>
        </div>
    """, unsafe_allow_html=True)


def result_card(title: str, value: str, sub: str = "", color: str = "#4F8BF9"):
    st.markdown(f"""
        <div style="border-left:5px solid {color};padding:12px 16px;
                    background:#f8f9fa;border-radius:4px;margin-bottom:12px">
          <p style="margin:0;font-size:12px;color:#888;text-transform:uppercase">{title}</p>
          <p style="margin:0;font-size:28px;font-weight:700;color:#212529">{value}</p>
          {"<p style='margin:0;font-size:12px;color:#6c757d'>"+sub+"</p>" if sub else ""}
        </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

st.title("🧬 Multimodal Age & Gender Predictor")
st.markdown(
    "Upload a **face image** and / or a **voice recording** to predict age and gender. "
    "When both are provided the system combines the two age estimates via weighted late fusion."
)

face_model, face_loaded   = load_face_model()
voice_model, voice_loaded = load_voice_model()
gender_model, gender_loaded = load_gender_model()

# Model status banner
with st.expander("ℹ️ Model status", expanded=False):
    col1, col2, col3 = st.columns(3)
    col1.metric("Face Age Model", "✅ Loaded" if face_loaded else "⚠️ Random Initialization")
    col2.metric("Voice Age Model", "✅ Loaded" if voice_loaded else "⚠️ Random Initialization")
    col3.metric("ViT Gender Model", "✅ Active" if gender_loaded else "⚠️ Inactive (Fallback)")
    st.caption(
        f"Running on: **{str(DEVICE).upper()}** | "
        f"Face weight (fusion): **{FACE_WEIGHT}** | "
        f"Voice weight (fusion): **{VOICE_WEIGHT}**"
    )

st.divider()

# ── Input columns ──────────────────────────────────────────────────────────
left, right = st.columns(2)

processed_face_tensor = None
face_detected = False

with left:
    st.subheader("📷 Face Image")
    face_file = st.file_uploader(
        "Upload a frontal face photo (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        key="face_upload",
    )
    if face_file:
        img = Image.open(face_file)
        with st.spinner("Analyzing image features..."):
            processed_face_tensor, annotated_image, face_detected = detect_and_crop_face(img)
        
        st.image(annotated_image, caption="Processed Image (Bounding Box Overlay)", use_container_width=True)
        if not face_detected:
            st.warning("⚠️ No face detected. Defaulting to full-image crop. Predictions might be inaccurate.")

with right:
    st.subheader("🎤 Voice Recording")
    voice_file = st.file_uploader(
        "Upload a speech clip (WAV / MP3 / OGG, ideally 3–10 s)",
        type=["wav", "mp3", "ogg", "flac", "m4a"],
        key="voice_upload",
    )
    if voice_file:
        st.audio(voice_file, format="audio/wav")

st.divider()

# ── Predict button ─────────────────────────────────────────────────────────
if st.button("🔍 Predict", type="primary", use_container_width=True):

    if not face_file and not voice_file:
        st.warning("Please upload at least one input (image or audio).")
        st.stop()

    face_age = voice_age = gender_label = gender_conf = None

    # ── Face & Gender inference ──
    if face_file and processed_face_tensor is not None:
        with st.spinner("Running face model…"):
            face_age = predict_face(face_model, processed_face_tensor)
            gender_label, gender_conf = predict_gender(gender_model, img)

    # ── Voice inference ──
    if voice_file:
        with st.spinner("Running voice model…"):
            audio_bytes = voice_file.read()
            try:
                voice_tensor = preprocess_audio(audio_bytes)
                voice_age = predict_voice(voice_model, voice_tensor)
            except Exception as e:
                st.error(f"Audio preprocessing failed: {e}")
                voice_age = None

    # ── Late fusion ──
    fused_age = None
    if face_age is not None and voice_age is not None:
        fused_age = fuse_ages(face_age, voice_age)

    # ── Results ────────────────────────────────────────────────────────────
    st.subheader("📊 Results")

    r1, r2, r3 = st.columns(3)

    with r1:
        if face_age is not None:
            result_card("Face — Predicted Age", f"{face_age:.1f} yrs",
                        "From face image", "#4F8BF9")
            result_card("Gender (Transformers ViT)",
                        f"{gender_label}",
                        f"Confidence: {gender_conf:.1f}%",
                        "#6f42c1" if gender_label == "Female" else "#0d6efd")
        else:
            st.info("No face image provided.")

    with r2:
        if voice_age is not None:
            result_card("Voice — Predicted Age", f"{voice_age:.1f} yrs",
                        "From speech recording", "#20c997")
        else:
            st.info("No voice recording provided.")

    with r3:
        if fused_age is not None:
            result_card("Fused Age (Late Fusion)",
                        f"{fused_age:.1f} yrs",
                        f"= {FACE_WEIGHT}×face + {VOICE_WEIGHT}×voice",
                        "#fd7e14")
        elif face_age is not None:
            result_card("Final Age", f"{face_age:.1f} yrs",
                        "Single modality (no voice provided)", "#fd7e14")
        elif voice_age is not None:
            result_card("Final Age", f"{voice_age:.1f} yrs",
                        "Single modality (no face provided)", "#fd7e14")

    # Age bars
    st.markdown("#### Age estimates at a glance")
    if face_age is not None:
        age_bar("Face model age",  face_age,  color="#4F8BF9")
        st.markdown("")
    if voice_age is not None:
        age_bar("Voice model age", voice_age, color="#20c997")
        st.markdown("")
    if fused_age is not None:
        age_bar("Fused age",       fused_age, color="#fd7e14")

    # Summary sentence
    st.divider()
    modality = "face and voice (fused)" if fused_age else ("face" if face_age else "voice")
    final_age = fused_age if fused_age is not None else (face_age if face_age is not None else voice_age)
    gender_str = f", predicted gender **{gender_label}**" if gender_label else ""
    st.success(
        f"Based on the **{modality}** input, the predicted age is "
        f"**{final_age:.1f} years**{gender_str}."
    )

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — About
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("About")
    st.markdown("""
This application demonstrates a **multimodal age and gender prediction** system
built as a final-year undergraduate project.

**Face model**
- Dataset: UTKFace (crop_part1 subset, 9,775 images)
- Architecture: ImprovedFaceModel (9.27 M params, dual-head CNN)
- Test MAE: **5.54 years** | R²: **0.881**

**Gender Classifier**
- Model: ViT (`dima806/man_woman_face_image_detection`)
- Evaluated completely detached from the age estimation task

**Voice model**
- Dataset: Mozilla Common Voice (73,768 samples)
- Architecture: VoiceAgeEstimator (4-block CNN on mel-spectrogram)
- Val MAE: **6.97 years** | Val R²: **0.798**

**Late fusion**
- Weighted average: 0.6 × face age + 0.4 × voice age

**Tech stack**
- PyTorch · Transformers · torchaudio · torchvision · Streamlit
    """)
