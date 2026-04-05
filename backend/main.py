"""
Deepfake Detection API — Ensemble of 5 Forensic Models
Hosts on Render/Railway, frontend on Netlify.
"""

import os, io, cv2, torch, numpy as np, mediapipe as mp, sys
import torch.nn as nn
import torchvision.transforms as T
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

# Make sure we can import from the same directory
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    # Put backend path first so local `models` resolves before any site-packages module.
    sys.path.insert(0, backend_dir)
# Project root for model weights
ROOT = os.path.dirname(backend_dir)

# ── Import model architectures ─────────────────────────────────────
from models.eye_model import EyeModel
from models.lip_model import LipModel
from models.nose_model import NoseModel
from models.skin_model import SkinModel
from models.geometry_model import GeometryClassifier

# ═══════════════════════════════════════════════════════════════════
#  GLOBALS
# ═══════════════════════════════════════════════════════════════════
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IN      = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_tf  = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(**IN)])
norm    = T.Normalize(**IN)

models: Dict[str, nn.Module] = {}
geometry_scaler = None  # (mean, std) arrays


def get_enabled_models() -> set:
    raw = os.getenv("ENABLED_MODELS", "all").strip().lower()
    valid = {"eye", "lip", "nose", "skin", "geometry"}
    if not raw or raw == "all":
        return valid
    requested = {x.strip() for x in raw.split(",") if x.strip()}
    enabled = requested & valid
    if not enabled:
        print("⚠️  ENABLED_MODELS had no valid entries. Falling back to all models.")
        return valid
    return enabled

# ── MediaPipe indices ──────────────────────────────────────────────
LEFT_EYE  = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
LIP_OUTER = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]
LIP_INNER = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]
ALL_LIP   = list(set(LIP_OUTER + LIP_INNER))
NOSE_IDX  = [1,2,98,327,168,6,197,195,5,4,19,94,370,141]
SKIN_REGIONS = {
    "left_cheek":  [234,227,116,123,147,213,192,214],
    "right_cheek": [454,447,345,352,376,433,416,434],
    "forehead":    [10,151,9,8,107,336,54,284],
    "chin":        [175,171,148,152,377,400,378,379],
}

# Geometry landmark indices for 52-dim feature vector
LM = {
    'left_eye_inner':   133,  'left_eye_outer':   33,
    'right_eye_inner':  362,  'right_eye_outer':  263,
    'left_eyebrow_in':  107,  'left_eyebrow_out': 70,
    'right_eyebrow_in': 336,  'right_eyebrow_out':300,
    'nose_tip':         1,    'nose_bridge':      6,
    'nose_base':        2,    'nose_left':        98,
    'nose_right':       327,
    'mouth_left':       61,   'mouth_right':      291,
    'upper_lip':        13,   'lower_lip':        14,
    'chin':             152,  'forehead':         10,
    'left_cheek':       234,  'right_cheek':      454,
    'jaw_left':         172,  'jaw_right':        397,
    'left_ear':         234,  'right_ear':        454,
}

# ═══════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════
def _load_weights(model, path):
    if not os.path.exists(path):
        print(f"⚠️  Not found: {path}")
        return False
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict):
        for k in ('model_state_dict', 'state_dict'):
            if k in ckpt:
                model.load_state_dict(ckpt[k]); return True
        model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return True

def load_all_models():
    global geometry_scaler
    # Use the robust ROOT defined at the top
    root = ROOT
    enabled = get_enabled_models()
    print(f" [+] Loading models from: {root}")
    print(f" [+] Enabled models: {', '.join(sorted(enabled))}")

    # Eye
    if "eye" in enabled:
        eye = EyeModel(dropout=0.4)
        if _load_weights(eye, os.path.join(root, "eye_model.pth")):
            models["eye"] = eye
            print("✅ Eye model loaded")

    # Lip
    if "lip" in enabled:
        lip = LipModel(dropout=0.4)
        if _load_weights(lip, os.path.join(root, "lip_model.pth")):
            models["lip"] = lip
            print("✅ Lip model loaded")

    # Nose
    if "nose" in enabled:
        nose = NoseModel(dropout=0.4)
        if _load_weights(nose, os.path.join(root, "nose_model.pth")):
            models["nose"] = nose
            print("✅ Nose model loaded")

    # Skin
    if "skin" in enabled:
        skin = SkinModel(dropout=0.4)
        if _load_weights(skin, os.path.join(root, "skin_model.pth")):
            models["skin"] = skin
            print("✅ Skin model loaded")

    # Geometry
    if "geometry" in enabled:
        geom = GeometryClassifier(input_dim=52, dropout=0.3)
        if _load_weights(geom, os.path.join(root, "geometry_model.pth")):
            models["geometry"] = geom
            print("✅ Geometry model loaded")

    # Scaler for geometry
    if "geometry" in enabled:
        scaler_path = os.path.join(root, "geometry_scaler.npy")
        if os.path.exists(scaler_path):
            s = np.load(scaler_path)
            geometry_scaler = (s[0], s[1])
            print("✅ Geometry scaler loaded")

# ═══════════════════════════════════════════════════════════════════
#  PREPROCESSING HELPERS (from user code)
# ═══════════════════════════════════════════════════════════════════
def apply_clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(2.0,(4,4)).apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def high_freq(bgr):
    hf = cv2.subtract(bgr, cv2.GaussianBlur(bgr,(5,5),0))
    return np.clip(hf.astype(np.float32)*8+128, 0, 255).astype(np.uint8)

def laplacian_map(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = np.uint8(np.clip(np.abs(cv2.Laplacian(g, cv2.CV_64F)), 0, 255))
    return cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

def color_normalize(bgr):
    out = bgr.astype(np.float32)
    for c in range(3):
        mn,mx = out[:,:,c].min(), out[:,:,c].max()
        if mx>mn: out[:,:,c] = (out[:,:,c]-mn)/(mx-mn)*255
    return out.astype(np.uint8)

def to_skin_tensors(bgr_patch):
    img = cv2.resize(bgr_patch, (224,224), interpolation=cv2.INTER_LANCZOS4)
    img = apply_clahe(img); img = color_normalize(img)
    def t(arr): return norm(T.ToTensor()(arr))
    rgb = t(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    hf  = t(cv2.cvtColor(high_freq(img), cv2.COLOR_BGR2RGB))
    lap = t(cv2.cvtColor(laplacian_map(img), cv2.COLOR_BGR2RGB))
    return rgb, hf, lap

# ── Geometry feature extraction (52 features) ─────────────────────
def extract_geometry_features(lms, h, w):
    """Extract 52-dim geometry feature vector from MediaPipe landmarks."""
    def pt(idx): return np.array([lms[idx].x*w, lms[idx].y*h])
    def dist(a, b): return np.linalg.norm(pt(a) - pt(b))

    feats = []
    # Inter-eye distance (normalizer)
    eye_dist = dist(LM['left_eye_outer'], LM['right_eye_outer']) + 1e-8

    # 1-6: Eye dimensions (6)
    feats.append(dist(LM['left_eye_inner'],  LM['left_eye_outer'])  / eye_dist)
    feats.append(dist(LM['right_eye_inner'], LM['right_eye_outer']) / eye_dist)
    feats.append(dist(LM['left_eye_inner'],  LM['right_eye_inner']) / eye_dist)
    feats.append(dist(33, 145) / eye_dist)   # left eye height
    feats.append(dist(263, 374) / eye_dist)   # right eye height
    feats.append(dist(LM['left_eye_outer'], LM['right_eye_outer']) / eye_dist)  # normalized=1

    # 7-10: Eyebrow dimensions (4)
    feats.append(dist(LM['left_eyebrow_in'],  LM['left_eyebrow_out'])  / eye_dist)
    feats.append(dist(LM['right_eyebrow_in'], LM['right_eyebrow_out']) / eye_dist)
    feats.append(dist(LM['left_eyebrow_in'],  LM['left_eye_inner'])  / eye_dist)
    feats.append(dist(LM['right_eyebrow_in'], LM['right_eye_inner']) / eye_dist)

    # 11-16: Nose dimensions (6)
    feats.append(dist(LM['nose_tip'],    LM['nose_bridge']) / eye_dist)
    feats.append(dist(LM['nose_left'],   LM['nose_right'])  / eye_dist)
    feats.append(dist(LM['nose_tip'],    LM['nose_base'])   / eye_dist)
    feats.append(dist(LM['nose_bridge'], LM['forehead'])    / eye_dist)
    feats.append(dist(LM['nose_left'],   LM['nose_tip'])    / eye_dist)
    feats.append(dist(LM['nose_right'],  LM['nose_tip'])    / eye_dist)

    # 17-22: Mouth dimensions (6)
    feats.append(dist(LM['mouth_left'],  LM['mouth_right']) / eye_dist)
    feats.append(dist(LM['upper_lip'],   LM['lower_lip'])   / eye_dist)
    feats.append(dist(LM['mouth_left'],  LM['upper_lip'])   / eye_dist)
    feats.append(dist(LM['mouth_right'], LM['upper_lip'])   / eye_dist)
    feats.append(dist(LM['mouth_left'],  LM['nose_tip'])    / eye_dist)
    feats.append(dist(LM['mouth_right'], LM['nose_tip'])    / eye_dist)

    # 23-28: Face proportions (6)
    feats.append(dist(LM['forehead'],    LM['chin'])         / eye_dist)
    feats.append(dist(LM['left_cheek'],  LM['right_cheek'])  / eye_dist)
    feats.append(dist(LM['jaw_left'],    LM['jaw_right'])    / eye_dist)
    feats.append(dist(LM['forehead'],    LM['nose_tip'])     / eye_dist)
    feats.append(dist(LM['nose_tip'],    LM['chin'])         / eye_dist)
    feats.append(dist(LM['left_cheek'],  LM['chin'])         / eye_dist)

    # 29-34: Symmetry ratios (6)
    feats.append(dist(LM['left_eye_outer'],  LM['nose_tip']) /
                (dist(LM['right_eye_outer'], LM['nose_tip']) + 1e-8))
    feats.append(dist(LM['left_eyebrow_out'],  LM['nose_tip']) /
                (dist(LM['right_eyebrow_out'], LM['nose_tip']) + 1e-8))
    feats.append(dist(LM['mouth_left'],  LM['nose_tip']) /
                (dist(LM['mouth_right'], LM['nose_tip']) + 1e-8))
    feats.append(dist(LM['left_cheek'],  LM['nose_tip']) /
                (dist(LM['right_cheek'], LM['nose_tip']) + 1e-8))
    feats.append(dist(LM['jaw_left'],    LM['nose_tip']) /
                (dist(LM['jaw_right'],   LM['nose_tip']) + 1e-8))
    feats.append(dist(LM['left_ear'],    LM['nose_tip']) /
                (dist(LM['right_ear'],   LM['nose_tip']) + 1e-8))

    # 35-40: Angles (6)
    def angle(a_idx, b_idx, c_idx):
        va = pt(a_idx) - pt(b_idx)
        vc = pt(c_idx) - pt(b_idx)
        cos = np.dot(va, vc) / (np.linalg.norm(va)*np.linalg.norm(vc) + 1e-8)
        return np.arccos(np.clip(cos, -1, 1)) / np.pi

    feats.append(angle(LM['left_eye_outer'],  LM['nose_tip'], LM['right_eye_outer']))
    feats.append(angle(LM['mouth_left'],      LM['nose_tip'], LM['mouth_right']))
    feats.append(angle(LM['left_eyebrow_out'],LM['forehead'], LM['right_eyebrow_out']))
    feats.append(angle(LM['left_cheek'],      LM['chin'],     LM['right_cheek']))
    feats.append(angle(LM['jaw_left'],        LM['chin'],     LM['jaw_right']))
    feats.append(angle(LM['forehead'],        LM['nose_tip'], LM['chin']))

    # 41-46: Vertical ratios (6)
    face_h = dist(LM['forehead'], LM['chin']) + 1e-8
    feats.append(dist(LM['forehead'],   LM['left_eyebrow_in'])  / face_h)
    feats.append(dist(LM['left_eyebrow_in'],  LM['left_eye_outer'])  / face_h)
    feats.append(dist(LM['left_eye_outer'],   LM['nose_tip'])   / face_h)
    feats.append(dist(LM['nose_tip'],         LM['upper_lip'])  / face_h)
    feats.append(dist(LM['upper_lip'],        LM['lower_lip'])  / face_h)
    feats.append(dist(LM['lower_lip'],        LM['chin'])       / face_h)

    # 47-52: Horizontal ratios (6)
    face_w = dist(LM['left_cheek'], LM['right_cheek']) + 1e-8
    feats.append(dist(LM['left_cheek'],       LM['left_eye_outer'])  / face_w)
    feats.append(dist(LM['left_eye_outer'],   LM['left_eye_inner'])  / face_w)
    feats.append(dist(LM['left_eye_inner'],   LM['right_eye_inner']) / face_w)
    feats.append(dist(LM['right_eye_inner'],  LM['right_eye_outer']) / face_w)
    feats.append(dist(LM['right_eye_outer'],  LM['right_cheek'])     / face_w)
    feats.append(dist(LM['nose_left'],        LM['nose_right'])      / face_w)

    return np.array(feats, dtype=np.float32)

# ── Patch extraction helpers ───────────────────────────────────────
def get_patch(bgr, lms, indices, h, w, pad=0.4):
    pts = np.array([[int(lms[i].x*w), int(lms[i].y*h)] for i in indices])
    x1,y1 = pts.min(axis=0); x2,y2 = pts.max(axis=0)
    px,py = int((x2-x1)*pad), int((y2-y1)*pad)
    x1,y1 = max(0,x1-px), max(0,y1-py)
    x2,y2 = min(w,x2+px), min(h,y2+py)
    if x2-x1<8 or y2-y1<8: return None
    return bgr[y1:y2, x1:x2]

# ═══════════════════════════════════════════════════════════════════
#  APP
# ═══════════════════════════════════════════════════════════════════
face_landmarker = None

def init_face_mesh():
    global face_landmarker
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    
    root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(root, "face_landmarker_v2_with_blendshapes.task")
    
    if not os.path.exists(model_path):
        print(f"❌ Face Landmarker model not found at {model_path}")
        return

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)
    print("✅ MediaPipe FaceLandmarker initialized (Tasks API)")

@asynccontextmanager
async def lifespan(app):
    init_face_mesh()
    load_all_models()
    yield

app = FastAPI(title="Deepfake Detection API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ═══════════════════════════════════════════════════════════════════
#  MAIN DETECTION ENDPOINT
# ═══════════════════════════════════════════════════════════════════
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    data  = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Invalid image")

    if face_landmarker is None:
        raise HTTPException(500, "Face detection engine not initialized")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    detection_result = face_landmarker.detect(mp_image)
    if not detection_result.face_landmarks:
        return {"face_detected": False, "verdict": "NO_FACE"}

    # Extract landmarks from the first detected face
    lms = detection_result.face_landmarks[0]
    h, w = bgr.shape[:2]
    analyses: Dict[str, Any] = {}
    all_real: List[float] = []

    # ── 1. EYE ─────────────────────────────────────────────────────
    if "eye" in models:
        for name, indices in [("left_eye", LEFT_EYE), ("right_eye", RIGHT_EYE)]:
            patch = get_patch(bgr, lms, indices, h, w, pad=0.4)
            if patch is not None:
                t = val_tf(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    probs = torch.softmax(models["eye"](t), dim=1)[0]
                rp = float(probs[0])  # class 0 = fake prob in user code
                analyses[name] = {"real_prob": 1.0-rp, "fake_prob": rp}
                all_real.append(1.0 - rp)

    # ── 2. LIP ─────────────────────────────────────────────────────
    if "lip" in models:
        patch = get_patch(bgr, lms, ALL_LIP, h, w, pad=0.3)
        if patch is not None:
            t = val_tf(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = models["lip"](t)
                probs = torch.softmax(out["logits"], dim=1)[0]
            rp = float(probs[1])  # class 1 = real? check user code
            analyses["lip"] = {
                "real_prob": rp,
                "artifact": float(out["artifact"][0]),
                "texture":  float(out["texture"][0]),
            }
            all_real.append(rp)

    # ── 3. NOSE ────────────────────────────────────────────────────
    if "nose" in models:
        patch = get_patch(bgr, lms, NOSE_IDX, h, w, pad=0.4)
        if patch is not None:
            t = val_tf(Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = models["nose"](t)
                probs = torch.softmax(out["logits"], dim=1)[0]
            rp = float(probs[1])
            analyses["nose"] = {
                "real_prob": rp,
                "geometry": float(out["geometry"][0]),
                "texture":  float(out["texture"][0]),
                "artifact": float(out["artifact"][0]),
            }
            all_real.append(rp)

    # ── 4. SKIN (triple-stream) ────────────────────────────────────
    if "skin" in models:
        skin_results = []
        for region, indices in SKIN_REGIONS.items():
            pts = np.array([[int(lms[i].x*w), int(lms[i].y*h)] for i in indices])
            if len(pts) < 3: continue
            cx, cy = pts.mean(axis=0).astype(int)
            half = 40
            x1,y1 = max(0,cx-half), max(0,cy-half)
            x2,y2 = min(w,cx+half), min(h,cy+half)
            if x2-x1<20 or y2-y1<20: continue
            patch = bgr[y1:y2, x1:x2]
            rgb_t, hf_t, lap_t = to_skin_tensors(patch)
            with torch.no_grad():
                out = models["skin"](
                    rgb_t.unsqueeze(0).to(DEVICE),
                    hf_t.unsqueeze(0).to(DEVICE),
                    lap_t.unsqueeze(0).to(DEVICE))
                probs = torch.softmax(out["logits"], dim=1)[0]
            skin_results.append({
                "region": region,
                "real_prob": float(probs[1]),
                "texture":  float(out["texture"][0]),
                "artifact": float(out["artifact"][0]),
                "frequency":float(out["frequency"][0]),
            })
        if skin_results:
            avg_skin = float(np.mean([r["real_prob"] for r in skin_results]))
            analyses["skin"] = {
                "real_prob": avg_skin,
                "regions": skin_results,
                "texture":  float(np.mean([r["texture"]   for r in skin_results])),
                "artifact": float(np.mean([r["artifact"]  for r in skin_results])),
                "frequency":float(np.mean([r["frequency"] for r in skin_results])),
            }
            all_real.append(avg_skin)

    # ── 5. GEOMETRY ────────────────────────────────────────────────
    if "geometry" in models and geometry_scaler is not None:
        try:
            feat = extract_geometry_features(lms, h, w)
            scaled = (feat - geometry_scaler[0]) / (geometry_scaler[1] + 1e-8)
            t = torch.FloatTensor(scaled).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = models["geometry"](t)
                probs = torch.softmax(out["logits"], dim=1)[0]
            rp = float(probs[1])
            analyses["geometry"] = {
                "real_prob": rp,
                "symmetry": float(out["symmetry"][0]),
                "deviation":float(out["deviation"][0]),
            }
            all_real.append(rp)
        except Exception:
            pass

    # ── ENSEMBLE VERDICT ───────────────────────────────────────────
    if all_real:
        avg = float(np.mean(all_real))
        if   avg >= 0.95: verdict = "DEFINITELY_REAL"
        elif avg >= 0.88: verdict = "LIKELY_REAL"
        elif avg >= 0.75: verdict = "SUSPICIOUS"
        elif avg >= 0.60: verdict = "LIKELY_FAKE"
        else:             verdict = "DEFINITELY_FAKE"
        return {
            "face_detected":     True,
            "verdict":           verdict,
            "overall_real_prob": round(avg, 4),
            "confidence":       round(abs(avg - 0.5) * 2, 4),
            "analyses":         analyses,
        }
    return {"face_detected": True, "verdict": "INCONCLUSIVE", "analyses": analyses}


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(models.keys())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
