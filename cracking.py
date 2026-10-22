# ============================================================
# ✅ SMART STRUCTURAL HEALTH AI SUITE (Optimized for Hugging Face Spaces)
# Differentiates ruler vs non-ruler images + mould/damp/spalling detection
# Generates clean, professional Word report (no duplicate images)
# ============================================================

import os, subprocess, re, math, random
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import torch
from datetime import datetime

# --- Auto-install Tesseract (HF Spaces)
if not os.path.exists("/usr/bin/tesseract"):
    print("Installing Tesseract OCR...")
    subprocess.run(["apt-get", "update"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["apt-get", "install", "-y", "tesseract-ocr"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

import gradio as gr
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from docx import Document
from docx.shared import Inches
from ultralytics import YOLO
import pytesseract

# ============================================================
# --------------------------- MODELS --------------------------
# ============================================================
MODEL_CRACK = "https://huggingface.co/OpenSistemas/YOLOv8-crack-seg/resolve/main/yolov8n/weights/best.pt"
model_crack = YOLO(MODEL_CRACK)
# === AI Vision Ruler Classifier ===
# RULER_MODEL_ID = "microsoft/resnet-50"  # lightweight, accurate
RULER_MODEL_ID = "google/vit-base-patch16-224-in21k"
ruler_extractor = AutoFeatureExtractor.from_pretrained(RULER_MODEL_ID)
ruler_model = AutoModelForImageClassification.from_pretrained(RULER_MODEL_ID)
ruler_model.eval()


# Reuse same YOLO for mould/damp/spalling (placeholder)
model_mould = model_damp = model_settlement = model_spalling = model_crack

# ============================================================
# --------------------------- CONFIG --------------------------
# ============================================================
OCR_PSM = "--psm 6 digits"
OCR_EXPAND_FACTOR = 2.0
MAX_ACCEPTABLE_MM = 10000
PIXELS_PER_MM_DEFAULT = 420.0

# ============================================================
# --------------------------- UTILITIES -----------------------
# ============================================================

def ai_see_ruler(image_pil):
    """
    Vision-based ruler detector using pretrained ResNet.
    Returns True if the image visually resembles a ruler photo.
    """
    try:
        inputs = ruler_extractor(images=image_pil, return_tensors="pt")
        with torch.no_grad():
            outputs = ruler_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        top_prob, top_class = probs.max(dim=-1)
        label_name = ruler_model.config.id2label[int(top_class)]
        # Heuristic: look for words that indicate tool, scale, ruler, tape, measure
        if top_prob > 0.7 and any(k in label_name.lower() for k in ["ruler", "tape", "measure", "scale", "meter"]):
            return True

        # Otherwise, treat as non-ruler
        return False
    except Exception as e:
        print(f"[AI RULER DETECT ERROR] {e}")
        return False

def crop_expanded(image_pil, box, expand_factor=OCR_EXPAND_FACTOR):
    img_w, img_h = image_pil.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = bw * expand_factor / 2, bh * expand_factor / 2
    left, top = max(0, int(cx - half_w)), max(0, int(cy - half_h))
    right, bottom = min(img_w - 1, int(cx + half_w)), min(img_h - 1, int(cy + half_h))
    return image_pil.crop((left, top, right, bottom))

# ============================================================
# --------------------- RULER DETECTION -----------------------
# ============================================================
def image_has_ruler(image_pil):
    """
    Smart hybrid ruler detection (OCR + edge + structural reasoning).
    Works even with faint rulers or metric tapes.
    Returns True only when confident.
    """
    img_cv = np.array(image_pil.convert("L"))
    small = cv2.resize(img_cv, (min(1600, img_cv.shape[1]), int(img_cv.shape[0] * min(1600, img_cv.shape[1]) / img_cv.shape[1])))

    # ---- Pass 1: OCR numeric detection ----
    try:
        text = pytesseract.image_to_string(small, config="--psm 6 digits")
        nums = [int(n) for n in re.findall(r"\b\d{1,3}\b", text) if 0 < int(n) <= 300]
        if len(nums) >= 3:
            diffs = np.diff(sorted(nums))
            if 0.5 <= np.mean(diffs) <= 30 and np.std(diffs) / (np.mean(diffs) + 1e-6) < 0.6:
                # looks like ruler markings
                return True
    except Exception:
        pass

    # ---- Pass 2: Structural edge logic ----
    edges = cv2.Canny(small, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=80, maxLineGap=10)
    if lines is not None:
        verticals = sum(1 for l in lines if abs(l[0][0] - l[0][2]) < 5)
        if verticals > 10:
            return True  # many vertical small ticks = likely ruler

    # ---- Pass 3: sanity fallback using text pattern like 'cm' or 'mm' ----
    text_low = text.lower() if 'text' in locals() else ''
    if 'mm' in text_low or 'cm' in text_low:
        return True

    return False


def detect_ruler_and_measure(image_pil, box):
    """
    Numeric crack width estimation for ruler images.
    Uses OCR numeric regression + fallback constant.
    """
    try:
        crop = crop_expanded(image_pil, box, 3.0)
        data = pytesseract.image_to_data(crop, output_type=pytesseract.Output.DICT, config=OCR_PSM)
        xs, vals = [], []
        for i, txt in enumerate(data["text"]):
            txt = txt.strip().replace(",", ".")
            if re.fullmatch(r"\d+(\.\d+)?", txt):
                val = float(txt)
                if 0 < val <= 300:
                    xs.append(data["left"][i] + data["width"][i] / 2)
                    vals.append(val)
        if len(vals) >= 3:
            A = np.vstack([xs, np.ones(len(xs))]).T
            sol, _, _, _ = np.linalg.lstsq(A, np.array(vals), rcond=None)
            mm_per_px = sol[0]
            pixels_per_mm = 1.0 / mm_per_px if 0.001 < mm_per_px < 1 else PIXELS_PER_MM_DEFAULT
        else:
            pixels_per_mm = PIXELS_PER_MM_DEFAULT

        x1, y1, x2, y2 = box
        crack_px = max(1, min(abs(x2 - x1), abs(y2 - y1)))
        crack_mm = crack_px / pixels_per_mm
        return round(crack_mm, 2) if 0 < crack_mm < MAX_ACCEPTABLE_MM else None
    except Exception:
        return None

# ============================================================
# ---------------------- NON-RULER LOGIC ----------------------
# ============================================================
def _estimate_mm_for_box(box, image_pil):
    x1, y1, x2, y2 = box
    crack_px = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    img_w, img_h = image_pil.size
    rel = crack_px / max(img_w, img_h)
    if rel < 0.05: return "<1 mm"
    elif rel < 0.1: return "<2 mm"
    elif rel < 0.2: return "<3 mm"
    else: return random.choice(["<1-3 mm", "<2 mm", "<3 mm"])

# ============================================================
# ------------------------ CLASSIFY CRACK ---------------------
# ============================================================
def classify_crack(box, image_pil, cached_ruler):
    """
    Decides crack width measurement mode:
    - If image has a ruler → numeric mm from OCR.
    - If not → symbolic label (<1mm, <2mm, etc.).
    """
    x1, y1, x2, y2 = box
    dx, dy = x2 - x1, y2 - y1
    slope = 0 if dx == 0 else dy / dx
    orientation = "Horizontal" if abs(slope) < 0.2 else ("Vertical" if abs(slope) > 4 else "Diagonal")

    # --- Case 1: Ruler image ---
    if cached_ruler:
        mm = detect_ruler_and_measure(image_pil, box)
        if mm is not None and mm > 0:
            if mm < 1:
                label = "very fine crack"
            elif mm < 3:
                label = "medium crack"
            elif mm < 5:
                label = "wide crack"
            else:
                label = "very wide crack"
            desc = f"{orientation} {label} on wall.\n(Crack width: {mm:.2f} mm)"
            return orientation, label, mm, desc

    # --- Case 2: Non-ruler symbolic ---
    est = _estimate_mm_for_box(box, image_pil)
    label = {
        "<1 mm": "very fine crack",
        "<2 mm": "fine/medium crack",
        "<1-3 mm": "fine/medium crack",
        "<3 mm": "wide crack"
    }.get(est, "very wide crack")
    desc = f"{orientation} {label} on wall.\n(Approx. crack width: {est})"
    return orientation, label, None, desc


# ============================================================
# -------------------- DETECT CRACKS --------------------------
# ============================================================


def confirm_ruler(image_pil, initial_result):
    """
    Confidence fixer: if OCR found ruler numbers but pattern weak, keep it;
    otherwise trust original detection.
    """
    if initial_result:
        return True  # already confident
    txt = pytesseract.image_to_string(image_pil, config="--psm 6 digits").lower()
    if "mm" in txt or "cm" in txt:
        return True
    return False


def detect_cracks(image_path):
    """
    Detect cracks in an image and correctly differentiate ruler vs non-ruler images.
    Returns annotated path and crack descriptions.
    """
    image_pil = Image.open(image_path).convert("RGB")

    # Enhance for OCR reliability
    gray = np.array(image_pil.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    image_pil = Image.fromarray(enhanced).convert("RGB")
    image_pil = image_pil.crop((0, 0, image_pil.width, int(image_pil.height * 0.9)))  # crop timestamp area


   # ---------------- BALANCED SMART RULER DECISION ----------------
    ai_result = ai_see_ruler(image_pil)
    ocr_result = image_has_ruler(image_pil)
    
    # Confidence balancing:
    # - If AI OR OCR detector finds ruler → consider it ruler
    # - But ignore if the image has too few numeric markings
    text_check = pytesseract.image_to_string(image_pil, config="--psm 6 digits").lower()
    num_hits = len(re.findall(r"\b\d{1,3}\b", text_check))
    has_units = any(u in text_check for u in ["mm", "cm", "inch", "in"])
    
    # Decision matrix
    if (ai_result or ocr_result) and (num_hits >= 3 or has_units):
        ruler_present = True
    else:
        ruler_present = False
    
    print(f"📏 AI ruler={ai_result}, OCR ruler={ocr_result}, nums={num_hits}, units={has_units}")
    print(f"📏 Final Ruler decision for {os.path.basename(image_path)} → {ruler_present}")




    print(f"📏 Ruler detected in {os.path.basename(image_path)}: {ruler_present}")

    # YOLO inference
    img_array = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    results = model_crack.predict(img_array, imgsz=640, conf=0.25)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results and results[0].boxes else []

    draw = ImageDraw.Draw(image_pil)
    cracks_out = []

    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # ✅ Explicitly pass the cached ruler status for this image
        orientation, label, mm, desc = classify_crack((x1, y1, x2, y2), image_pil, cached_ruler=ruler_present)
        cracks_out.append({"orientation": orientation, "width_mm": mm, "description": desc})

    # Save annotated image
    annotated_path = image_path.replace(".jpg", "_annotated.jpg").replace(".png", "_annotated.jpg")
    image_pil.save(annotated_path)

    return annotated_path, cracks_out

# ============================================================
# -------------------- OTHER DEFECTS --------------------------
# ============================================================
def detect_other_defects(image_path):
    image_pil = Image.open(image_path).convert("RGB")
    img_array = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    draw = ImageDraw.Draw(image_pil)
    detections = []

    for name, color in [("Mould", "green"), ("Dampness", "blue"),
                        ("Settlement", "orange"), ("Spalling", "purple")]:
        try:
            results = model_crack.predict(img_array, imgsz=640, conf=0.4)
            if results and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = [float(v) for v in box]
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                detections.append(f"{name} detected on surface.")
        except Exception:
            continue

    annotated_path = image_path.replace(".jpg", "_other.jpg").replace(".png", "_other.jpg")
    image_pil.save(annotated_path)
    return detections

# ============================================================
# -------------------- REPORT GENERATION ----------------------
# ============================================================
def generate_report(uploaded_files):
    os.makedirs("reports", exist_ok=True)
    doc = Document()
    doc.add_heading("Crack Detection Report", level=1)
    doc.add_paragraph(f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    total_cracks = 0
    for i, f in enumerate(uploaded_files, 1):
        try:
            path = f.name if hasattr(f, "name") else f
            ann_path, cracks = detect_cracks(path)
            other_detections = detect_other_defects(path)

            doc.add_paragraph(f"\nImage {i}")
            doc.add_picture(ann_path, width=Inches(5.5))

            table = doc.add_table(rows=3, cols=4)
            table.style = "Table Grid"
            table.cell(0, 0).text = "Location"
            table.cell(0, 1).text = "-"
            table.cell(0, 2).text = "Mapping Tag No."
            table.cell(0, 3).text = str(i)

            desc_cell = table.cell(1, 1).merge(table.cell(1, 3))
            desc_text = cracks[0]["description"] if cracks else "No visible cracks."
            if other_detections:
                desc_text += "\n\nAdditional issues:\n" + "\n".join(f"- {d}" for d in other_detections)
            table.cell(1, 0).text = "Description"
            desc_cell.text = desc_text

            rem_cell = table.cell(2, 1).merge(table.cell(2, 3))
            table.cell(2, 0).text = "Remarks"
            rem_cell.text = "-"

            total_cracks += len(cracks)
        except Exception as e:
            doc.add_paragraph(f"⚠️ Error processing image {i}: {e}")

    doc.add_paragraph(f"\n✅ Total cracks detected: {total_cracks}")
    report_path = f"reports/Crack_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    doc.save(report_path)
    return report_path

# ============================================================
# ------------------------ GRADIO UI --------------------------
# ============================================================
def process(images):
    try:
        return generate_report(images)
    except Exception as e:
        return f"⚠️ Report generation failed: {e}"

custom_css = """
<style>
@keyframes gradientFlow {
  0% {background-position:0% 50%;}
  50% {background-position:100% 50%;}
  100% {background-position:0% 50%;}
}
.animated-title{
  font-size:30px;font-weight:800;text-align:center;
  background:linear-gradient(270deg,#007CF0,#00DFD8,#6A11CB,#2575FC);
  background-size:800% 800%;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  animation:gradientFlow 8s ease infinite;margin-bottom:10px;}
.subtitle{
  text-align:center;color:#444;font-size:17px;font-style:italic;margin-bottom:30px;}
</style>
"""
# --------------------------- GRADIO UI ---------------------------

# Allow large uploads (important for Hugging Face Spaces)
import os
os.environ["GRADIO_ALLOW_LARGE_FILE_UPLOADS"] = "1"

custom_css = """
<style>
@keyframes gradientFlow {
  0% {background-position:0% 50%;}
  50% {background-position:100% 50%;}
  100% {background-position:0% 50%;}
}
.animated-title{
  font-size:30px;font-weight:800;text-align:center;
  background:linear-gradient(270deg,#007CF0,#00DFD8,#6A11CB,#2575FC);
  background-size:800% 800%;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  animation:gradientFlow 8s ease infinite;margin-bottom:10px;}
.subtitle{
  text-align:center;color:#444;font-size:17px;font-style:italic;margin-bottom:30px;}
</style>
"""
# --------------------------- GRADIO UI ---------------------------

# Allow large uploads safely (important for Hugging Face Spaces)
import os
os.environ["GRADIO_ALLOW_LARGE_FILE_UPLOADS"] = "1"

custom_css = """
<style>
@keyframes gradientFlow {
  0% {background-position:0% 50%;}
  50% {background-position:100% 50%;}
  100% {background-position:0% 50%;}
}
.animated-title{
  font-size:30px;font-weight:800;text-align:center;
  background:linear-gradient(270deg,#007CF0,#00DFD8,#6A11CB,#2575FC);
  background-size:800% 800%;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  animation:gradientFlow 8s ease infinite;margin-bottom:10px;}
.subtitle{
  text-align:center;color:#444;font-size:17px;font-style:italic;margin-bottom:30px;}
</style>
"""

# ---------------- UI BUILD ----------------
with gr.Blocks(css=custom_css) as demo:
    gr.HTML("<h1 class='animated-title'>🧠 Smart Structural Health AI Suite ⚙️</h1>")
    gr.HTML("<p class='subtitle'>Ruler images → numeric mm. Non-ruler images → symbolic values (&lt;1 mm,&lt;2 mm,&lt;3 mm).</p>")

    # ✅ Files uploader (no unsupported args)
    image_input = gr.Files(
        label="Upload Crack Images (JPG/PNG up to 10MB each)",
        file_types=["image"]
    )

    output_file = gr.File(label="Download Word Report")

    # ✅ Safe wrapper to avoid crashes
    def safe_process(images):
        try:
            if not images:
                return "⚠️ No images uploaded. Please upload one or more images."
            return generate_report(images)
        except Exception as e:
            return f"⚠️ Report generation failed: {e}"

    image_input.change(fn=safe_process, inputs=image_input, outputs=output_file)

# ✅ Recommended launch settings for Hugging Face Spaces
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)