import difflib

import easyocr
import cv2
import numpy as np

_reader = None


def get_reader():
    """Load EasyOCR only once."""
    global _reader
    if _reader is None:
        print("Loading EasyOCR model...")
        _reader = easyocr.Reader(["en"], gpu=False)
        print("EasyOCR model loaded.")
    return _reader


def load_image(image_input):
    """Converts file path / bytes / uploaded file / ndarray into an OpenCV BGR image."""
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    elif isinstance(image_input, bytes):
        data = np.frombuffer(image_input, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif hasattr(image_input, "read"):
        data = np.frombuffer(image_input.read(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif isinstance(image_input, np.ndarray):
        image = image_input.copy()
    else:
        image = None

    if image is None:
        raise ValueError("Could not load image for OCR.")
    return image


def preprocess_variants(image):
    """
    Exactly 3 preprocessing variants — deliberately kept lean.

    We previously tried 7 (original/CLAHE/sharpened/adaptive/inverted-
    adaptive/OTSU/inverted-OTSU) and it made results WORSE: 224 OCR
    fragments instead of ~120, mostly 5-6 corrupted duplicate reads of
    the SAME text (e.g. "PEPSICO..." read 6 different broken ways).
    More passes just meant more garbage for the extractor to sift
    through, not more real information.

    3 variants covers the cases that actually matter:
    1. Original color image  — helps colored/branded logo text.
    2. CLAHE grayscale       — general contrast enhancement.
    3. Adaptive threshold    — handles uneven lighting on packaging.
    """
    variants = {}

    original_upscaled = cv2.resize(
        image.copy(), None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
    )
    variants["original"] = original_upscaled

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_upscaled = cv2.resize(
        gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_upscaled)
    variants["clahe"] = enhanced

    adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )
    variants["adaptive"] = adaptive

    return variants


def normalize_text(text):
    """Used only for duplicate comparison — never used as the displayed text."""
    if not text:
        return ""
    keep_chars = set("@.-/:%₹&'()+")
    return "".join(
        c.lower() for c in text if c.isalnum() or c in keep_chars
    ).strip()


def is_useful_text(text):
    if not text:
        return False
    text = text.strip()
    if len(text) < 1:
        return False
    return any(c.isalnum() for c in text)


def _similar(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def _fuzzy_dedup(results, threshold=0.82):
    """
    Multi-pass OCR often produces several corrupted copies of the SAME
    physical text (e.g. "PEPSICO INDIA HOLDINGS PVT. LTD." read 6
    different broken ways). Exact-match dedup alone doesn't catch
    these near-duplicates. This groups texts that are >82% similar
    and keeps only the highest-confidence (i.e. cleanest) version of
    each group, which meaningfully reduces noise before extraction.
    """
    # Sort by confidence so we build clusters around the best reads first.
    sorted_results = sorted(results, key=lambda r: r["confidence"], reverse=True)

    kept = []
    for result in sorted_results:
        norm = normalize_text(result["text"])
        if not norm:
            continue

        is_duplicate = False
        for existing in kept:
            existing_norm = normalize_text(existing["text"])
            # Only compare texts of roughly similar length to keep this fast
            # and avoid merging unrelated short fragments.
            if abs(len(norm) - len(existing_norm)) > max(6, 0.4 * len(existing_norm)):
                continue
            if _similar(norm, existing_norm) >= threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(result)

    return kept


def run_ocr(image_input):
    """
    Package Image -> 3 preprocessing variants -> EasyOCR -> confidence
    filter -> fuzzy dedup -> positional sort -> OCR results.
    """
    reader = get_reader()

    try:
        image = load_image(image_input)
    except ValueError:
        print("Warning: could not decode image. Returning empty OCR result.")
        return []

    variants = preprocess_variants(image)
    all_results = []

    print(f"Running OCR on {len(variants)} image variants...")

    for source_name, processed_image in variants.items():
        print(f"OCR pass: {source_name}")

        try:
            results = reader.readtext(
                processed_image,
                detail=1,
                paragraph=False,
                text_threshold=0.4,
                low_text=0.25,
                link_threshold=0.4,
                width_ths=0.7,
                mag_ratio=1.0
            )
        except Exception as e:
            print(f"Warning: OCR pass '{source_name}' failed: {e}")
            continue

        for item in results:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue

            bbox = item[0]
            text = str(item[1]).strip()
            try:
                confidence = float(item[2])
            except Exception:
                confidence = 0.0

            if not is_useful_text(text):
                continue
            if confidence < 0.3:
                continue

            all_results.append({
                "bbox": bbox,
                "text": text,
                "confidence": confidence,
                "source": source_name
            })

    # Exact-match dedup first (fast, catches most repeats)
    exact_unique = {}
    for result in all_results:
        norm = normalize_text(result["text"])
        if not norm:
            continue
        if norm not in exact_unique or result["confidence"] > exact_unique[norm]["confidence"]:
            exact_unique[norm] = result

    # Then fuzzy dedup to catch corrupted near-duplicates
    cleaned_results = _fuzzy_dedup(list(exact_unique.values()))

    def get_y_position(result):
        try:
            bbox = result.get("bbox")
            if bbox:
                return min(point[1] for point in bbox)
        except Exception:
            pass
        return float("inf")

    cleaned_results.sort(key=get_y_position)

    print("================================================")
    print(f"OCR DETECTED {len(cleaned_results)} UNIQUE TEXT REGIONS")
    print("================================================")
    for result in cleaned_results:
        print(f"[{result['confidence']:.2f}] {result['text']}")

    return cleaned_results


def convert_to_extractor_format(results):
    """Kept for compatibility with older test files."""
    return [[r["bbox"], r["text"], r["confidence"]] for r in results]