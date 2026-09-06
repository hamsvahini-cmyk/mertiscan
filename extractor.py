import re
import difflib


# ================================================================
# OCR NORMALIZATION
# ================================================================

def _get_min_y(bbox):
    try:
        return min(point[1] for point in bbox)
    except Exception:
        return float("inf")


def get_ocr_lines(ocr_results):
    lines = []

    for result in ocr_results:
        if isinstance(result, dict):
            text = result.get("text", "")
            confidence = result.get("confidence", 0.0)
            bbox = result.get("bbox")
        elif isinstance(result, (list, tuple)) and len(result) >= 2:
            bbox = result[0]
            text = result[1]
            confidence = result[2] if len(result) > 2 else 0.0
        else:
            continue

        text = str(text).strip()
        if not text:
            continue

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.0

        lines.append({
            "text": text,
            "confidence": confidence,
            "bbox": bbox,
            "y": _get_min_y(bbox) if bbox else float("inf")
        })

    lines.sort(key=lambda l: l["y"])
    return lines


def get_text_lines(ocr_results):
    return [line["text"] for line in get_ocr_lines(ocr_results)]


def clean_text(text):
    text = re.sub(r"^[\s\(\)\[\]\{\}'\".,:;\-]+", "", str(text))
    text = re.sub(r"[\s\(\)\[\]\{\}'\".,:;\-]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


# Words that are never a brand/product name by themselves — filters
# out one/two-word OCR junk fragments ("to", "on", "with", "these").
_STOPWORDS = {
    "to", "on", "for", "with", "these", "and", "of", "or", "is", "it",
    "all", "from", "at", "the", "a", "an", "in", "by", "us", "your",
    "this", "that", "add", "make", "add.", "life", "these:", "acd"
}

_NOISE_LABEL_PATTERNS = [
    r"^mrp\b", r"^m\.?r\.?p\.?\b", r"^net\s*(qty|quantity|weight)?\b",
    r"^ingredients?\b", r"^nutrition", r"^manufactured\b", r"^packed\b",
    r"^packer\b", r"^importer\b", r"^fssai\b", r"^f+s+a+[ti]\b",
    r"^license\b", r"^batch\b", r"^lot\b", r"^b\.?\s*no\b", r"^mfd\b",
    r"^mfg\b", r"^exp\b", r"^expiry\b", r"^best\s+before\b",
    r"^use\s+by\b", r"^country\s+of\s+origin\b", r"^made\s+in\b",
    r"^customer\s+care\b", r"^consumer\s+care\b", r"^helpline\b",
    r"^storage\b", r"^directions\b", r"^per\s*\d", r"^energy\b",
    r"^protein\b", r"^fat\b", r"^sodium\b", r"^carbohydrate\b"
]


def is_noise_line(text):
    t = text.lower().strip()
    return any(re.search(p, t, re.IGNORECASE) for p in _NOISE_LABEL_PATTERNS)


def looks_like_real_text(text):
    text = clean_text(text)
    if len(text) < 2:
        return False
    letters = sum(c.isalpha() for c in text)
    if letters == 0:
        return False
    if len(text) > 120:
        return False
    if letters < 0.4 * len(text):
        return False
    if text.lower() in _STOPWORDS:
        return False
    # Single-word fragments must be reasonably long to be meaningful
    words = text.split()
    if len(words) == 1 and len(text) < 3:
        return False
    if len(words) > 14:
        return False
    return True


# ================================================================
# COMPANY-NAME DETECTION (generic — works for any company suffix)
# ================================================================

_COMPANY_SUFFIX_PATTERN = re.compile(
    r"\b([A-Z][A-Za-z&.,'\- ]{2,60}?"
    r"(?:PVT\.?\s*LTD\.?|PRIVATE\s*LIMITED|LTD\.?|LIMITED|LLP|INC\.?))\b",
    re.IGNORECASE
)


def find_best_company_name(lines):
    """
    Scans every OCR line for a generic 'X ... Pvt Ltd / Limited / LLP'
    pattern. Multi-pass OCR often produces several corrupted copies of
    the same company name — this picks the cleanest, highest-confidence
    one instead of a random fragment. Works for ANY company, not a
    hardcoded brand.
    """
    candidates = []

    for line in lines:
        text = clean_text(line["text"])
        match = _COMPANY_SUFFIX_PATTERN.search(text)
        if not match:
            continue

        candidate = clean_text(match.group(1))
        if len(candidate) < 5:
            continue

        # Score: confidence + cleanliness (fewer stray punctuation/
        # digits = less corrupted OCR) + length (prefer fuller matches)
        junk_chars = sum(1 for c in candidate if c in ";:_~^*")
        clean_score = -junk_chars * 2
        score = line["confidence"] * 10 + clean_score + min(len(candidate), 40) * 0.1

        candidates.append((score, candidate))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


# ================================================================
# BRAND
# ================================================================

_KNOWN_BRAND_HINTS = [
    "lay's", "lays", "kurkure", "doritos", "bingo", "haldiram",
    "nestle", "nestlé", "britannia", "parle", "amul", "mtr", "itc",
    "maggi", "cadbury", "patanjali", "dabur", "everest", "mdh",
    "tata", "aashirvaad", "fortune", "saffola", "tropicana",
    "frooti", "sunfeast", "kellogg's", "kelloggs"
]


def extract_brand(lines, full_text):
    # 1. Explicit "Brand:" label
    match = re.search(
        r"\bbrand\s*(?:name)?\s*[:\-]\s*([A-Za-z0-9&.'\- ]{2,40})",
        full_text, re.IGNORECASE
    )
    if match:
        candidate = clean_text(match.group(1))
        if looks_like_real_text(candidate):
            return candidate

    # 2. Known-brand assistive list — checked against EVERY line
    # (punctuation-stripped), keeping the highest-confidence hit
    # rather than just the first substring found in the raw blob.
    # This is what fixes "(ays)" [0.99, corrupted] losing to
    # "Lays Indias" [0.85, clean] — we score by cleanliness, not
    # just raw confidence of a garbled read.
    hint_candidates = []
    for line in lines:
        cleaned = clean_text(line["text"])
        if is_noise_line(cleaned):
            continue
        lower = cleaned.lower()
        for hint in _KNOWN_BRAND_HINTS:
            if hint.replace("'", "") in lower.replace("'", ""):
                hint_candidates.append((line["confidence"], cleaned))
                break

    if hint_candidates:
        # Prefer short, concise, high-confidence matches — real brand
        # names are short (1-3 words); a long sentence that happens to
        # contain a brand-like substring (e.g. a manufacturer's legal
        # name mentioning "Nestle") is not what's printed as the brand.
        def _brand_score(c):
            confidence, txt = c
            word_count = len(txt.split())
            concise_bonus = 5 if word_count <= 3 else 0
            return (concise_bonus, confidence, -word_count)

        hint_candidates.sort(key=_brand_score, reverse=True)
        return hint_candidates[0][1]

    # 3. Positional/heuristic fallback for unknown brands
    candidates = []
    for line in lines:
        text = clean_text(line["text"])
        if not looks_like_real_text(text) or is_noise_line(text):
            continue

        word_count = len(text.split())
        if word_count > 4:
            continue

        score = line["confidence"] * 5
        if line["y"] != float("inf"):
            score += max(0, 10 - line["y"] / 200)
        if word_count <= 3:
            score += 4

        candidates.append((score, text))

    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    return "Not detected"


# ================================================================
# PRODUCT / FOOD NAME
# ================================================================

_FOOD_CATEGORY_KEYWORDS = {
    "noodle", "noodles", "biscuit", "biscuits", "cookie", "cookies",
    "chips", "snack", "snacks", "wafer", "wafers", "namkeen", "masala",
    "spice", "spices", "chocolate", "drink", "beverage", "juice",
    "sauce", "ketchup", "rice", "flour", "atta", "cereal", "milk",
    "coffee", "tea", "oil", "ghee", "pickle", "jam", "honey", "candy"
}

_LEGAL_CONTEXT_WORDS = {
    "manufactured", "manufacture", "packed", "packer", "importer",
    "ingredients", "nutrition", "information", "contains", "energy",
    "protein", "carbohydrate", "storage", "license", "fssai", "batch",
    "expiry", "mrp"
}


def extract_food_name(lines, full_text):
    for pattern in [
        r"\bproduct\s*(?:name)?\s*[:\-]\s*(.{2,80}?)(?=,|\.|\n|$)",
        r"\bfood\s*(?:name)?\s*[:\-]\s*(.{2,80}?)(?=,|\.|\n|$)",
    ]:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            candidate = clean_text(match.group(1))
            if looks_like_real_text(candidate):
                return candidate

    candidates = []
    for line in lines:
        text = clean_text(line["text"])
        if not looks_like_real_text(text) or is_noise_line(text):
            continue

        lower_words = set(re.findall(r"[a-z']+", text.lower()))
        if len(lower_words & _LEGAL_CONTEXT_WORDS) >= 2:
            continue

        word_count = len(text.split())
        score = 0

        # Substring match tolerates OCR suffix corruption
        # (e.g. "masalal" still recognized via "masala").
        category_hit = any(
            keyword in word or word in keyword
            for word in lower_words
            for keyword in _FOOD_CATEGORY_KEYWORDS
            if len(word) >= 4
        )
        if category_hit:
            score += 10

        if 1 <= word_count <= 6:
            score += 5
        elif 7 <= word_count <= 10:
            score += 2
        else:
            score -= 4

        score += line["confidence"] * 8

        # Product/variant names are usually capitalized on packaging;
        # a lowercase short fragment is more likely a chunk of a
        # longer marketing sentence (e.g. "cooked to" from "...cooked
        # to crispy perfection...") than an actual product name.
        if text[:1].isupper():
            score += 3
        elif word_count <= 3:
            score -= 3

        # Tie-breaker: prefer the fuller/more complete match
        score += min(len(text), 30) * 0.05

        candidates.append((score, text))

    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][1]

    return "Not detected"


def extract_common_or_generic_name(full_text, food_name):
    """
    Tries an explicit 'Common Name:' / 'Generic Name:' label first;
    only falls back to reusing the product name (not automatically
    identical by design) when no separate declaration exists.
    """
    for pattern in [
        r"\bcommon\s*(?:or\s*generic)?\s*name\s*[:\-]\s*(.{2,80}?)(?=,|\.|\n|$)",
        r"\bgeneric\s*name\s*[:\-]\s*(.{2,80}?)(?=,|\.|\n|$)",
    ]:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            candidate = clean_text(match.group(1))
            if looks_like_real_text(candidate):
                return candidate

    return food_name


# ================================================================
# NET QUANTITY  (label required — no risky bare-number fallback)
# ================================================================

def extract_net_quantity(text):
    unit_group = r"(kg|kgs|g|gm|gms|mg|l|ltr|litre|litres|liter|liters|ml)"
    aliases = {
        "kgs": "kg", "gm": "g", "gms": "g", "ltr": "l",
        "litre": "l", "litres": "l", "liter": "l", "liters": "l"
    }

    # Fuzzy-tolerant label: real OCR often mangles "Net Qty" into
    # "N.Qty", "MQty", "NETQTY" etc. — only trust an EXPLICIT label
    # match. We deliberately do NOT fall back to a bare "<num><unit>"
    # anywhere in the text anymore: that previously grabbed nutrition
    # panel numbers (e.g. "7g" sodium) whenever OCR garbled the word
    # "Sodium" beyond recognition. Wrong data is worse than "Not
    # detected" for a legal compliance tool.
    pattern = (
        r"\b(?:n\.?\s*[mn]?\s*qty|net\s*(?:qty|quantity|weight|wt|content)?)"
        r"\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*" + unit_group + r"\b"
    )

    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        raw_unit = match.group(2).lower()
        return {
            "value": match.group(1),
            "unit": aliases.get(raw_unit, raw_unit),
            "raw_text": match.group(0)
        }

    return {
        "value": "Not detected",
        "unit": "Not detected",
        "raw_text": "Not detected"
    }


# ================================================================
# MRP
# ================================================================

def extract_mrp(text):
    patterns = [
        r"\bmrp\s*[:\-]?\s*(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d{1,2})?)",
        r"\bm\.?\s*r\.?\s*p\.?\s*[:\-]?\s*(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d{1,2})?)",
        r"\bmaximum\s+retail\s+price\s*[:\-]?\s*(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d{1,2})?)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {"value": match.group(1), "currency": "INR", "raw_text": match.group(0)}

    if re.search(r"\bmrp\b|\bprice\b", text, re.IGNORECASE):
        match = re.search(r"(?:₹|rs\.?|inr)\s*(\d+(?:\.\d{1,2})?)", text, re.IGNORECASE)
        if match:
            return {"value": match.group(1), "currency": "INR", "raw_text": match.group(0)}

    return {"value": "Not detected", "currency": "INR", "raw_text": "Not detected"}


# ================================================================
# FSSAI
# ================================================================

def extract_fssai_license(text):
    contextual = re.search(
        r"\bf+s+a+[ti]?[^0-9]{0,25}(\d{14})",
        text, re.IGNORECASE
    )
    if contextual:
        return contextual.group(1)

    contextual = re.search(r"\blic[^0-9]{0,20}(\d{14})", text, re.IGNORECASE)
    if contextual:
        return contextual.group(1)

    matches = re.findall(r"\b\d{14}\b", text)
    if matches:
        return matches[0]

    return "Not detected"


# ================================================================
# BATCH / LOT
# ================================================================

def extract_batch_lot_code(text):
    match = re.search(
        r"\b(?:b[:\.]?\s*no|batch\s*(?:no|number)?|lot\s*(?:no|number|code)?)"
        r"\s*[:\-\.]?\s*([A-Za-z0-9\-\/]{2,30})",
        text, re.IGNORECASE
    )
    if match:
        value = match.group(1).strip()
        # Real batch/lot codes always contain at least one digit;
        # reject pure-letter OCR noise (e.g. a garbled adjacent word).
        if any(c.isdigit() for c in value):
            return value
    return "Not detected"


# ================================================================
# MANUFACTURER / PACKER / IMPORTER / BRAND OWNER
# ================================================================

def _extract_after_label(text, label_pattern):
    match = re.search(
        rf"(?:{label_pattern})\s*[:\-]?\s*(.{{3,100}}?)"
        r"(?=,|\.|\n|$|\bpin\b|\bfssai\b|\bmarketed\b|\bpacked\b|"
        r"\bimporter\b|\bimported\b|\bmanufactured\b)",
        text, re.IGNORECASE
    )
    if match:
        value = clean_text(match.group(1))
        if len(value) >= 3:
            return value
    return None


def extract_manufacturer(text, lines):
    # 1. Try explicit "Manufactured by:" label (works when OCR keeps
    #    label + value in one detection box).
    name = _extract_after_label(
        text,
        r"manufactured\s*(?:&|and)?\s*marketed\s*by|"
        r"manufactured\s*for|manufactured\s*by|mfd\.?\s*by"
    )
    if name:
        return {"name": name, "address": "Not detected"}

    # 2. Generic fallback: real-world OCR usually splits the label and
    #    the company name into SEPARATE detection boxes, so scan every
    #    line for a "<Name> Pvt Ltd / Limited" pattern instead.
    company = find_best_company_name(lines)
    if company:
        return {"name": company, "address": "Not detected"}

    return {"name": "Not detected", "address": "Not detected"}


def extract_packer(text):
    name = _extract_after_label(text, r"packed\s*by|pkd\.?\s*by")
    if name:
        return {"name": name, "address": "Not detected"}
    return "Not detected"


def extract_importer(text):
    name = _extract_after_label(text, r"imported\s*by|importer")
    if name:
        return {"name": name, "address": "Not detected"}
    return "Not detected"


def extract_brand_owner(text, manufacturer_name):
    name = _extract_after_label(text, r"marketed\s*by")
    if name:
        return name
    if manufacturer_name and manufacturer_name != "Not detected":
        return manufacturer_name
    return "Not detected"


# ================================================================
# CONSUMER CONTACT
# ================================================================

def extract_consumer_contact(text):
    phone = "Not detected"
    email = "Not detected"

    phone_match = re.search(r"\b1800[\s\-]?\d{2}[\s\-]?\d{4,6}\b", text)
    if phone_match:
        phone = phone_match.group(0)
    else:
        care_match = re.search(
            r"(?:consumer\s*care|customer\s*care|helpline|toll\s*free|telephone|tel)"
            r"[^0-9]{0,25}(\d{10})",
            text, re.IGNORECASE
        )
        if care_match:
            phone = care_match.group(1)

    email_match = re.search(r"\b[\w.\-+]+@[\w.\-]+\.\w+\b", text, re.IGNORECASE)
    if email_match:
        email = email_match.group(0)

    return {"telephone": phone, "email": email, "address": "Not detected"}


# ================================================================
# DATES
# ================================================================

def _find_date(text, label_pattern):
    date_pattern = r"(\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4})"
    match = re.search(rf"(?:{label_pattern})\s*[:\-]?\s*{date_pattern}", text, re.IGNORECASE)
    if match:
        return match.group(1)

    month_pattern = (
        r"((?:\d{1,2}\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)"
        r"[a-z]*\s+\d{2,4})"
    )
    match = re.search(rf"(?:{label_pattern})\s*[:\-]?\s*{month_pattern}", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return "Not detected"


def extract_date_marking(text):
    manufacturing_date = _find_date(text, r"\bmfd\b|\bmfg\b|manufactured|manufacturing\s*date")
    packaging_date = _find_date(text, r"packed\s*on|pkd\s*on|packaging\s*date")
    expiry_date = _find_date(text, r"expiry\s*date|exp\.?\s*date|\bexp\b")

    use_by = "Not detected"
    # Fuzzy-tolerant: real packaging OCR often garbles "MONTHS" into
    # "MONTKS"/"MORTHS" and "MANUFACTURE" gets similarly mangled — a
    # loose "best before ... from ... manufactur*" match still counts.
    match = re.search(
        r"best\s*before[^.\n]{0,80}",
        text, re.IGNORECASE
    )
    if match:
        use_by = clean_text(match.group(0))
    else:
        match = re.search(r"use\s*by[^.\n]{0,80}", text, re.IGNORECASE)
        if match:
            use_by = clean_text(match.group(0))

    return {
        "manufacturing_date": manufacturing_date,
        "packaging_date": packaging_date,
        "expiry_date": expiry_date,
        "use_by_date": use_by
    }


# ================================================================
# INGREDIENTS
# ================================================================

def extract_ingredients(text):
    match = re.search(
        r"\bingredients?\s*[:\-]?\s*(.{5,500}?)"
        r"(?=\bnutritional\s+information\b|\bnutrition\b|\bmanufactured\b|"
        r"\bmarketed\b|\bpacked\b|\bfssai\b|\bstorage\b|\bbest\s+before\b|"
        r"\ballergen\b|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    if match:
        value = clean_text(match.group(1))
        if len(value) > 10:
            return value
    return "Not detected"


# ================================================================
# NUTRITION / VEG-NONVEG / COUNTRY OF ORIGIN
# ================================================================

def extract_nutrition_info(text):
    if re.search(r"\bnutrition(?:al)?\s*(?:information|facts)?", text, re.IGNORECASE):
        return "Detected"
    keywords = ["energy", "protein", "carbohydrate", "total fat", "sodium"]
    hits = sum(1 for w in keywords if re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE))
    if hits >= 2:
        return "Detected"
    return "Not detected"


def extract_veg_nonveg(text):
    lower = text.lower()
    if re.search(r"\bnon[\s\-]?veg(?:etarian)?\b", lower):
        return "Non-Vegetarian"
    if re.search(r"\bveg(?:etarian)?\b", lower):
        return "Vegetarian"
    return "Needs visual verification"


def extract_country_of_origin(text):
    match = re.search(
        r"(?:country\s*of\s*origin|made\s*in|product\s*of)\s*[:\-]?\s*([A-Za-z ]{3,30})",
        text, re.IGNORECASE
    )
    if match:
        value = clean_text(match.group(1))
        value = re.split(r"\b(?:batch|mrp|fssai|net|ingredients)\b", value, flags=re.IGNORECASE)[0].strip()
        if value:
            return value
    return "Not detected"


# ================================================================
# MAIN EXTRACTION
# ================================================================

def extract_structured_data(ocr_results):
    lines = get_ocr_lines(ocr_results)
    full_text = "\n".join(line["text"] for line in lines)

    brand_name = extract_brand(lines, full_text)
    food_name = extract_food_name(lines, full_text)
    common_name = extract_common_or_generic_name(full_text, food_name)

    net_quantity = extract_net_quantity(full_text)
    mrp = extract_mrp(full_text)
    fssai_license = extract_fssai_license(full_text)

    manufacturer = extract_manufacturer(full_text, lines)
    packer = extract_packer(full_text)
    importer = extract_importer(full_text)
    brand_owner = extract_brand_owner(full_text, manufacturer["name"])

    structured_output = {
        "product_label": {
            "food_name": food_name,
            "common_or_generic_name": common_name,
            "brand_name": brand_name,

            "net_quantity": net_quantity,
            "maximum_retail_price": mrp,

            "ingredients": extract_ingredients(full_text),
            "nutrition_information": extract_nutrition_info(full_text),
            "veg_nonveg_declaration": extract_veg_nonveg(full_text),

            "manufacturer": manufacturer,
            "packer": packer,
            "importer": importer,
            "brand_owner": brand_owner,

            "fssai_license_number": fssai_license,
            "batch_lot_code": extract_batch_lot_code(full_text),
            "date_marking": extract_date_marking(full_text),
            "consumer_contact": extract_consumer_contact(full_text),

            "country_of_origin": extract_country_of_origin(full_text),

            "instructions_for_use": "Not detected",
            "allergen_declaration": "Not detected",
            "food_additives": "Not detected"
        }
    }
    return structured_output


def extract_product_label(ocr_results):
    return extract_structured_data(ocr_results)