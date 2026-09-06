import cv2
import easyocr

from extractor import extract_information


# =========================================================
# 1. LOAD IMAGE
# =========================================================

image_path = "test.jpg"

image = cv2.imread(image_path)

if image is None:
    print("❌ Could not find test.jpg")
    exit()

print("✅ Image loaded successfully")


# =========================================================
# 2. UPSCALE IMAGE
# =========================================================

scale = 4

resized = cv2.resize(
    image,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_CUBIC
)

print("✅ Image enlarged")


# =========================================================
# 3. GRAYSCALE
# =========================================================

gray = cv2.cvtColor(
    resized,
    cv2.COLOR_BGR2GRAY
)

print("✅ Converted to grayscale")


# =========================================================
# 4. CLAHE CONTRAST ENHANCEMENT
# =========================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced = clahe.apply(gray)

print("✅ Contrast enhanced")


# =========================================================
# 5. DENOISING
# =========================================================

denoised = cv2.fastNlMeansDenoising(
    enhanced,
    None,
    10,
    7,
    21
)

print("✅ Noise reduced")


# =========================================================
# 6. SHARPENING
# =========================================================

blur = cv2.GaussianBlur(
    denoised,
    (0, 0),
    3
)

sharpened = cv2.addWeighted(
    denoised,
    1.5,
    blur,
    -0.5,
    0
)

print("✅ Image sharpened")


# =========================================================
# 7. SAVE PROCESSED IMAGE
# =========================================================

cv2.imwrite(
    "processed.jpg",
    sharpened
)

print("✅ Saved processed.jpg")


# =========================================================
# 8. OCR READER
# =========================================================

print("\n🔍 Starting EasyOCR...")

reader = easyocr.Reader(
    ["en"]
)


# =========================================================
# 9. OCR ORIGINAL ENLARGED IMAGE
# =========================================================

print("\n🔍 OCR pass 1: Enhanced image...")

results_1 = reader.readtext(
    sharpened,
    detail=1,
    paragraph=False
)


# =========================================================
# 10. OCR THRESHOLDED IMAGE
# =========================================================

print("\n🔍 OCR pass 2: Threshold image...")

threshold = cv2.adaptiveThreshold(
    sharpened,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    11
)

cv2.imwrite(
    "processed_threshold.jpg",
    threshold
)

results_2 = reader.readtext(
    threshold,
    detail=1,
    paragraph=False
)


# =========================================================
# 11. COMBINE RESULTS
# =========================================================

all_results = results_1 + results_2


# =========================================================
# 12. DISPLAY OCR RESULTS
# =========================================================

print(
    "\n========== OCR RESULTS ==========\n"
)

if len(all_results) == 0:

    print("❌ No text detected.")

else:

    for result in all_results:

        text = result[1]

        confidence = result[2]

        print(
            f"Text: {text}"
        )

        print(
            f"Confidence: {confidence:.2f}"
        )

        print(
            "--------------------------------"
        )


# =========================================================
# 13. STRUCTURED EXTRACTION
# =========================================================

print(
    "\n========== STRUCTURED DATA ==========\n"
)

package_data = extract_information(
    all_results
)

product = package_data["product_label"]


for key, value in product.items():

    print(
        f"{key}: {value}"
    )


# =========================================================
# 14. COMPLETE
# =========================================================

print(
    "\n===================================="
)

print(
    "✅ OCR → EXTRACTION COMPLETE"
)

print(
    "===================================="
)