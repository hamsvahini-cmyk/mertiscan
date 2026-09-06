from ocr_engine import run_ocr

print("==============================")
print("EASYOCR TEST")
print("==============================")

image_path = "test.jpg"

print(f"Testing image: {image_path}")
print("Starting OCR...")

results = run_ocr(image_path)

print(f"\nTotal detections: {len(results)}")
print("\n--- OCR TEXT ---")

for i, item in enumerate(results, 1):
    print(f"{i}. {item['text']}  | confidence: {item['confidence']:.2f}")

print("\n==============================")
print("OCR TEST COMPLETE")
print("==============================")