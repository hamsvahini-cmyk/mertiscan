import json
import cv2
from ocr_engine import run_ocr, convert_to_extractor_format
from extractor import extract_structured_data
from compliance_engine import load_compliance_rules, evaluate_compliance

def run_full_pipeline(image_path):
    print("=" * 50)
    print("SIH26034 PACKAGED FOOD COMPLIANCE CHECKER")
    print("=" * 50)

    # 1. Read Image
    print("\n[1/4] Loading Image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # 2. Run OCR
    print("[2/4] Running EasyOCR Engine...")
    ocr_results = run_ocr(image)
    extractor_format = convert_to_extractor_format(ocr_results)

    # 3. Extract Structured Fields
    print("[3/4] Extracting Label Fields to JSON Schema...")
    structured_label = extract_structured_data(extractor_format)

    # 4. Evaluate Legal Rules
    print("[4/4] Evaluating Legal Metrology & FSSAI Rules...")
    rules_data = load_compliance_rules()
    compliance_report = evaluate_compliance(structured_label, rules_data)

    # Full Output
    final_output = {
        "extracted_product_label": structured_label["product_label"],
        "compliance_report": compliance_report
    }

    print("\n================ COMPLIANCE ASSESSMENT REPORT ================")
    print(f"OVERALL STATUS : {compliance_report['overall_status']}")
    print(f"SUMMARY        : {json.dumps(compliance_report['summary'], indent=2)}")
    print("\nRULE EVALUATIONS:")
    for item in compliance_report["assessments"]:
        print(f" - [{item['status']}] {item['rule_id']}: {item['reason']}")

    with open("full_compliance_output.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print("\nSaved report to full_compliance_output.json")

if __name__ == "__main__":
    run_full_pipeline("test.jpg")