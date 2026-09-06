import json
from ocr_engine import run_ocr
from extractor import extract_product_label
from compliance_engine import evaluate_compliance

def test_pipeline():
    image_path = "test.jpg"
    print(f"--- Running Full Pipeline Test on {image_path} ---")
    
    # 1. OCR Execution
    ocr_results = run_ocr(image_path)
    print(f"Total Unique Detections: {len(ocr_results)}")
    
    # 2. Field Extraction
    structured_json = extract_product_label(ocr_results)
    
    # 3. Compliance Assessment
    compliance_report = evaluate_compliance(structured_json)
    
    # 4. Display Outputs
    print("\n--- Extracted Label JSON ---")
    print(json.dumps(structured_json, indent=2))
    
    print("\n--- Legal Assessment Output ---")
    print(json.dumps(compliance_report, indent=2))

if __name__ == "__main__":
    test_pipeline()