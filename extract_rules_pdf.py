from pypdf import PdfReader

PDF_PATH = "legal_data/rules/packaged_commodities_rules_consolidated.pdf"
OUTPUT_PATH = "legal_data/rules/extracted_text.txt"

reader = PdfReader(PDF_PATH)

all_text = []

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()

    if text:
        all_text.append(
            f"\n\n===== PAGE {page_number} =====\n\n{text}"
        )

full_text = "".join(all_text)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    file.write(full_text)

print(f"Pages processed: {len(reader.pages)}")
print(f"Characters extracted: {len(full_text)}")
print(f"Saved to: {OUTPUT_PATH}")