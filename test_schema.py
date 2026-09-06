from extractor import extract_information


fake_ocr = [
    [[0, 0], "LAY'S", 0.95],
    [[0, 0], "CLASSIC SALTED", 0.94],
    [[0, 0], "MRP ₹50", 0.95],
    [[0, 0], "NET QUANTITY 100 g", 0.92],
    [[0, 0], "Mfd. By: ABC Foods Pvt Ltd", 0.90],
    [[0, 0], "B. NO: AB123", 0.89],
    [[0, 0], "MFD: 08/2026", 0.91],
    [[0, 0], "FSSAI LIC NO 12345678901234", 0.93],
    [[0, 0], "Consumer Care: 9876543210", 0.90]
]


data = extract_information(fake_ocr)

product = data["product_label"]


required_fields = [
    "food_name",
    "common_or_generic_name",
    "brand_name",
    "net_quantity",
    "maximum_retail_price",
    "ingredients",
    "nutrition_information",
    "veg_nonveg_declaration",
    "manufacturer",
    "packer",
    "importer",
    "brand_owner",
    "fssai_license_number",
    "batch_lot_code",
    "date_marking",
    "consumer_contact",
    "country_of_origin",
    "instructions_for_use",
    "allergen_declaration",
    "food_additives"
]


print("\n==============================")
print("SCHEMA VALIDATION")
print("==============================\n")


missing_fields = []


for field in required_fields:

    if field in product:
        print(f"✅ {field}")

    else:
        print(f"❌ {field}")
        missing_fields.append(field)


print("\n==============================")


if len(missing_fields) == 0:

    print("✅ SCHEMA STRUCTURE VALID")

else:

    print("❌ MISSING FIELDS:")
    print(missing_fields)


print("==============================")