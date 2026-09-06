from pathlib import Path

TEXT_PATH = Path("legal_data/rules/extracted_text.txt")

text = TEXT_PATH.read_text(encoding="utf-8")

keywords = [
    "net quantity",
    "maximum retail price",
    "retail sale price",
    "manufacturer",
    "packer",
    "importer",
    "name and address",
    "consumer care",
    "declarations",
    "country of origin"
]

for keyword in keywords:
    print("\n" + "=" * 70)
    print(f"SEARCH: {keyword}")
    print("=" * 70)

    lower_text = text.lower()
    start = 0
    count = 0

    while True:
        position = lower_text.find(keyword.lower(), start)

        if position == -1:
            break

        beginning = max(0, position - 250)
        ending = min(len(text), position + 500)

        print(text[beginning:ending].replace("\n", " "))
        print("\n---")

        start = position + len(keyword)
        count += 1

        if count >= 5:
            break