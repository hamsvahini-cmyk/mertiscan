import json
import os

def load_rules():
    """
    Loads rules from legal_data/compliance_rules.json if present.
    The file structure is: {"rules": [ {id, field, requirement, legal_reference, failure_status, ...}, ... ]}
    Returns the list under the "rules" key (or [] if missing/unreadable).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rules_path = os.path.join(base_dir, "legal_data", "compliance_rules.json")

    if os.path.exists(rules_path):
        try:
            with open(rules_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("rules", [])
        except Exception as e:
            print(f"Warning: Could not load compliance_rules.json: {e}")
    return []

def evaluate_compliance(structured_data, rules_data=None):
    """
    Evaluates extracted product label data against all configured Legal Metrology & FSSAI rules.
    """
    label = structured_data.get("product_label", {})
    assessments = []

    if rules_data is None:
        rules_data = load_rules()

    # Rule 1: FSSAI License Number Check
    fssai_no = label.get("fssai_license_number", "Not detected")
    if fssai_no != "Not detected" and len(str(fssai_no)) == 14 and str(fssai_no).isdigit():
        assessments.append({
            "rule_id": "FSSAI_LICENSE",
            "title": "FSSAI License Number",
            "status": "COMPLIANT",
            "reason": f"Valid 14-digit FSSAI license detected: {fssai_no}",
            "legal_reference": "FSSAI (Labelling & Display) Reg 2020, Sec 5(2)"
        })
    else:
        assessments.append({
            "rule_id": "FSSAI_LICENSE",
            "title": "FSSAI License Number",
            "status": "NEEDS_REVIEW",
            "reason": "FSSAI License number could not be validated on packaging.",
            "legal_reference": "FSSAI (Labelling & Display) Reg 2020, Sec 5(2)"
        })

    # Rule 2: Food / Product Name
    food_name = label.get("food_name", "Not detected")
    if food_name != "Not detected":
        assessments.append({
            "rule_id": "FOOD_NAME",
            "title": "Product Name Declaration",
            "status": "COMPLIANT",
            "reason": f"Product name declared as '{food_name}'.",
            "legal_reference": "Legal Metrology Rule 6(1)(a) & FSSAI Sec 5(1)"
        })
    else:
        assessments.append({
            "rule_id": "FOOD_NAME",
            "title": "Product Name Declaration",
            "status": "NEEDS_REVIEW",
            "reason": "Product name missing or obscured in label crop.",
            "legal_reference": "Legal Metrology Rule 6(1)(a)"
        })

    # Rule 3: Net Quantity Check
    net_qty = label.get("net_quantity", {})
    if isinstance(net_qty, dict) and net_qty.get("value") != "Not detected":
        assessments.append({
            "rule_id": "NET_QUANTITY",
            "title": "Net Quantity Declaration",
            "status": "COMPLIANT",
            "reason": f"Net quantity declared: {net_qty.get('value')} {net_qty.get('unit')}",
            "legal_reference": "Legal Metrology Rule 6(1)(c)"
        })
    else:
        assessments.append({
            "rule_id": "NET_QUANTITY",
            "title": "Net Quantity Declaration",
            "status": "NEEDS_REVIEW",
            "reason": "Net quantity statement not clearly visible.",
            "legal_reference": "Legal Metrology Rule 6(1)(c)"
        })

    # Rule 4: Maximum Retail Price (MRP)
    mrp = label.get("maximum_retail_price", {})
    if isinstance(mrp, dict) and mrp.get("value") != "Not detected":
        assessments.append({
            "rule_id": "MRP_DECLARATION",
            "title": "Maximum Retail Price (MRP)",
            "status": "COMPLIANT",
            "reason": f"MRP declared: ₹{mrp.get('value')}",
            "legal_reference": "Legal Metrology Rule 6(1)(e)"
        })
    else:
        assessments.append({
            "rule_id": "MRP_DECLARATION",
            "title": "Maximum Retail Price (MRP)",
            "status": "NEEDS_REVIEW",
            "reason": "MRP declaration statement not found.",
            "legal_reference": "Legal Metrology Rule 6(1)(e)"
        })

    # Rule 5: Consumer Care Contact Information
    contact = label.get("consumer_contact", {})
    phone = contact.get("telephone") if isinstance(contact, dict) else "Not detected"
    email = contact.get("email") if isinstance(contact, dict) else "Not detected"
    if phone != "Not detected" or email != "Not detected":
        assessments.append({
            "rule_id": "CONSUMER_CARE",
            "title": "Consumer Care Details",
            "status": "COMPLIANT",
            "reason": f"Consumer support contact details detected.",
            "legal_reference": "Legal Metrology Rule 6(1)(aa)"
        })
    else:
        assessments.append({
            "rule_id": "CONSUMER_CARE",
            "title": "Consumer Care Details",
            "status": "NEEDS_REVIEW",
            "reason": "Consumer care helpline or email not detected.",
            "legal_reference": "Legal Metrology Rule 6(1)(aa)"
        })

    # Rule 6: Country of Origin
    origin = label.get("country_of_origin", "Not detected")
    if origin != "Not detected":
        assessments.append({
            "rule_id": "COUNTRY_OF_ORIGIN",
            "title": "Country of Origin",
            "status": "COMPLIANT",
            "reason": f"Country of origin declared as {origin}.",
            "legal_reference": "Legal Metrology Rule 6(1)(n)"
        })
    else:
        assessments.append({
            "rule_id": "COUNTRY_OF_ORIGIN",
            "title": "Country of Origin",
            "status": "NEEDS_REVIEW",
            "reason": "Country of origin field check required.",
            "legal_reference": "Legal Metrology Rule 6(1)(n)"
        })

    def _is_detected(value):
        """True only if the extractor actually found something for this field."""
        if value is None:
            return False
        if isinstance(value, dict):
            # A nested field (e.g. date_marking, net_quantity) counts as
            # detected only if at least one of its sub-values is real.
            return any(
                v not in (None, "Not detected", "")
                for v in value.values()
            )
        return value not in ("Not detected", "")

    # Dynamically append remaining checks from legal_data/compliance_rules.json
    evaluated_ids = {a["rule_id"] for a in assessments}
    for rule in rules_data:
        r_id = rule.get("id")
        if not r_id or r_id in evaluated_ids:
            continue

        title = rule.get("requirement", r_id)
        legal_reference = rule.get("legal_reference", "Legal Metrology / FSSAI Act")
        failure_status = rule.get("failure_status", "NEEDS_REVIEW")

        field_name = rule.get("field")
        field_value = label.get(field_name) if field_name else None

        if field_name and field_value == "Needs visual verification":
            status = "NEEDS_REVIEW"
            reason = (
                f"{title} — appears as a graphic symbol that OCR cannot "
                f"reliably read; needs manual visual verification."
            )
        elif field_name and _is_detected(field_value):
            status = "COMPLIANT"
            reason = f"{title} — detected on packaging."
        else:
            status = failure_status if field_name else "NEEDS_REVIEW"
            reason = f"{title} — pending manual crop verification." if not field_name \
                else f"{title} — not detected on packaging."

        assessments.append({
            "rule_id": r_id,
            "title": title,
            "status": status,
            "reason": reason,
            "legal_reference": legal_reference
        })

        evaluated_ids.add(r_id)

    # Determine overall status
    statuses = [a["status"] for a in assessments]
    if "POTENTIAL_VIOLATION" in statuses:
        overall = "POTENTIAL_VIOLATION"
    elif "NEEDS_REVIEW" in statuses:
        overall = "NEEDS_REVIEW"
    else:
        overall = "COMPLIANT"

    return {
        "overall_status": overall,
        "total_rules_checked": len(assessments),
        "assessments": assessments
    }