import json
from datetime import datetime

import streamlit as st
import numpy as np
import cv2

from ocr_engine import run_ocr
from extractor import extract_product_label
from compliance_engine import evaluate_compliance

# ----------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------

st.set_page_config(
    page_title="METRISCAN | SIH26034",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------
# CUSTOM CSS — professional compliance/inspection dashboard theme
# ----------------------------------------------------------------

st.markdown("""
<style>
    :root {
        --bg: #0b0f17;
        --card: #141a24;
        --border: #262f40;
        --text: #e6ebf5;
        --muted: #8b97ac;
        --accent: #3b82f6;
        --header-dark: #05070c;
        --header-light: #101827;
        --green: #34d399;
        --green-bg: #0f2b21;
        --amber: #fbbf24;
        --amber-bg: #332708;
        --red: #f87171;
        --red-bg: #331414;
    }

    .stApp {
        background-color: var(--bg);
    }

    /* Force readable light text everywhere in the main content area,
       regardless of the browser's own light/dark preference — this
       is the fix for the earlier white-on-white bug, now applied
       consistently for a dark theme (light text on dark backgrounds
       throughout, no mixed light/dark overrides). */
    section.main, section.main * :not(svg):not(path) {
        color: var(--text) !important;
    }
    section.main .stApp, section.main .block-container {
        background-color: var(--bg) !important;
    }

    .metriscan-header {
        background: linear-gradient(135deg, var(--header-dark) 0%, var(--header-light) 100%);
        padding: 28px 36px;
        border-radius: 14px;
        margin-bottom: 24px;
        border: 1px solid var(--border);
    }
    .metriscan-header, .metriscan-header * {
        color: #ffffff !important;
    }
    .metriscan-header h1 {
        margin: 0;
        font-size: 30px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .metriscan-header p.tagline { color: #a9c2ea !important; font-size: 14.5px; margin: 4px 0 0 0; }
    .metriscan-header p.meta { color: #7893bd !important; font-size: 12.5px; margin: 14px 0 0 0; letter-spacing: 0.3px; }

    .ms-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .ms-card .ms-label {
        font-size: 11.5px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: var(--muted) !important;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .ms-card .ms-value {
        font-size: 15.5px;
        color: var(--text) !important;
        font-weight: 600;
        word-break: break-word;
    }

    .status-banner {
        border-radius: 14px;
        padding: 22px 26px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid var(--border);
    }
    .status-banner.compliant { background: var(--green-bg); }
    .status-banner.review { background: var(--amber-bg); }
    .status-banner.violation { background: var(--red-bg); }
    .status-banner h2 { margin: 0; font-size: 26px; letter-spacing: 0.5px; }
    .status-banner.compliant h2 { color: var(--green) !important; }
    .status-banner.review h2 { color: var(--amber) !important; }
    .status-banner.violation h2 { color: var(--red) !important; }
    .status-banner p { margin: 6px 0 0 0; color: var(--muted) !important; font-size: 13.5px; }

    .tally-chip {
        border-radius: 12px;
        padding: 14px 10px;
        text-align: center;
        border: 1px solid var(--border);
        background: var(--card);
    }
    .tally-chip .tally-num { font-size: 24px; font-weight: 700; color: #ffffff !important; }
    .tally-chip .tally-label { font-size: 12px; color: var(--muted) !important; margin-top: 2px; }

    .rule-compliant { border-left: 4px solid var(--green); }
    .rule-review { border-left: 4px solid var(--amber); }
    .rule-violation { border-left: 4px solid var(--red); }

    section[data-testid="stSidebar"] {
        background-color: var(--header-dark);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    div.stButton > button[kind="primary"] {
        background-color: var(--accent);
        border: none;
        border-radius: 8px;
        font-weight: 600;
        color: #ffffff !important;
    }

    div[data-testid="stExpander"] {
        background-color: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
    }
    .stTextInput input, .stRadio, .stDataFrame {
        background-color: var(--card) !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📦 METRISCAN")
    st.caption("AI-Powered Compliance Inspection")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Dashboard", "New Inspection", "Compliance Results", "Legal Rules", "About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**SIH 2026**")
    st.markdown("Problem Statement: `SIH26034`")
    st.markdown("Team: **NEXUS**")
    st.caption("K.L.N. College of Engineering")

# ----------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------

st.markdown("""
<div class="metriscan-header">
    <h1>METRISCAN</h1>
    <p class="tagline">AI-Powered Packaged Commodity Compliance System</p>
    <p class="meta">SIH26034 &nbsp;•&nbsp; Smart India Hackathon 2026 &nbsp;•&nbsp; TEAM NEXUS &nbsp;•&nbsp; K.L.N. College of Engineering</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------

if "report_ready" not in st.session_state:
    st.session_state.report_ready = False
    st.session_state.extracted_data = None
    st.session_state.compliance_report = None
    st.session_state.ocr_results = None
    st.session_state.image_rgb = None


def status_style(status):
    if status == "COMPLIANT":
        return "compliant", "🟢", "rule-compliant"
    if status == "NEEDS_REVIEW":
        return "review", "🟡", "rule-review"
    return "violation", "🔴", "rule-violation"


def render_status_banner(overall_status):
    css_class, emoji, _ = status_style(overall_status)
    label = overall_status.replace("_", " ")
    st.markdown(f"""
    <div class="status-banner {css_class}">
        <h2>{emoji} {label}</h2>
        <p>Overall compliance assessment based on detected packaging declarations</p>
    </div>
    """, unsafe_allow_html=True)


def render_info_card(col, label, value):
    with col:
        st.markdown(f"""
        <div class="ms-card">
            <div class="ms-label">{label}</div>
            <div class="ms-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)


def field_display(value, subfield=None):
    """Formats extracted field values (dicts, strings) for card display."""
    if isinstance(value, dict):
        if subfield:
            return value.get(subfield, "Not detected")
        parts = [str(v) for v in value.values() if v not in (None, "Not detected", "")]
        return ", ".join(parts) if parts else "Not detected"
    return value if value else "Not detected"


# ----------------------------------------------------------------
# PAGE: NEW INSPECTION
# ----------------------------------------------------------------

def render_new_inspection():
    st.subheader("📥 Package Inspection")
    st.caption("Upload a packaged commodity image to run an automated Legal Metrology & FSSAI compliance audit.")

    uploaded_file = st.file_uploader(
        "Upload Product Packaging Image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:
        return

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    if image is None:
        st.error("Could not read this image. Please upload a valid JPG/PNG file.")
        st.stop()

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image_rgb, caption="Uploaded Packaging", use_container_width=True)

    with col2:
        st.markdown("#### Ready to audit")
        st.write(
            "This image will be processed through EasyOCR, generic field "
            "extraction, and the Legal Metrology / FSSAI rule engine."
        )
        run_clicked = st.button("🔍 Run Compliance Audit", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("Running OCR and evaluating legal rules..."):
            ocr_results = run_ocr(image)
            extracted_data = extract_product_label(ocr_results)
            compliance_report = evaluate_compliance(extracted_data)

        st.session_state.report_ready = True
        st.session_state.extracted_data = extracted_data
        st.session_state.compliance_report = compliance_report
        st.session_state.ocr_results = ocr_results
        st.session_state.image_rgb = image_rgb

        st.success("Audit complete — see the **Compliance Results** page in the sidebar, or scroll down.")

    if st.session_state.report_ready:
        st.markdown("---")
        render_results_summary()


def render_results_summary():
    report = st.session_state.compliance_report
    data = st.session_state.extracted_data["product_label"]

    render_status_banner(report["overall_status"])

    assessments = report["assessments"]
    total = len(assessments)
    compliant = sum(1 for a in assessments if a["status"] == "COMPLIANT")
    review = sum(1 for a in assessments if a["status"] == "NEEDS_REVIEW")
    violation = sum(1 for a in assessments if a["status"] == "POTENTIAL_VIOLATION")

    t1, t2, t3, t4 = st.columns(4)
    for col, num, label in [
        (t1, total, "Total Rules"),
        (t2, compliant, "Compliant"),
        (t3, review, "Needs Review"),
        (t4, violation, "Potential Violations"),
    ]:
        with col:
            st.markdown(f"""
            <div class="tally-chip">
                <div class="tally-num">{num}</div>
                <div class="tally-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("#### Extracted Information")

    qty_value = field_display(data["net_quantity"], "value")
    qty_unit = field_display(data["net_quantity"], "unit")
    qty_display = f"{qty_value} {qty_unit}" if qty_value != "Not detected" else "Not detected"

    mrp_value = field_display(data["maximum_retail_price"], "value")
    mrp_display = f"₹{mrp_value}" if mrp_value != "Not detected" else "Not detected"

    c1, c2, c3, c4 = st.columns(4)
    render_info_card(c1, "Brand", field_display(data["brand_name"]))
    render_info_card(c2, "Product", field_display(data["food_name"]))
    render_info_card(c3, "Common / Generic Name", field_display(data["common_or_generic_name"]))
    render_info_card(c4, "Net Quantity", qty_display)

    c5, c6, c7, c8 = st.columns(4)
    render_info_card(c5, "MRP", mrp_display)
    render_info_card(c6, "FSSAI License", field_display(data["fssai_license_number"]))
    render_info_card(c7, "Manufacturer", field_display(data["manufacturer"], "name"))
    render_info_card(c8, "Country of Origin", field_display(data["country_of_origin"]))

    st.markdown("#### Declaration Status")

    contact_detected = (
        field_display(data["consumer_contact"], "telephone") != "Not detected"
        or field_display(data["consumer_contact"], "email") != "Not detected"
    )
    expiry_detected = any(
        data["date_marking"].get(k, "Not detected") != "Not detected"
        for k in ["expiry_date", "use_by_date"]
    )

    decl_items = [
        ("Net Quantity", qty_value != "Not detected"),
        ("MRP", mrp_value != "Not detected"),
        ("Batch/Lot", data["batch_lot_code"] != "Not detected"),
        ("Manufacturing Date", data["date_marking"].get("manufacturing_date", "Not detected") != "Not detected"),
        ("Expiry / Use By", expiry_detected),
        ("Ingredients", data["ingredients"] != "Not detected"),
        ("Nutrition Info", data["nutrition_information"] != "Not detected"),
        ("Veg / Non-Veg", data["veg_nonveg_declaration"]),
        ("Consumer Contact", contact_detected),
    ]

    cols = st.columns(3)
    for i, (label, state) in enumerate(decl_items):
        with cols[i % 3]:
            if state == "Needs visual verification":
                st.markdown(f"🟡 **{label}** — Needs visual verification")
            elif state is True:
                st.markdown(f"🟢 **{label}** — Detected")
            elif isinstance(state, str) and state not in ("Not detected",):
                st.markdown(f"🟢 **{label}** — {state}")
            else:
                st.markdown(f"🔴 **{label}** — Not detected")

    with st.expander("🔎 View OCR Details"):
        st.write("**OCR engine:** EasyOCR (multi-pass: CLAHE, adaptive threshold, OTSU)")
        st.write(f"**Text regions detected:** {len(st.session_state.ocr_results)}")
        rows = []
        for r in st.session_state.ocr_results:
            if isinstance(r, dict):
                rows.append({"Text": r.get("text", ""), "Confidence": round(r.get("confidence", 0), 2)})
            elif isinstance(r, (list, tuple)) and len(r) >= 3:
                rows.append({"Text": r[1], "Confidence": round(float(r[2]), 2)})
        st.dataframe(rows, use_container_width=True, height=280)

    jc1, jc2 = st.columns(2)
    with jc1:
        with st.expander("📄 View Extracted JSON"):
            st.json(st.session_state.extracted_data)
    with jc2:
        with st.expander("📄 View Compliance JSON"):
            st.json(report)

    st.markdown("#### Generate Report")
    report_bundle = {
        "generated_at": datetime.now().isoformat(),
        "problem_statement": "SIH26034",
        "team": "NEXUS",
        "extracted_data": st.session_state.extracted_data,
        "compliance_report": report
    }
    st.download_button(
        "⬇️ Download Inspection Report (JSON)",
        data=json.dumps(report_bundle, indent=2),
        file_name=f"metriscan_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


# ----------------------------------------------------------------
# PAGE: DASHBOARD
# ----------------------------------------------------------------

def render_dashboard():
    st.subheader("📊 Dashboard")

    if not st.session_state.report_ready:
        st.info("No inspection has been run yet. Go to **New Inspection** in the sidebar to upload a package image.")
        return

    render_results_summary()


# ----------------------------------------------------------------
# PAGE: COMPLIANCE RESULTS
# ----------------------------------------------------------------

def render_compliance_results():
    st.subheader("📋 Compliance Results")

    if not st.session_state.report_ready:
        st.info("No inspection has been run yet. Go to **New Inspection** in the sidebar to upload a package image.")
        return

    report = st.session_state.compliance_report
    render_status_banner(report["overall_status"])

    filter_choice = st.radio(
        "Filter",
        ["All", "Compliant", "Needs Review", "Potential Violations"],
        horizontal=True
    )
    status_map = {
        "Compliant": "COMPLIANT",
        "Needs Review": "NEEDS_REVIEW",
        "Potential Violations": "POTENTIAL_VIOLATION"
    }

    for item in report["assessments"]:
        if filter_choice != "All" and item["status"] != status_map[filter_choice]:
            continue

        _, emoji, css_class = status_style(item["status"])
        with st.expander(f"{emoji} {item['title']}  —  {item['status'].replace('_', ' ')}"):
            st.markdown(f"""
            <div class="ms-card {css_class}">
                <p>{item['reason']}</p>
                <p style="color:#6b7a90; font-size:12.5px; margin-top:8px;">
                    <strong>Legal reference:</strong> {item['legal_reference']}
                </p>
            </div>
            """, unsafe_allow_html=True)


# ----------------------------------------------------------------
# PAGE: LEGAL RULES
# ----------------------------------------------------------------

def render_legal_rules():
    st.subheader("⚖️ Legal Rules")
    st.caption("Rules loaded from legal_data/compliance_rules.json")

    try:
        with open("legal_data/compliance_rules.json", "r", encoding="utf-8") as f:
            rules_data = json.load(f)
    except Exception as e:
        st.error(f"Could not load compliance_rules.json: {e}")
        return

    rules = rules_data.get("rules", [])
    st.metric("Configured Rules", len(rules))

    search = st.text_input("Search rules", placeholder="e.g. FSSAI, net quantity, batch")

    for rule in rules:
        haystack = f"{rule.get('id','')} {rule.get('field','')} {rule.get('requirement','')}".lower()
        if search and search.lower() not in haystack:
            continue
        with st.expander(f"{rule.get('id', 'RULE')} — {rule.get('field', '')}"):
            st.write(rule.get("requirement", ""))
            st.caption(f"Legal reference: {rule.get('legal_reference', 'N/A')}")
            st.caption(f"Automation level: {rule.get('automation_level', 'N/A')} • Failure status: {rule.get('failure_status', 'N/A')}")


# ----------------------------------------------------------------
# PAGE: ABOUT
# ----------------------------------------------------------------

def render_about():
    st.subheader("ℹ️ About METRISCAN")
    st.markdown("""
**METRISCAN** is an AI-assisted inspection prototype that checks packaged
commodity labels for compliance with:

- **Legal Metrology (Packaged Commodities) Rules, 2011**
- **FSSAI (Labelling & Display) Regulations, 2020**

**Pipeline:**

Uploaded image → Multi-pass EasyOCR (CLAHE / adaptive threshold / OTSU) →
generic deterministic text extraction → structured product data →
Legal Metrology + FSSAI rule engine → compliance assessment.

**Important note:** This is an AI-assisted inspection aid, not a legal
authority. Where OCR cannot confidently confirm a declaration, the system
reports **Needs Review** rather than asserting a violation.

---

**Problem Statement:** SIH26034 — Software System to Check Compliance of
Packaged Commodities under Legal Metrology
**Team:** NEXUS
**College:** K.L.N. College of Engineering
""")


# ----------------------------------------------------------------
# ROUTER
# ----------------------------------------------------------------

if page == "Dashboard":
    render_dashboard()
elif page == "New Inspection":
    render_new_inspection()
elif page == "Compliance Results":
    render_compliance_results()
elif page == "Legal Rules":
    render_legal_rules()
elif page == "About":
    render_about()