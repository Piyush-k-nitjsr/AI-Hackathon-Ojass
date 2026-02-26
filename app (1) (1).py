import streamlit as st
import requests
import json
import tempfile
import os
import pdfplumber

API_URL = "http://localhost:8000/analyze-chargesheet"

st.set_page_config(
    page_title="Chargesheet Analyzer",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Chargesheet Analyzer")
st.markdown("Upload or paste Hindi chargesheet text to get structured analysis.")


input_method = st.radio("Input method", ["Paste text", "Upload .txt file", "Upload PDF"], horizontal=True, key="input_method_radio")

text = ""
if input_method == "Paste text":
    text = st.text_area("Paste chargesheet text here", height=200)

elif input_method == "Upload .txt file":
    uploaded = st.file_uploader("Upload .txt file", type=["txt"])
    if uploaded:
        text = uploaded.read().decode("utf-8")
        st.success(f"File loaded — {len(text)} characters")

elif input_method == "Upload PDF":
    uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])
    if uploaded_pdf:
        with st.spinner("Extracting and preprocessing PDF... this may take a moment"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.read())
                    tmp_path = tmp.name

                from api import preprocess_pdf
                text = preprocess_pdf(tmp_path)

                os.unlink(tmp_path)  

                st.success(f"PDF preprocessed — {len(text)} characters extracted")
                with st.expander("Preview extracted text (first 1000 chars)"):
                    st.text(text[:1000])

            except ImportError:
                st.warning("combined_pipeline not found — running basic PDF extraction without preprocessing.")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.getvalue())
                    tmp_path = tmp.name
                raw_text = ""
                with pdfplumber.open(tmp_path) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            raw_text += extracted + "\n"
                os.unlink(tmp_path)
                text = raw_text
                st.success(f"PDF extracted (no preprocessing) — {len(text)} characters")
                with st.expander("Preview extracted text (first 1000 chars)"):
                    st.text(text[:1000])

            except Exception as e:
                st.error(f"Failed to process PDF: {e}")


if st.button("Analyze Chargesheet", type="primary"):
    if not text.strip():
        st.error("Please provide chargesheet text.")
    else:
        with st.spinner("Analyzing... this may take 10-20 seconds"):
            try:
                response = requests.post(API_URL, json={"text": text})
                result   = response.json()

                st.markdown("---")
                st.subheader("📋 Output A — Case Summary")

                summary = result["output_a_case_summary"]
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("FIR Number",     summary["fir_number"] or "Not found")
                    st.metric("FIR Date",        summary["fir_date"] or "Not found")

                with col2:
                    st.metric("Police Station",  summary["police_station"] or "Not found")
                    st.metric("Legal Sections",  ", ".join(summary["legal_sections"]) or "None")

                with col3:
                    st.metric("Accused",  ", ".join(summary["accused_names"]) or "Unknown")
                    st.metric("Victim",   ", ".join(summary["victim_names"]) or "Unknown")

                st.markdown("**Incident Summary:**")
                st.info(summary["incident_summary"] or "No summary available")

                st.markdown("---")
                st.subheader("🔍 Output B — Crime Classification")

                classification = result["output_b_classification"]

                conf_color = {
                    "HIGH":    "🟢",
                    "MEDIUM":  "🟡",
                    "LOW":     "🔴"
                }

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Crime Type",  classification["display_name"])
                    st.metric("Confidence",  f"{conf_color.get(classification['confidence'], '')} {classification['confidence']}")
                with col2:
                    st.metric("Matched Sections", ", ".join(classification["matched_sections"]) or "None")
                    st.metric("Reason", classification["reason"])

                
                st.markdown("---")
                st.subheader("✅ Output C — Missing Items Checklist")

                checklist = result["output_c_checklist"]

                if not checklist:
                    st.warning("No checklist available — crime type unknown")
                else:
                    present = [i for i in checklist if i["status"] == "PRESENT"]
                    partial = [i for i in checklist if i["status"] == "PARTIAL"]
                    missing = [i for i in checklist if i["status"] == "MISSING"]

                    col1, col2, col3 = st.columns(3)
                    col1.metric("✅ Present", len(present))
                    col2.metric("⚠️ Partial", len(partial))
                    col3.metric("❌ Missing", len(missing))

                    st.markdown("**Detailed Checklist:**")
                    for item in checklist:
                        with st.expander(f"{item['label']} {item['item']}"):
                            st.write(item["detail"])

               
                st.markdown("---")
                st.subheader("🏷️ Output D — Named Entity Recognition")

                entities = result.get("output_d_entities", {}).get("entities", [])

                if not entities:
                    st.warning("No entities extracted.")
                else:
                    type_counts = {}
                    for ent in entities:
                        t = ent.get("type", "UNKNOWN")
                        type_counts[t] = type_counts.get(t, 0) + 1

                    count_cols = st.columns(len(type_counts))
                    type_icons = {
                        "PERSON":        "👤",
                        "LEGAL_SECTION": "⚖️",
                        "DATE_TIME":     "📅",
                        "LOCATION":      "📍",
                        "DOCUMENT":      "📄",
                        "AMOUNT":        "💰",
                    }
                    for col, (etype, count) in zip(count_cols, type_counts.items()):
                        icon = type_icons.get(etype, "🔹")
                        col.metric(f"{icon} {etype}", count)

                    st.markdown("**All Extracted Entities:**")

                    grouped = {}
                    for ent in entities:
                        t = ent.get("type", "UNKNOWN")
                        grouped.setdefault(t, []).append(ent)

                    for etype, ents in grouped.items():
                        icon = type_icons.get(etype, "🔹")
                        with st.expander(f"{icon} {etype} ({len(ents)} found)"):
                            for ent in ents:
                                text_val = ent.get("text", "")
                                role     = ent.get("role", "")
                                event    = ent.get("event", "")
                                unit     = ent.get("unit", "")
                                badge    = role or event or unit
                                if badge:
                                    st.markdown(f"- **{text_val}** `{badge}`")
                                else:
                                    st.markdown(f"- **{text_val}**")

                st.markdown("---")
                st.download_button(
                    label="Download Full Report (JSON)",
                    data=json.dumps(result, ensure_ascii=False, indent=2),
                    file_name="final_report.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Error: {e}")