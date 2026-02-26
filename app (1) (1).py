import streamlit as st
import requests
import json
import tempfile
import os
import pdfplumber

API_URL = "http://localhost:8000/analyze-chargesheet"

st.set_page_config(page_title="Chargesheet Analyzer", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }
    .section-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 4px solid #4f8bf9;
    }
    .section-card-green  { border-left-color: #22c55e; }
    .section-card-yellow { border-left-color: #eab308; }
    .section-card-purple { border-left-color: #a855f7; }
    .section-card-orange { border-left-color: #f97316; }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 6px;
    }
    .badge-green  { background: #14532d; color: #4ade80; }
    .badge-yellow { background: #422006; color: #fbbf24; }
    .badge-red    { background: #450a0a; color: #f87171; }
    .entity-pill {
        display: inline-block;
        background: #2d3148;
        border-radius: 8px;
        padding: 4px 12px;
        margin: 3px;
        font-size: 0.85rem;
        color: #e2e8f0;
    }
    .stMetric label { font-size: 0.78rem !important; color: #94a3b8 !important; }
    .stMetric [data-testid="metric-container"] { background: #1e2130; border-radius: 10px; padding: 0.8rem; }
    div[data-testid="stExpander"] { background: #1a1d2e; border-radius: 8px; border: 1px solid #2d3148; }
    .stButton>button { background: linear-gradient(135deg, #4f8bf9, #7c3aed); color: white; border: none; border-radius: 8px; padding: 0.6rem 2rem; font-weight: 600; font-size: 1rem; }
    .stButton>button:hover { opacity: 0.9; }
    h1 { background: linear-gradient(135deg, #4f8bf9, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Chargesheet Analyzer")
st.markdown("<p style='color:#94a3b8; margin-top:-10px;'>AI-powered legal document analysis for Indian police chargesheets</p>", unsafe_allow_html=True)
st.divider()

with st.container():
    input_method = st.radio("Select Input Method", ["Paste Text", "Upload TXT", "Upload PDF"], horizontal=True, key="input_method_radio")

text = ""

if input_method == "Paste Text":
    text = st.text_area("Paste chargesheet text here", height=220, placeholder="Paste Hindi or English chargesheet content...")

elif input_method == "Upload TXT":
    uploaded = st.file_uploader("Upload .txt file", type=["txt"])
    if uploaded:
        text = uploaded.read().decode("utf-8")
        st.success(f"✅ Loaded {len(text):,} characters")

elif input_method == "Upload PDF":
    uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])
    if uploaded_pdf:
        with st.spinner("Preprocessing PDF..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_pdf.read())
                    tmp_path = tmp.name
                from api import preprocess_pdf
                text = preprocess_pdf(tmp_path)
                os.unlink(tmp_path)
                st.success(f"✅ PDF preprocessed — {len(text):,} characters extracted")
                with st.expander("Preview extracted text"):
                    st.text(text[:1000])
            except ImportError:
                st.warning("Preprocessing unavailable — using basic extraction")
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
                st.success(f"✅ Extracted {len(text):,} characters")
                with st.expander("Preview extracted text"):
                    st.text(text[:1000])
            except Exception as e:
                st.error(f"Failed to process PDF: {e}")

st.divider()
analyze_btn = st.button("🔍 Analyze Chargesheet", type="primary", use_container_width=True)

def conf_color(score):
    if score is None:
        return "#94a3b8", "—"
    if score >= 0.8:
        return "#22c55e", f"{score:.2f}"
    if score >= 0.5:
        return "#eab308", f"{score:.2f}"
    return "#ef4444", f"{score:.2f}"

def conf_badge_html(score):
    color, label = conf_color(score)
    return f"<span style='background:{color}22; color:{color}; padding:2px 8px; border-radius:999px; font-size:0.75rem; font-weight:600;'>{label}</span>"

if analyze_btn:
    if not text.strip():
        st.error("Please provide chargesheet text first.")
    else:
        with st.spinner("Analyzing chargesheet... this may take 20–40 seconds"):
            try:
                response = requests.post(API_URL, json={"text": text}, timeout=300)
                result = response.json()

                summary        = result["output_a_case_summary"]
                classification = result["output_b_classification"]
                checklist      = result["output_c_checklist"]
                sem_checklist  = result.get("output_c2_semantic_checklist", {}).get("checklist", [])
                entities       = result.get("output_d_entities", {}).get("entities", [])
                field_conf     = summary.get("field_confidence", {})

                st.markdown("<br>", unsafe_allow_html=True)

                col_left, col_right = st.columns([1.2, 1])

                with col_left:
                    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                    st.markdown("### 📋 Case Summary")
                    st.divider()

                    r1c1, r1c2 = st.columns(2)
                    with r1c1:
                        sc = field_conf.get("fir_number")
                        st.metric("FIR Number", summary["fir_number"] or "—")
                        st.markdown(conf_badge_html(sc), unsafe_allow_html=True)
                    with r1c2:
                        sc = field_conf.get("fir_date")
                        st.metric("FIR Date", summary["fir_date"] or "—")
                        st.markdown(conf_badge_html(sc), unsafe_allow_html=True)

                    r2c1, r2c2 = st.columns(2)
                    with r2c1:
                        sc = field_conf.get("police_station")
                        st.metric("Police Station", summary["police_station"] or "—")
                        st.markdown(conf_badge_html(sc), unsafe_allow_html=True)
                    with r2c2:
                        sc = field_conf.get("legal_sections")
                        st.metric("Legal Sections", ", ".join(summary["legal_sections"]) or "—")
                        st.markdown(conf_badge_html(sc), unsafe_allow_html=True)

                    r3c1, r3c2 = st.columns(2)
                    with r3c1:
                        sc = field_conf.get("accused_names")
                        st.metric("Accused", ", ".join(summary["accused_names"]) or "—")
                        st.markdown(conf_badge_html(sc), unsafe_allow_html=True)
                    with r3c2:
                        sc = field_conf.get("victim_names")
                        st.metric("Victim", ", ".join(summary["victim_names"]) or "—")
                        st.markdown(conf_badge_html(sc), unsafe_allow_html=True)

                    st.markdown("**Incident Summary**")
                    st.info(summary["incident_summary"] or "No summary available")
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_right:
                    st.markdown("<div class='section-card section-card-green'>", unsafe_allow_html=True)
                    st.markdown("### 🔍 Crime Classification")
                    st.divider()

                    num_conf = classification.get("classification_confidence", 0.0)
                    color, label = conf_color(num_conf)

                    st.markdown(f"<h2 style='color:{color}; margin:0;'>{classification['display_name']}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color:#94a3b8; margin-top:4px;'>{classification['reason']}</p>", unsafe_allow_html=True)

                    cc1, cc2 = st.columns(2)
                    with cc1:
                        badge_map = {"HIGH": "#22c55e", "MEDIUM": "#eab308", "LOW": "#ef4444"}
                        bc = badge_map.get(classification["confidence"], "#94a3b8")
                        st.markdown(f"<p style='color:#94a3b8; font-size:0.8rem;'>CONFIDENCE LEVEL</p><span style='background:{bc}22; color:{bc}; padding:4px 14px; border-radius:999px; font-weight:700; font-size:1rem;'>{classification['confidence']}</span>", unsafe_allow_html=True)
                    with cc2:
                        st.markdown(f"<p style='color:#94a3b8; font-size:0.8rem;'>NUMERIC SCORE</p><span style='background:{color}22; color:{color}; padding:4px 14px; border-radius:999px; font-weight:700; font-size:1rem;'>{label}</span>", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    if classification["matched_sections"]:
                        st.markdown("**Matched Sections**")
                        pills = " ".join([f"<span class='entity-pill'>{s}</span>" for s in classification["matched_sections"]])
                        st.markdown(pills, unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                tab1, tab2, tab3, tab4 = st.tabs(["✅ Checklist (LLM)", "🧠 Semantic Checklist", "🏷️ Named Entities", "📥 Download"])

                with tab1:
                    if not checklist:
                        st.warning("No checklist available — crime type unknown.")
                    else:
                        present = [i for i in checklist if i["status"] == "PRESENT"]
                        partial = [i for i in checklist if i["status"] == "PARTIAL"]
                        missing = [i for i in checklist if i["status"] == "MISSING"]

                        tc1, tc2, tc3 = st.columns(3)
                        tc1.metric("✅ Present", len(present))
                        tc2.metric("⚠️ Partial", len(partial))
                        tc3.metric("❌ Missing", len(missing))
                        st.divider()

                        for item in checklist:
                            item_conf = item.get("confidence", 0.0)
                            color, label = conf_color(item_conf)
                            header = f"{item['label']} {item['item']}"
                            with st.expander(header):
                                st.markdown(f"<span style='background:{color}22; color:{color}; padding:2px 10px; border-radius:999px; font-size:0.8rem; font-weight:600;'>Confidence: {label}</span>", unsafe_allow_html=True)
                                st.markdown(f"<br>{item.get('detail', '')}", unsafe_allow_html=True)

                with tab2:
                    if not sem_checklist:
                        st.warning("Semantic checklist unavailable.")
                    else:
                        sp = [i for i in sem_checklist if i["status"] == "PRESENT"]
                        sw = [i for i in sem_checklist if i["status"] == "PARTIAL"]
                        sm = [i for i in sem_checklist if i["status"] == "MISSING"]

                        sc1, sc2, sc3 = st.columns(3)
                        sc1.metric("✅ Present", len(sp))
                        sc2.metric("⚠️ Partial", len(sw))
                        sc3.metric("❌ Missing", len(sm))
                        st.caption("TF-IDF cosine similarity — catches Hindi paraphrases of English checklist items")
                        st.divider()

                        for item in sem_checklist:
                            score = item.get("similarity_score", 0)
                            color, label = conf_color(score)
                            with st.expander(f"{item['label']} {item['item']}"):
                                st.markdown(f"<span style='background:{color}22; color:{color}; padding:2px 10px; border-radius:999px; font-size:0.8rem; font-weight:600;'>Similarity: {score:.4f}</span>", unsafe_allow_html=True)
                                if item.get("matched_text"):
                                    st.markdown("<br>**Best matching sentence:**", unsafe_allow_html=True)
                                    st.info(item["matched_text"])
                                else:
                                    st.markdown("<br>No match above threshold.", unsafe_allow_html=True)

                with tab3:
                    if not entities:
                        st.warning("No entities extracted.")
                    else:
                        type_icons = {
                            "PERSON": "👤", "LEGAL_SECTION": "⚖️", "DATE_TIME": "📅",
                            "LOCATION": "📍", "DOCUMENT": "📄", "AMOUNT": "💰"
                        }
                        type_counts = {}
                        for ent in entities:
                            t = ent.get("type", "UNKNOWN")
                            type_counts[t] = type_counts.get(t, 0) + 1

                        ec_cols = st.columns(len(type_counts))
                        for col, (etype, count) in zip(ec_cols, type_counts.items()):
                            col.metric(f"{type_icons.get(etype, '🔹')} {etype}", count)
                        st.divider()

                        grouped = {}
                        for ent in entities:
                            grouped.setdefault(ent.get("type", "UNKNOWN"), []).append(ent)

                        for etype, ents in grouped.items():
                            icon = type_icons.get(etype, "🔹")
                            with st.expander(f"{icon} {etype} — {len(ents)} found"):
                                pills_html = ""
                                for ent in ents:
                                    text_val = ent.get("text", "")
                                    badge = ent.get("role") or ent.get("event") or ent.get("unit") or ""
                                    if badge:
                                        pills_html += f"<span class='entity-pill'>{text_val} <span style='color:#94a3b8; font-size:0.7rem;'>{badge}</span></span>"
                                    else:
                                        pills_html += f"<span class='entity-pill'>{text_val}</span>"
                                st.markdown(pills_html, unsafe_allow_html=True)

                with tab4:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button(
                        label="⬇️ Download Full Report (JSON)",
                        data=json.dumps(result, ensure_ascii=False, indent=2),
                        file_name="chargesheet_report.json",
                        mime="application/json",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Error: {e}")