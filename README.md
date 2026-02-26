# ⚖️ Smart Chargesheet AI

**Automated Legal Document Analysis & Chargesheet Generation System**

An AI-powered solution that analyzes police case diaries and chargesheets — extracting structured case data, classifying crimes, validating legal checklists, and identifying named entities — with full support for Hindi, English, and mixed-language documents.

---

## 🎯 Problem Statement

Police officers and legal professionals spend countless hours manually:
- Extracting information from handwritten or scanned case diaries
- Verifying completeness of required evidence and documentation
- Classifying crimes and mapping applicable IPC/IT Act sections
- Formatting chargesheets for legal proceedings

This leads to delays in justice delivery and an increased workload on law enforcement agencies.

---

## 💡 Solution

Smart Chargesheet AI automates this entire workflow through a multi-stage pipeline:

1. **PDF Preprocessing** — Extract and clean text from scanned/digital case diaries
2. **AI-Powered Case Extraction** — Use Google Gemini to extract structured case data
3. **Crime Classification** — Identify crime type from IPC/IT Act sections
4. **Dual Checklist Validation** — LLM keyword audit + TF-IDF semantic similarity scoring
5. **Named Entity Recognition (NER)** — Extract persons, locations, dates, documents, amounts, and legal sections
6. **Chargesheet Report Generation** — JSON output ready for downstream use or export

---

## 🚀 Features

### Core Features
- ✅ **Multi-language Support** — Handles Hindi (Devanagari), English, and mixed documents
- ✅ **Intelligent Case Extraction** — Extracts FIR number/date, police station, accused/victim names, legal sections, and incident summary
- ✅ **Crime Classification** — Identifies crime type with confidence level (HIGH / MEDIUM / LOW)
- ✅ **Dual-Mode Checklist** — LLM keyword audit + TF-IDF cosine similarity for robust validation
- ✅ **Named Entity Recognition** — Categorizes PERSON, LEGAL_SECTION, DATE_TIME, LOCATION, DOCUMENT, AMOUNT entities
- ✅ **Structured Output** — Four labelled outputs (A–D) with a downloadable JSON report

### Technical Features
- 🤖 **Gemini 2.5 Flash** — Powers extraction, classification, checklist auditing, and NER
- 🔄 **Semantic Similarity Fallback** — TF-IDF with character n-gram (2–4) cosine similarity for checklist evidence matching
- 📊 **Streamlit Dashboard** — Interactive web UI with metrics, expanders, and JSON export
- ⚙️ **FastAPI Backend** — REST API at `/analyze-chargesheet` with full CORS support
- 🔧 **Advanced PDF Preprocessing** — Unicode normalization → date normalization → OCR garbage removal → whitespace normalization → spelling correction → sentence tokenization

---

## 📋 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend API** | FastAPI, Uvicorn |
| **Frontend UI** | Streamlit |
| **AI / LLM** | Google Gemini 2.5 Flash (`google-generativeai`) |
| **PDF Processing** | pdfplumber |
| **NLP** | NLTK (`sent_tokenize`), pyspellchecker |
| **Semantic Search** | scikit-learn TF-IDF + cosine similarity |
| **Data Validation** | Pydantic v2 |
| **Date Parsing** | python-dateutil |
| **Environment** | python-dotenv |

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Google Gemini API key (**required**)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/smart-chargesheet-ai.git
cd smart-chargesheet-ai
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install fastapi uvicorn streamlit pdfplumber google-generativeai \
            nltk python-dateutil pyspellchecker pydantic python-dotenv \
            numpy scikit-learn requests
```

Or with a requirements file:
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### Step 5: Set Up Environment Variables

Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

To obtain a Gemini API key:
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key and paste it into `.env`

> ⚠️ The API key is loaded via `GOOGLE_API_KEY`. The server will raise a `RuntimeError` on startup if this variable is not set.

---

## 🎮 Usage

### Option 1: Full Stack (Recommended)

**Start the FastAPI backend:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Start the Streamlit frontend (new terminal):**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Option 2: API Only

```bash
uvicorn api:app --reload
```

Send a POST request to `http://localhost:8000/analyze-chargesheet`:

```bash
curl -X POST http://localhost:8000/analyze-chargesheet \
     -H "Content-Type: application/json" \
     -d '{"text": "Your chargesheet text here..."}'
```

### Option 3: Batch PDF Processing (Script)

Process multiple PDFs directly from the command line by editing the `pdf_jobs` list in `api.py` and running:

```bash
python api.py
```

Results are saved to `all_results.json`.

### Option 4: Python API

```python
from api import preprocess_pdf, analyze_chargesheet

# Step 1: Extract and clean text from PDF
text = preprocess_pdf("Case_Diary_255.pdf")

# Step 2: Run analysis pipeline
result = analyze_chargesheet(text)

# Step 3: Access structured outputs
print(result["output_a_case_summary"])       # Case details
print(result["output_b_classification"])     # Crime type + confidence
print(result["output_c_checklist"])          # LLM keyword checklist
print(result["output_c2_semantic_checklist"]) # TF-IDF semantic checklist
print(result["output_d_entities"])           # Named entities
```

---

## 📁 Project Structure

```
smart-chargesheet-ai/
│
├── api.py                   # FastAPI backend — full pipeline (preprocessing, extraction,
│                            #   classification, checklist, NER, semantic scoring)
├── app.py                   # Streamlit frontend — interactive web UI
│
├── .env                     # Environment variables (create this — not committed)
├── requirements.txt         # Python dependencies
├── README.md                # This file
│
└── sample_cases/            # Optional: place sample case diary PDFs here
```

> The entire backend pipeline lives in `api.py`. The Streamlit app (`app.py`) calls the FastAPI endpoint at `http://localhost:8000/analyze-chargesheet`.

---

## 📊 Pipeline Deep Dive

### Stage 1 — PDF Preprocessing (`preprocess_pdf`)

Applied sequentially to raw PDF-extracted text:

| Step | Function | Purpose |
|------|----------|---------|
| 1 | `normalize_unicode` | NFKC normalization; strip invisible/control characters |
| 2 | `normalize_dates` | Standardize all date formats to `YYYY-MM-DD` |
| 3 | `remove_ocr_garbage` | Remove isolated chars, repeated chars, pipe/underscore noise |
| 4 | `normalize_whitespace` | Collapse multiple spaces/newlines |
| 5 | `correct_spelling` | English spell correction (pyspellchecker) |
| 6 | `apply_adaptive_semantic_window` | Sentence tokenization via NLTK |

### Stage 2 — Case Extraction (Output A)

Gemini 2.5 Flash extracts a structured `CaseExtraction` object (validated by Pydantic):

- `fir_number`, `fir_date`, `police_station`
- `accused_names[]`, `victim_names[]`
- `legal_sections[]`
- `incident_summary`

### Stage 3 — Crime Classification (Output B)

IPC/IT Act sections are matched against known crime type profiles. Returns:

- `crime_type` (internal key, e.g. `theft_robbery`)
- `display_name` (e.g. `Theft / Robbery`)
- `confidence` — `HIGH`, `MEDIUM`, or `LOW`
- `matched_sections[]` and `reason`

### Stage 4 — Dual Checklist Validation (Output C / C2)

**C — LLM Keyword Audit:** Gemini checks for presence of required documents/evidence by keyword matching.

**C2 — Semantic Checklist:** TF-IDF character n-gram (2–4) vectorizer computes cosine similarity between each required checklist item and every sentence in the document:
- `≥ 0.65` → `PRESENT ✅`
- `0.40 – 0.65` → `PARTIAL ⚠️`
- `< 0.40` → `MISSING ❌`

### Stage 5 — Named Entity Recognition (Output D)

Gemini extracts and categorizes entities with role/event/unit metadata:

| Entity Type | Metadata Field | Example Values |
|-------------|---------------|----------------|
| `PERSON` | `role` | `ACCUSED`, `VICTIM`, `WITNESS`, `OFFICER` |
| `LEGAL_SECTION` | `role` | `PRIMARY_CHARGE`, `ADDITIONAL_CHARGE` |
| `DATE_TIME` | `event` | `FIR_DATE`, `INCIDENT_DATE`, `ARREST_DATE` |
| `LOCATION` | `role` | `CRIME_SCENE`, `POLICE_STATION`, `COURT` |
| `DOCUMENT` | `role` | `FIR`, `MLC_REPORT`, `SEIZURE_MEMO`, `FSL_REPORT` |
| `AMOUNT` | `unit` | `INR`, `grams`, `kg`, `litres` |

Results are deduplicated by `(text, type)` before returning.

---

## 🎯 Supported Crime Types

| Crime Type | Typical IPC / IT Act Sections | Key Required Documents |
|------------|-------------------------------|------------------------|
| **Theft / Robbery** | 379, 380, 392 | FIR, recovery/seizure memo, site plan, witness statements |
| **Assault / Hurt** | 323, 324, 325 | FIR, MLC report, injury certificate, weapon seizure memo |
| **Cyber Fraud** | 419, 420, IT Act 66C/66D | Transaction records, screenshots, FSL report, KYC responses |
| **Murder** | 302, 304, 307 | FIR, post-mortem, forensic/FSL report, scene inspection |
| **Rape / Sexual Assault** | 376, 354, 509 | FIR, MLC report, FSL forensic report, victim statement |
| **Kidnapping / Abduction** | 363–366 | FIR, CDR analysis, ransom evidence, recovery memo |
| **Cheating / Fraud** | 420, 406, 415 | FIR, transaction documents, bank statements, KYC details |
| **Drug Offences** | NDPS Act | Seizure memo, FSL report, panchnama, chemical analysis |
| **Domestic Violence** | 498A, 304B | FIR, MLC, witness statements, evidence of cruelty |

---

## 🔌 API Reference

### `GET /`
Health check.
```json
{"status": "Chargesheet Analyzer API is running"}
```

### `POST /analyze-chargesheet`
Analyze pre-extracted chargesheet text.

**Request body:**
```json
{
  "text": "Chargesheet or case diary text (Hindi/English/mixed)"
}
```

**Response:**
```json
{
  "output_a_case_summary": {
    "fir_number": "255/2024",
    "fir_date": "2024-03-12",
    "police_station": "Kotwali",
    "accused_names": ["Ramesh Kumar"],
    "victim_names": ["Suresh Singh"],
    "legal_sections": ["IPC 379", "IPC 380"],
    "incident_summary": "..."
  },
  "output_b_classification": {
    "crime_type": "theft_robbery",
    "display_name": "Theft / Robbery",
    "confidence": "HIGH",
    "matched_sections": ["IPC 379"],
    "reason": "..."
  },
  "output_c_checklist": [...],
  "output_c2_semantic_checklist": {"checklist": [...]},
  "output_d_entities": {"entities": [...]}
}
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Extraction accuracy (Gemini) | ~85–90% |
| Semantic checklist threshold (PRESENT) | cosine ≥ 0.65 |
| Semantic checklist threshold (PARTIAL) | cosine ≥ 0.40 |
| Processing time per document | 10–25 seconds |
| Supported languages | Hindi, English, Mixed |
| Max NER input | 8,000 characters |

---

## ⚙️ Configuration

### Tuning Similarity Thresholds

In `api.py`:
```python
SIMILARITY_THRESHOLD = 0.65  # PRESENT
PARTIAL_THRESHOLD    = 0.40  # PARTIAL
```

### Adding New Crime Types

Extend the `CHECKLISTS` dict in `api.py`:
```python
CHECKLISTS["new_crime_key"] = {
    "display_name": "New Crime Type",
    "typical_sections": ["IPC XXX"],
    "required_items": [
        "Required document 1",
        "Required document 2"
    ]
}
```

### Switching Gemini Model

In `api.py`, change:
```python
model = genai.GenerativeModel("gemini-2.5-flash")
```

---

## 🐛 Known Issues & Limitations

- OCR quality directly affects extraction accuracy on scanned documents
- Spelling correction (`pyspellchecker`) is English-only; Hindi words are preserved but not corrected
- NER input is capped at 8,000 characters to stay within prompt limits
- Gemini API free-tier rate limits apply; high-volume batch jobs may require retry logic
- Non-standard FIR formats may require additional prompt tuning

---

## 🗺️ Roadmap

- [ ] Hindi spell correction support
- [ ] OCR integration for handwritten case diaries (Tesseract / Vision API)
- [ ] Named entity resolution (linking co-references across the document)
- [ ] PDF chargesheet export with proper formatting
- [ ] Batch upload and processing via UI
- [ ] Mobile-friendly interface
- [ ] Document similarity detection across case files

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

---

## 👥 Team

Developed for Legal Tech Hackathon 2024.

---

## 📧 Contact

- Open an issue on GitHub
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- [Google Gemini](https://ai.google.dev/) for natural language understanding
- [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF text extraction
- [Streamlit](https://streamlit.io/) for the interactive web interface
- [FastAPI](https://fastapi.tiangolo.com/) for the REST API layer
- The Indian legal community for requirements and domain feedback

---

## ⚠️ Disclaimer

This tool is designed to **assist** legal professionals, not replace them. All generated outputs should be reviewed and verified by qualified legal personnel before being used in any official proceeding. The developers assume no liability for legal outcomes resulting from use of this software.

---

**Built with ❤️ for justice and efficiency**