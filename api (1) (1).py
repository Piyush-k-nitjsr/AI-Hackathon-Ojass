import json
import re
import unicodedata
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from dateutil.parser import parse
from dateutil.parser import ParserError
from spellchecker import SpellChecker
from pydantic import BaseModel, Field
from typing import List, Optional
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download('punkt')
nltk.download('punkt_tab')


load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set")

genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")


def normalize_unicode(text):
    """Normalizes Unicode characters and removes non-printable/control characters."""
    normalized_text = unicodedata.normalize('NFKC', text)
    cleaned_chars = [
        char for char in normalized_text
        if unicodedata.category(char).startswith(('L', 'N', 'P', 'S', 'Z'))
    ]
    normalized_text = "".join(cleaned_chars)
    normalized_text = re.sub(r'[\u200B-\u200F\u2028-\u202F\uFEFF]', '', normalized_text)
    return normalized_text


def normalize_dates(text):
    """Identifies and normalizes various date formats in the text to 'YYYY-MM-DD'."""
    date_pattern = re.compile(
        r'\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|' +
        r'\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4}|' +
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},?\s\d{4})\b',
        re.IGNORECASE
    )

    def replace_date(match):
        date_str = match.group(0)
        try:
            parsed_date = parse(date_str, fuzzy=True)
            return parsed_date.strftime('%Y-%m-%d')
        except ParserError:
            return date_str

    return date_pattern.sub(replace_date, text)


def remove_ocr_garbage(text):
    """Removes common OCR garbage such as isolated characters, repeated characters, and non-alphanumeric sequences."""
    cleaned_text = text
    cleaned_text = re.sub(r'\s[a-zA-Z0-9]\s', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'(.)\1{3,}', '', cleaned_text)
    cleaned_text = re.sub(r'[\|_]{2,}', ' ', cleaned_text)
    cleaned_text = re.sub(r'\b[a-z]\b', ' ', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\b\d\b', ' ', cleaned_text)
    return cleaned_text.strip()


def normalize_whitespace(text):
    """Normalizes whitespace in the text."""
    cleaned_text = re.sub(r'\s+', ' ', text)
    cleaned_text = cleaned_text.strip()
    cleaned_text = cleaned_text.replace('\r\n', '\n').replace('\r', '\n')
    return cleaned_text


def correct_spelling(text, language='en'):
    """Corrects spelling mistakes in the text using a spell checker."""
    spell = SpellChecker(language=language)
    corrected_words = []
    words = re.findall(r'\b\w+\b|\S', text)
    for word in words:
        if re.match(r'^\w+$', word):
            corrected_word = spell.correction(word)
            if corrected_word is not None:
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    return ' '.join(corrected_words)


def apply_adaptive_semantic_window(text):
    """Applies an adaptive semantic window by splitting text into sentences."""
    sentences = sent_tokenize(text)
    return sentences


def preprocess_pdf(pdf_path: str) -> str:
    """
    Full preprocessing pipeline for a single PDF.
    Extracts text and runs all cleaning steps.
    Returns the final cleaned text as a single string.
    """
    print(f'\nProcessing {pdf_path}...')

    # 1. Extract text
    pdf_text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                pdf_text += extracted + '\n'
    print(f"  Extracted text length: {len(pdf_text)} characters")

    # 2. Unicode normalization
    processed = normalize_unicode(pdf_text)
    print(f"  After Unicode normalization: {len(processed)} characters")

    # 3. Date normalization
    processed = normalize_dates(processed)
    print(f"  After Date normalization: {len(processed)} characters")

    # 4. OCR garbage removal
    processed = remove_ocr_garbage(processed)
    print(f"  After OCR garbage removal: {len(processed)} characters")

    # 5. Whitespace normalization
    processed = normalize_whitespace(processed)
    print(f"  After Whitespace normalization: {len(processed)} characters")

    # 6. Spelling correction
    processed = correct_spelling(processed)
    print(f"  After Spelling correction: {len(processed)} characters")

    # 7. Adaptive semantic window (sentence tokenization)
    sentences = apply_adaptive_semantic_window(processed)
    print(f"  After Adaptive semantic window: {len(sentences)} sentences")

    # Join sentences back with newline
    final_text = '\n'.join(sentences)
    return final_text




class CaseExtraction(BaseModel):
    fir_number:       Optional[str] = Field(None)
    fir_date:         Optional[str] = Field(None)
    police_station:   Optional[str] = Field(None)
    accused_names:    List[str]     = Field(default_factory=list)
    victim_names:     List[str]     = Field(default_factory=list)
    legal_sections:   List[str]     = Field(default_factory=list)
    incident_summary: Optional[str] = Field(None)

class AnalyzeRequest(BaseModel):
    text: str


CHECKLISTS = {
  "theft_robbery": {
    "display_name": "Theft / Robbery",
    "typical_sections": ["IPC 379", "IPC 380", "IPC 392"],
    "required_items": [
      "FIR number and date",
      "Police station name",
      "Place and time of occurrence",
      "Details of complainant and accused",
      "Description of stolen property and value",
      "Recovery/seizure memo of property",
      "Witness statements",
      "Site plan / spot inspection memo",
      "Arrest memo (if arrested)",
      "Chain of custody of recovered items"
    ]
  },
  "assault_hurt": {
    "display_name": "Assault / Hurt",
    "typical_sections": ["IPC 323", "IPC 324", "IPC 325"],
    "required_items": [
      "FIR details (number, date, PS)",
      "Victim and accused details",
      "Medical Legal Case (MLC) report",
      "Injury certificate with nature and number of injuries",
      "Photographs of injuries (if available)",
      "Description and seizure memo of weapon (if recovered)",
      "Doctor's opinion on nature of injuries",
      "Witness statements about assault"
    ]
  },
  "cyber_fraud": {
    "display_name": "Cyber Fraud / Online Cheating",
    "typical_sections": ["IPC 419", "IPC 420", "IT Act 66C", "IT Act 66D"],
    "required_items": [
      "Details of platform and mode of fraud",
      "Transaction records (bank/UPI statements)",
      "Screenshots or logs of chats/emails/calls",
      "Seizure memo of digital devices",
      "Requests to banks/companies for KYC and logs",
      "Responses from banks/companies",
      "Cyber-forensic / FSL report (if available)",
      "Chain of custody of digital evidence",
      "Linking of digital identifiers (IP/IMEI) to accused"
    ]
  },
  "ndps": {
    "display_name": "NDPS (Drugs / Narcotics)",
    "typical_sections": ["NDPS 20", "NDPS 21", "NDPS 22"],
    "required_items": [
      "Exact place, date and time of seizure",
      "Description of substance and quantity (small/intermediate/commercial)",
      "Search memo and seizure memo with seal details",
      "Compliance note for relevant NDPS provisions (e.g. Sections 42, 50)",
      "Weighment panchnama",
      "Sample drawing memo and sample details",
      "Chain of custody record (seizure to malkhana to FSL)",
      "Malkhana register entry",
      "FSL report",
      "Statements of independent witnesses"
    ]
  }
}


SYSTEM_PROMPT = """
You are a legal extraction engine for Indian police chargesheets written in Hindi.
RULES:
1. Return ONLY raw JSON — no markdown, no explanation, no ```json blocks.
2. Never guess. If something is not clearly stated, return null or [].
3. Names must be copied exactly as written in the document.
4. Legal sections must be exact — e.g. "IPC 379", "NDPS 20", "IT Act 66C".
5. Incident summary: factual only, max 120 words, no opinions.
FIR NUMBER RULE:
- Only extract fir_number if text explicitly says FIR or प्रथम सूचना रिपोर्ट.
- If unsure, return null.
OUTPUT FORMAT:
{
  "fir_number": string or null,
  "fir_date": string or null,
  "police_station": string or null,
  "accused_names": [],
  "victim_names": [],
  "legal_sections": [],
  "incident_summary": string or null
}
"""

def extract_case_details(text: str) -> CaseExtraction:
    try:
        prompt = f"{SYSTEM_PROMPT}\n\nExtract from this chargesheet:\n\n{text}\n\nReturn ONLY JSON."
        response = model.generate_content(prompt, generation_config={"temperature": 0})
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].replace("json", "").strip()
        return CaseExtraction.model_validate_json(raw)
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return CaseExtraction()


FIR_POSITIVE = [
    r'FIR\s*(No\.?|Number|नंबर)?',
    r'एफआईआर',
    r'प्रथम\s+सूचना\s+रिपोर्ट',
]

FIR_NEGATIVE = [
    r'काण्ड\s*संख्या',
    r'अपराध\s*क्रमांक',
    r'Case\s*No',
    r'Charge\s*Sheet\s*No',
    r'C\.?S\.?\s*No',
    r'GD\s*No',
]

def _context_window(text: str, match_start: int, match_end: int, window: int = 150) -> str:
    return text[max(0, match_start - window) : match_end + window]

def validate_fir_number(fir_number: Optional[str], text: str) -> Optional[str]:
    if not fir_number:
        return None
    for pattern in FIR_NEGATIVE:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            ctx = _context_window(text, m.start(), m.end())
            if fir_number in ctx:
                return None
    for pattern in FIR_POSITIVE:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            ctx = _context_window(text, m.start(), m.end())
            if fir_number in ctx:
                return fir_number
    return None


HINDI_MONTHS = {
    "जनवरी": "01", "फरवरी": "02", "मार्च": "03",
    "अप्रैल": "04", "मई": "05", "जून": "06",
    "जुलाई": "07", "अगस्त": "08", "सितंबर": "09",
    "अक्टूबर": "10", "नवंबर": "11", "दिसंबर": "12"
}

def normalize_date(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    date_str = date_str.strip()
    for hindi_month, month_num in HINDI_MONTHS.items():
        if hindi_month in date_str:
            match = re.search(r'(\d{1,2})\s+' + hindi_month + r'\s+(\d{4})', date_str)
            if match:
                return f"{match.group(1).zfill(2)}/{month_num}/{match.group(2)}"
    match = re.search(r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})', date_str)
    if match:
        return f"{match.group(1).zfill(2)}/{match.group(2).zfill(2)}/{match.group(3)}"
    match = re.search(r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2})', date_str)
    if match:
        return f"{match.group(1).zfill(2)}/{match.group(2).zfill(2)}/20{match.group(3)}"
    return date_str

def normalize_section(section: str) -> str:
    section = section.strip()
    section = re.sub(r'धारा\s*', '', section)
    section = re.sub(r'(IPC|NDPS|CrPC)(\d)', r'\1 \2', section, flags=re.IGNORECASE)
    section = re.sub(r'IT\s*Act\s*(\w+)', r'IT Act \1', section, flags=re.IGNORECASE)
    section = re.sub(r'\bipc\b', 'IPC', section, flags=re.IGNORECASE)
    section = re.sub(r'\bndps\b', 'NDPS', section, flags=re.IGNORECASE)
    section = re.sub(r'\bcrpc\b', 'CrPC', section, flags=re.IGNORECASE)
    return section.strip()

def normalize_sections_list(sections: List[str]) -> List[str]:
    if not sections:
        return []
    seen = set()
    result = []
    for s in [normalize_section(s) for s in sections]:
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return result

NOISE_PATTERNS = [
    r'\bS/O\b', r'\bW/O\b', r'\bD/O\b', r'\bR/O\b',
    r'\d+\s*वर्ष', r'\d+\s*years?\s*old',
    r'\bआयु\b', r'\bAge\b',
]

def clean_name(name: str) -> str:
    if not name:
        return ""
    for pattern in NOISE_PATTERNS:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)
    return re.sub(r'\s{2,}', ' ', name).strip().rstrip('.,;:')

def clean_names_list(names: List[str]) -> List[str]:
    if not names:
        return []
    seen = set()
    result = []
    for n in [clean_name(n) for n in names]:
        if n and n not in seen:
            seen.add(n)
            result.append(n)
    return result

def normalize_fir_number(fir: Optional[str]) -> Optional[str]:
    if not fir:
        return None
    fir = re.sub(r'^(FIR|No\.?|Number|नंबर|संख्या)\s*[:\-]?\s*', '', fir, flags=re.IGNORECASE)
    return fir.strip().rstrip('.,;') or None

def normalize(extraction: CaseExtraction) -> CaseExtraction:
    extraction.fir_number     = normalize_fir_number(extraction.fir_number)
    extraction.fir_date       = normalize_date(extraction.fir_date)
    extraction.accused_names  = clean_names_list(extraction.accused_names)
    extraction.victim_names   = clean_names_list(extraction.victim_names)
    extraction.legal_sections = normalize_sections_list(extraction.legal_sections)
    if extraction.incident_summary:
        extraction.incident_summary = re.sub(r'\s{2,}', ' ', extraction.incident_summary).strip()
    return extraction

# ── Classification ────────────────────────────────────────────────────────────

def match_section(extracted: str, typical: str) -> bool:
    extracted = extracted.upper().strip()
    typical   = typical.upper().strip()
    return extracted == typical or extracted in typical or typical in extracted

def classify_crime(legal_sections: List[str]) -> dict:
    if not legal_sections:
        return {
            "crime_type": "UNKNOWN",
            "display_name": "Unknown",
            "reason": "No legal sections extracted",
            "confidence": "LOW",
            "matched_sections": []
        }
    scores = {}
    matched_map = {}
    for crime_key, crime_data in CHECKLISTS.items():
        typical_sections = crime_data.get("typical_sections", [])
        matches = []
        for extracted_sec in legal_sections:
            for typical_sec in typical_sections:
                if match_section(extracted_sec, typical_sec):
                    matches.append(extracted_sec)
                    break
        if matches:
            scores[crime_key]      = len(matches)
            matched_map[crime_key] = matches
    if not scores:
        return {
            "crime_type": "UNKNOWN",
            "display_name": "Unknown",
            "reason": f"No matching sections found for: {legal_sections}",
            "confidence": "LOW",
            "matched_sections": []
        }
    best_match  = max(scores, key=scores.get)
    match_count = scores[best_match]
    total       = len(CHECKLISTS[best_match].get("typical_sections", []))
    if match_count >= 2:
        confidence = "HIGH"
    elif match_count == 1:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    return {
        "crime_type":       best_match,
        "display_name":     CHECKLISTS[best_match]["display_name"],
        "confidence":       confidence,
        "matched_sections": matched_map[best_match],
        "reason":           f"{match_count} of {total} typical sections matched"
    }


CHECKLIST_PROMPT = """
You are auditing an Indian police chargesheet.
For EACH required item below, check if it is present in the case text.
Return a JSON array ONLY. No markdown, no explanation.
Each element must have three fields: item, status, detail.
status must be one of: PRESENT, MISSING, PARTIAL.
PRESENT means clearly found in the text.
MISSING means not found at all.
PARTIAL means mentioned but incomplete.

Required items:
ITEMS_PLACEHOLDER

Case text:
TEXT_PLACEHOLDER
"""

def run_checklist_audit(text: str, crime_key: str) -> list:
    if crime_key == "UNKNOWN" or crime_key not in CHECKLISTS:
        return []
    required_items = CHECKLISTS[crime_key]["required_items"]
    items_str = "\n".join(f"- {item}" for item in required_items)
    prompt = CHECKLIST_PROMPT.replace("ITEMS_PLACEHOLDER", items_str)
    prompt = prompt.replace("TEXT_PLACEHOLDER", text[:6000])
    try:
        response = model.generate_content(prompt, generation_config={"temperature": 0})
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].replace("json", "").strip()
        results = json.loads(raw)
        emoji_map = {"PRESENT": "✅", "MISSING": "❌", "PARTIAL": "⚠️"}
        for item in results:
            item["label"] = emoji_map.get(item["status"], "❓")
        return results
    except Exception as e:
        print(f"[ERROR] Checklist audit failed: {e}")
        return []


NER_PROMPT = """
You are a Named Entity Recognition (NER) engine for Indian police chargesheets written in Hindi and English.

Extract ALL entities from the text and return a JSON object with a single key "entities" containing an array.

ENTITY TYPES AND RULES:

1. PERSON — Any human name (accused, victim, witness, officer, judge)
   - Required field: "role" → one of: ACCUSED, VICTIM, WITNESS, OFFICER, UNKNOWN
   - Example: {"text": "राम कुमार", "type": "PERSON", "role": "ACCUSED"}

2. LEGAL_SECTION — Any law section reference
   - No extra fields needed
   - Example: {"text": "IPC 379", "type": "LEGAL_SECTION"}

3. DATE_TIME — Any date or time reference
   - Required field: "event" → one of: FIR_DATE, INCIDENT_DATE, ARREST_DATE, HEARING_DATE, SEIZURE_DATE, UNKNOWN
   - Example: {"text": "12/03/2024", "type": "DATE_TIME", "event": "FIR_DATE"}

4. LOCATION — Any place, address, police station, city, village
   - Required field: "role" → one of: CRIME_SCENE, POLICE_STATION, RESIDENCE, COURT, UNKNOWN
   - Example: {"text": "थाना कोतवाली", "type": "LOCATION", "role": "POLICE_STATION"}

5. DOCUMENT — Any official document name
   - Required field: "role" → one of: FIR, MLC_REPORT, SEIZURE_MEMO, FSL_REPORT, ARREST_MEMO, PANCHNAMA, CHARGE_SHEET, OTHER
   - Example: {"text": "जब्ती मेमो", "type": "DOCUMENT", "role": "SEIZURE_MEMO"}

6. AMOUNT — Any monetary value, weight, or quantity of seized items
   - Required field: "unit" → e.g. "INR", "grams", "kg", "litres", "units", "UNKNOWN"
   - Example: {"text": "5000 रुपये", "type": "AMOUNT", "unit": "INR"}

STRICT RULES:
- Return ONLY raw JSON. No markdown, no explanation, no ```json blocks.
- Extract every entity you find, do not skip any.
- If a field value is unclear, use "UNKNOWN".
- Never invent entities not present in the text.
- Preserve original text exactly as it appears in the document.

OUTPUT FORMAT:
{"entities": [...]}
"""

def run_ner(text: str) -> list:
    """Runs the NER layer on chargesheet text and returns a list of entities."""
    try:
        prompt = f"{NER_PROMPT}\n\nChargesheet text:\n\n{text[:8000]}\n\nReturn ONLY JSON."
        response = model.generate_content(prompt, generation_config={"temperature": 0})
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].replace("json", "").strip()
        data = json.loads(raw)
        entities = data.get("entities", [])

        # Post-process: deduplicate by (text, type)
        seen = set()
        deduped = []
        for entity in entities:
            key = (entity.get("text", "").strip(), entity.get("type", ""))
            if key not in seen:
                seen.add(key)
                deduped.append(entity)

        return deduped
    except Exception as e:
        print(f"[ERROR] NER failed: {e}")
        return []


SIMILARITY_THRESHOLD = 0.65  # Minimum cosine similarity to count as PRESENT
PARTIAL_THRESHOLD    = 0.40  # Between this and SIMILARITY_THRESHOLD → PARTIAL

def _get_embeddings_tfidf(required_items: List[str], sentences: List[str]):
    """
    Fits a TF-IDF vectorizer on required_items + sentences together so both
    share the same vocabulary, then returns separate matrices.
    """
    corpus = required_items + sentences
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",   # character n-grams — handles Hindi + mixed text better
        ngram_range=(2, 4),
        min_df=1,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    items_matrix     = tfidf_matrix[:len(required_items)]
    sentences_matrix = tfidf_matrix[len(required_items):]
    return items_matrix, sentences_matrix

def run_semantic_checklist(text: str, crime_key: str) -> list:
    """
    Augments the keyword checklist with TF-IDF cosine similarity scoring.
    For each required checklist item:
      - Finds the most similar sentence in the chargesheet
      - Assigns PRESENT / PARTIAL / MISSING based on similarity score
      - Returns the top matching sentence for PRESENT/PARTIAL items
    Output format matches the minimum spec:
      { item, status, similarity_score, matched_text, label }
    """
    if crime_key == "UNKNOWN" or crime_key not in CHECKLISTS:
        return []

    required_items = CHECKLISTS[crime_key]["required_items"]

    
    sentences = [s.strip() for s in text.split("\n") if s.strip()]
    
    if len(sentences) < 3:
        sentences = [s.strip() for s in re.split(r'[।.!?]', text) if s.strip()]
    if not sentences:
        return []

    try:
        items_matrix, sentences_matrix = _get_embeddings_tfidf(required_items, sentences)

        sim_matrix = cosine_similarity(items_matrix, sentences_matrix)

        emoji_map = {"PRESENT": "✅", "MISSING": "❌", "PARTIAL": "⚠️"}
        results = []

        for idx, item in enumerate(required_items):
            scores        = sim_matrix[idx]          
            best_idx      = int(np.argmax(scores))
            best_score    = float(scores[best_idx])
            matched_text  = sentences[best_idx]

            if best_score >= SIMILARITY_THRESHOLD:
                status = "PRESENT"
            elif best_score >= PARTIAL_THRESHOLD:
                status = "PARTIAL"
            else:
                status       = "MISSING"
                matched_text = ""         

            results.append({
                "item":             item,
                "status":           status,
                "similarity_score": round(best_score, 4),
                "matched_text":     matched_text,
                "label":            emoji_map.get(status, "❓")
            })

        return results

    except Exception as e:
        print(f"[ERROR] Semantic checklist failed: {e}")
        return []


def analyze_chargesheet(text: str) -> dict:
    extraction = extract_case_details(text)
    extraction = normalize(extraction)
    extraction.fir_number = validate_fir_number(extraction.fir_number, text)
    classification = classify_crime(extraction.legal_sections)
    crime_key  = classification["crime_type"]
    checklist  = run_checklist_audit(text, crime_key)       # Stage 1 — LLM keyword audit (unchanged)
    sem_checklist = run_semantic_checklist(text, crime_key) # Stage 2B — Semantic similarity
    entities   = run_ner(text)                              # Stage 2A — NER
    return {
        "output_a_case_summary":      extraction.model_dump(),
        "output_b_classification":    classification,
        "output_c_checklist":         checklist,
        "output_c2_semantic_checklist": {"checklist": sem_checklist},
        "output_d_entities":          {"entities": entities}
    }



def process_pdf_and_analyze(pdf_path: str, save_txt_path: Optional[str] = None) -> dict:
    """
    End-to-end pipeline:
      1. Preprocess the PDF (clean text)
      2. Optionally save cleaned text to a .txt file
      3. Run chargesheet analysis on the cleaned text
      Returns the full analysis result dict.
    """
    cleaned_text = preprocess_pdf(pdf_path)

    if save_txt_path:
        with open(save_txt_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        print(f"  Cleaned text saved to '{save_txt_path}'")

    print(f"  Running chargesheet analysis...")
    result = analyze_chargesheet(cleaned_text)
    print(f"  Analysis complete.")
    return result


app = FastAPI(title="Chargesheet Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Chargesheet Analyzer API is running"}

@app.post("/analyze-chargesheet")
def analyze(request: AnalyzeRequest):
    """Accepts pre-extracted text and runs the analysis pipeline."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        result = analyze_chargesheet(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    pdf_jobs = [
        {"pdf": "/content/Case Diary- 255.pdf", "txt": "1.txt"},
        {"pdf": "/content/Case Diary-238.pdf",  "txt": "2.txt"},
        {"pdf": "/content/Case Diary-456.pdf",  "txt": "3.txt"},
    ]

    all_results = {}
    for job in pdf_jobs:
        pdf_path = job["pdf"]
        txt_path = job.get("txt")
        result = process_pdf_and_analyze(pdf_path, save_txt_path=txt_path)
        all_results[pdf_path] = result
        print(f"\n=== Result for {pdf_path} ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    with open("all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("\nAll results saved to 'all_results.json'")