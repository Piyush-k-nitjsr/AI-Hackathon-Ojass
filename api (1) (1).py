import json
import re
import unicodedata
import asyncio
import pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
from dateutil.parser import parse
from dateutil.parser import ParserError
from pydantic import BaseModel, Field
from typing import List, Optional
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor
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
    normalized_text = unicodedata.normalize('NFKC', text)
    cleaned_chars = [
        char for char in normalized_text
        if unicodedata.category(char).startswith(('L', 'N', 'P', 'S', 'Z'))
    ]
    normalized_text = "".join(cleaned_chars)
    normalized_text = re.sub(r'[\u200B-\u200F\u2028-\u202F\uFEFF]', '', normalized_text)
    return normalized_text


def normalize_dates(text):
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
    cleaned_text = text
    cleaned_text = re.sub(r'\s[a-zA-Z0-9]\s', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'(.)\1{3,}', '', cleaned_text)
    cleaned_text = re.sub(r'[\|_]{2,}', ' ', cleaned_text)
    cleaned_text = re.sub(r'\b[a-z]\b', ' ', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\b\d\b', ' ', cleaned_text)
    return cleaned_text.strip()


def normalize_whitespace(text):
    cleaned_text = re.sub(r'\s+', ' ', text)
    cleaned_text = cleaned_text.strip()
    cleaned_text = cleaned_text.replace('\r\n', '\n').replace('\r', '\n')
    return cleaned_text



def apply_adaptive_semantic_window(text):
    sentences = sent_tokenize(text)
    return sentences


def preprocess_pdf(pdf_path: str) -> str:
    print(f'\nProcessing {pdf_path}...')
    pdf_text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                pdf_text += extracted + '\n'
    print(f"  Extracted text length: {len(pdf_text)} characters")
    processed = normalize_unicode(pdf_text)
    print(f"  After Unicode normalization: {len(processed)} characters")
    processed = normalize_dates(processed)
    print(f"  After Date normalization: {len(processed)} characters")
    processed = remove_ocr_garbage(processed)
    print(f"  After OCR garbage removal: {len(processed)} characters")
    processed = normalize_whitespace(processed)
    print(f"  After Whitespace normalization: {len(processed)} characters")
    print(f"  Spelling correction skipped (not effective on Hindi text)")
    sentences = apply_adaptive_semantic_window(processed)
    print(f"  After Adaptive semantic window: {len(sentences)} sentences")
    final_text = '\n'.join(sentences)
    return final_text


# ── Schema ────────────────────────────────────────────────────────────────────

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

CONFIDENCE SCORING RULES (Stage 3A):
- Return a confidence score (0.0-1.0) for every field.
- High confidence (>0.85): value is explicitly and unambiguously stated in text.
- Medium confidence (0.5-0.85): value is stated but with some ambiguity.
- Low confidence (<0.5): value is inferred, partial, or unclear.
- Scores must NOT all be 1.0 — reflect actual certainty from the text.

OUTPUT FORMAT:
{
  "fir_number":       { "value": string or null, "confidence": float },
  "fir_date":         { "value": string or null, "confidence": float },
  "police_station":   { "value": string or null, "confidence": float },
  "accused_names":    { "value": [],             "confidence": float },
  "victim_names":     { "value": [],             "confidence": float },
  "legal_sections":   { "value": [],             "confidence": float },
  "incident_summary": { "value": string or null, "confidence": float }
}

--- FEW-SHOT EXAMPLE 1 (clear, complete input — high confidence) ---
Input:
प्रथम सूचना रिपोर्ट संख्या 123/2024, दिनांक 15/03/2024
थाना: कोतवाली नगर
अभियुक्त: राम कुमार पुत्र श्याम कुमार
पीड़ित: सीता देवी
धारा: IPC 379, IPC 411
घटना विवरण: दिनांक 14/03/2024 को रात्रि 10 बजे अभियुक्त राम कुमार ने पीड़ित सीता देवी का मोबाइल फोन चुरा लिया।

Output:
{
  "fir_number":       { "value": "123/2024",               "confidence": 0.97 },
  "fir_date":         { "value": "15/03/2024",              "confidence": 0.95 },
  "police_station":   { "value": "कोतवाली नगर",             "confidence": 0.93 },
  "accused_names":    { "value": ["राम कुमार"],              "confidence": 0.91 },
  "victim_names":     { "value": ["सीता देवी"],              "confidence": 0.90 },
  "legal_sections":   { "value": ["IPC 379", "IPC 411"],    "confidence": 0.95 },
  "incident_summary": { "value": "दिनांक 14/03/2024 को रात्रि 10 बजे अभियुक्त राम कुमार ने पीड़ित सीता देवी का मोबाइल फोन चुरा लिया।", "confidence": 0.92 }
}

--- FEW-SHOT EXAMPLE 2 (ambiguous/incomplete input — low confidence) ---
Input:
काण्ड संख्या 456/2024
किसी अज्ञात व्यक्ति द्वारा मारपीट की गई। पीड़ित का नाम अस्पष्ट है।
धारा लागू की जा सकती है।

Output:
{
  "fir_number":       { "value": null,   "confidence": 0.10 },
  "fir_date":         { "value": null,   "confidence": 0.10 },
  "police_station":   { "value": null,   "confidence": 0.10 },
  "accused_names":    { "value": [],     "confidence": 0.10 },
  "victim_names":     { "value": [],     "confidence": 0.20 },
  "legal_sections":   { "value": [],     "confidence": 0.15 },
  "incident_summary": { "value": "अज्ञात व्यक्ति द्वारा मारपीट की घटना। विवरण अपूर्ण।", "confidence": 0.35 }
}
"""


CHECKLIST_PROMPT = """
You are auditing an Indian police chargesheet.
For EACH required item below, check if it is present in the case text.
Return a JSON array ONLY. No markdown, no explanation.
Each element must have four fields: item, status, detail, confidence.
- status: one of PRESENT, MISSING, PARTIAL
- confidence (Stage 3A): float 0.0-1.0 — how certain you are about this status.
  Use low confidence when evidence is indirect or ambiguous.
  Use high confidence only when item is clearly and explicitly present or absent.
  Scores must NOT all be 1.0.

--- FEW-SHOT EXAMPLE 1 ---
Required items: ["FIR number and date", "Witness statements"]
Case text: "FIR No. 55/2024 dated 01/04/2024 registered at PS Kotwali. No witness was present at the scene."
Output:
[
  {"item": "FIR number and date", "status": "PRESENT", "detail": "FIR No. 55/2024 dated 01/04/2024 explicitly stated.", "confidence": 0.96},
  {"item": "Witness statements",  "status": "MISSING", "detail": "Text explicitly states no witness was present.", "confidence": 0.91}
]

--- FEW-SHOT EXAMPLE 2 (partial/ambiguous) ---
Required items: ["Medical Legal Case (MLC) report", "Arrest memo (if arrested)"]
Case text: "पीड़ित को अस्पताल भेजा गया। अभियुक्त को हिरासत में लिया गया।"
Output:
[
  {"item": "Medical Legal Case (MLC) report", "status": "PARTIAL", "detail": "पीड़ित को अस्पताल भेजा गया — MLC का उल्लेख नहीं, केवल अस्पताल जाने का जिक्र।", "confidence": 0.55},
  {"item": "Arrest memo (if arrested)",       "status": "PARTIAL", "detail": "हिरासत में लिया गया — लेकिन औपचारिक गिरफ्तारी मेमो का उल्लेख नहीं।",      "confidence": 0.50}
]

Required items:
ITEMS_PLACEHOLDER

Case text:
TEXT_PLACEHOLDER
"""



def _compute_classification_confidence(scores: dict, best_match: str) -> float:
    """
    Meaningful numeric confidence for crime classification (0.0-1.0).
    Combines section coverage ratio and separation margin from second-best.
    """
    if not scores:
        return 0.0
    total_typical = len(CHECKLISTS[best_match].get("typical_sections", []))
    match_count   = scores[best_match]
    coverage      = match_count / total_typical if total_typical > 0 else 0.0
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) >= 2:
        margin = (sorted_scores[0] - sorted_scores[1]) / max(sorted_scores[0], 1)
    else:
        margin = 1.0
    confidence = round((coverage * 0.6) + (margin * 0.4), 3)
    return min(confidence, 1.0)


def _add_checklist_confidence_from_semantic(
    llm_checklist: list,
    sem_checklist: list
) -> list:
    """
    Stage 3A: Blends LLM checklist status with semantic similarity score
    to produce a meaningful, non-binary per-item confidence.
    Formula: 60% LLM status base + 40% semantic similarity score.
    """
    sem_map     = {item["item"]: item["similarity_score"] for item in sem_checklist}
    status_base = {"PRESENT": 0.75, "PARTIAL": 0.45, "MISSING": 0.10}

    for item in llm_checklist:
        existing_conf = item.get("confidence")
        if existing_conf is None or existing_conf in (0.75, 0.45, 0.10):
            base      = status_base.get(item.get("status", "MISSING"), 0.10)
            sim_score = sem_map.get(item.get("item", ""), 0.0)
            item["confidence"] = round(min((base * 0.6) + (sim_score * 0.4), 1.0), 3)

    return llm_checklist



def extract_case_details(text: str) -> tuple:
    """
    Stage 3B: Uses few-shot prompt for extraction.
    Stage 3A: Returns per-field confidence scores alongside extracted values.
    Returns (CaseExtraction, summary_confidence_dict).
    """
    try:
        prompt = f"{SYSTEM_PROMPT}\n\nExtract from this chargesheet:\n\n{text}\n\nReturn ONLY JSON."
        response = model.generate_content(prompt, generation_config={"temperature": 0})
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].replace("json", "").strip()

        parsed = json.loads(raw)

        def get_val(key):
            entry = parsed.get(key, {})
            if isinstance(entry, dict):
                return entry.get("value")
            return entry

        def get_conf(key):
            entry = parsed.get(key, {})
            if isinstance(entry, dict):
                return round(float(entry.get("confidence", 0.5)), 3)
            return 0.5

        extraction = CaseExtraction(
            fir_number       = get_val("fir_number"),
            fir_date         = get_val("fir_date"),
            police_station   = get_val("police_station"),
            accused_names    = get_val("accused_names") or [],
            victim_names     = get_val("victim_names") or [],
            legal_sections   = get_val("legal_sections") or [],
            incident_summary = get_val("incident_summary"),
        )

        summary_confidence = {
            "fir_number":       get_conf("fir_number"),
            "fir_date":         get_conf("fir_date"),
            "police_station":   get_conf("police_station"),
            "accused_names":    get_conf("accused_names"),
            "victim_names":     get_conf("victim_names"),
            "legal_sections":   get_conf("legal_sections"),
            "incident_summary": get_conf("incident_summary"),
        }

        return extraction, summary_confidence

    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return CaseExtraction(), {k: 0.0 for k in [
            "fir_number", "fir_date", "police_station",
            "accused_names", "victim_names", "legal_sections", "incident_summary"
        ]}


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
            "classification_confidence": 0.0,
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
            "classification_confidence": 0.0,
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

    classification_confidence = _compute_classification_confidence(scores, best_match)

    return {
        "crime_type":                best_match,
        "display_name":              CHECKLISTS[best_match]["display_name"],
        "confidence":                confidence,
        "classification_confidence": classification_confidence,
        "matched_sections":          matched_map[best_match],
        "reason":                    f"{match_count} of {total} typical sections matched"
    }


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
        status_base = {"PRESENT": 0.75, "PARTIAL": 0.45, "MISSING": 0.10}
        for item in results:
            item["label"] = emoji_map.get(item["status"], "❓")
            # Fallback confidence if LLM omits it
            if "confidence" not in item:
                item["confidence"] = status_base.get(item.get("status", "MISSING"), 0.10)
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
    try:
        prompt = f"{NER_PROMPT}\n\nChargesheet text:\n\n{text[:8000]}\n\nReturn ONLY JSON."
        response = model.generate_content(prompt, generation_config={"temperature": 0})
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].replace("json", "").strip()
        data = json.loads(raw)
        entities = data.get("entities", [])
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


SIMILARITY_THRESHOLD = 0.65
PARTIAL_THRESHOLD    = 0.40

def _get_embeddings_tfidf(required_items: List[str], sentences: List[str]):
    corpus = required_items + sentences
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    items_matrix     = tfidf_matrix[:len(required_items)]
    sentences_matrix = tfidf_matrix[len(required_items):]
    return items_matrix, sentences_matrix

def run_semantic_checklist(text: str, crime_key: str) -> list:
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
            scores       = sim_matrix[idx]
            best_idx     = int(np.argmax(scores))
            best_score   = float(scores[best_idx])
            matched_text = sentences[best_idx]
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
    extraction, summary_confidence = extract_case_details(text)
    extraction = normalize(extraction)
    extraction.fir_number = validate_fir_number(extraction.fir_number, text)
    classification = classify_crime(extraction.legal_sections)
    crime_key = classification["crime_type"]

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_checklist     = executor.submit(run_checklist_audit, text, crime_key)
        future_sem_checklist = executor.submit(run_semantic_checklist, text, crime_key)
        future_ner           = executor.submit(run_ner, text)
        checklist     = future_checklist.result()
        sem_checklist = future_sem_checklist.result()
        entities      = future_ner.result()

    checklist = _add_checklist_confidence_from_semantic(checklist, sem_checklist)

    return {
        "output_a_case_summary": {
            **extraction.model_dump(),
            "field_confidence": summary_confidence
        },
        "output_b_classification":      classification,
        "output_c_checklist":           checklist,
        "output_c2_semantic_checklist": {"checklist": sem_checklist},
        "output_d_entities":            {"entities": entities}
    }



def process_pdf_and_analyze(pdf_path: str, save_txt_path: Optional[str] = None) -> dict:
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
async def analyze(request: AnalyzeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, analyze_chargesheet, request.text)
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