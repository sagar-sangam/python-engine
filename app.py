import os
import json
import time
import uuid
import traceback

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from google import genai
from google.genai import types

# =========================================================
# LOAD ENVIRONMENT VARIABLES (For GitHub Security)
# =========================================================
# Ye line local system me .env file padhegi aur Render par environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. "
        "Please add it to your Render Environment Variables or local .env file."
    )

# =========================================================
# FLASK APP
# =========================================================
app = Flask(__name__)
UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================================================
# GEMINI CLIENT
# =========================================================
client = genai.Client(api_key=GEMINI_API_KEY)

# Correct model for fast & accurate document extraction
MODEL_NAME = "gemini-1.5-flash"

# =========================================================
# JSON RESPONSE SCHEMA
# =========================================================
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "log_entries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                    "start_km": {"type": "number"},
                    "close_km": {"type": "number"},
                    "total_km": {"type": "number"},
                    "start_time": {"type": "string"},
                    "close_time": {"type": "string"},
                    "total_hrs": {"type": "number"},
                    "extra_km": {"type": "number"},
                    "extra_hrs": {"type": "number"},
                    "toll": {"type": "number"},
                    "remarks": {"type": "string"}
                },
                "required": [
                    "date", "start_km", "close_km", "total_km", "start_time", 
                    "close_time", "total_hrs", "extra_km", "extra_hrs", "toll", "remarks"
                ]
            }
        }
    },
    "required": ["log_entries"]
}

# =========================================================
# EXTRACTION PROMPT (Strict DD-MM-YYYY format)
# =========================================================
EXTRACTION_PROMPT = """
You are an expert vehicle logsheet data extraction system.
The attached PDF is an Indian vehicle logsheet.
Your task is to extract EVERY SINGLE LOG ENTRY / TABLE ROW from the COMPLETE PDF.

=========================================================
VERY IMPORTANT
=========================================================
PROCESS THE ENTIRE PDF. Read ALL pages.
Do NOT only read the first page.
Every physical row in the vehicle logsheet must become exactly ONE object inside "log_entries".

DO NOT:
- skip rows
- summarize rows
- combine rows
- invent values
- ignore later pages
- ignore small table rows
- ignore scanned pages

If the same date appears in multiple rows, create multiple separate objects.

=========================================================
TABLE EXTRACTION
=========================================================
Carefully understand the table columns and their alignment.
Make sure each value belongs to the correct row.
Do not move values from one row to another.
If a table continues on another page, continue extracting the rows from that page.

=========================================================
NUMBER FIELDS
=========================================================
These fields MUST be numbers: start_km, close_km, total_km, total_hrs, extra_km, extra_hrs, toll
Remove commas and other formatting characters.
Examples: "1,250" -> 1250 | "2,500.50" -> 2500.50
If a numeric field is genuinely missing: 0

=========================================================
TIME FIELDS
=========================================================
Fields: start_time, close_time
Use: HH:MM:SS
Examples: 09:30 -> 09:30:00 | 5:45 PM -> 17:45:00
If genuinely missing: 00:00:00

=========================================================
DATE
=========================================================
Return date strictly as:
DD-MM-YYYY

Example: 05-09-2026
Only convert the date when the actual date is clear.
NEVER invent a date.

=========================================================
TOTALS & REMARKS
=========================================================
If the document already contains total_km or total_hrs, use the printed value.
Extract actual remarks. If there is no remark: ""

=========================================================
MISSING VALUES
=========================================================
If information is genuinely missing: Number -> 0, Time -> "00:00:00", Text -> ""
Do not guess. Return ONLY valid JSON matching the supplied schema.
"""

# =========================================================
# RETRY CONFIGURATION
# =========================================================
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 2
MAX_RETRY_DELAY = 10

def is_retryable_error(error):
    error_text = str(error).upper()
    retryable_errors = ["503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL", "502", "BAD_GATEWAY", "504", "DEADLINE_EXCEEDED", "TIMEOUT"]
    return any(item in error_text for item in retryable_errors)

def generate_with_retry(uploaded_file):
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"🧠 Gemini request {attempt}/{MAX_RETRIES}")
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[uploaded_file, EXTRACTION_PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    temperature=0.1
                )
            )
            print("✅ Gemini request successful")
            return response
        except Exception as error:
            last_error = error
            print(f"⚠️ Gemini request failed (attempt {attempt}/{MAX_RETRIES})")
            if not is_retryable_error(error):
                raise
            if attempt >= MAX_RETRIES:
                print("❌ Maximum Gemini retries reached.")
                raise
            delay = min(INITIAL_RETRY_DELAY * (2 ** (attempt - 1)), MAX_RETRY_DELAY)
            print(f"⏳ Retrying in {delay} seconds...")
            time.sleep(delay)
    raise last_error

# =========================================================
# PROCESS PDF WITH GEMINI
# =========================================================
def process_with_gemini(file_path, filename):
    uploaded_file = None
    try:
        print("\n========================================")
        print("📄 GEMINI PDF EXTRACTION")
        print("========================================")
        print(f"Filename: {filename}")

        print("⬆️ Uploading PDF to Gemini...")
        uploaded_file = client.files.upload(file=file_path)
        print(f"✅ PDF uploaded: {uploaded_file.name}")

        response = generate_with_retry(uploaded_file)

        if not response or not response.text:
            return {"error": "Gemini returned no/empty response."}

        ai_response = response.text.strip()
        print(f"📥 Response received: {len(ai_response)} characters")

        try:
            parsed_json = json.loads(ai_response)
        except json.JSONDecodeError as error:
            print("\n❌ GEMINI RETURNED INVALID JSON")
            return {"error": "Gemini returned invalid JSON."}

        log_entries = parsed_json.get("log_entries", [])
        if not isinstance(log_entries, list):
            return {"error": "Invalid log_entries format."}

        cleaned_entries = []
        for row in log_entries:
            if not isinstance(row, dict):
                continue
            cleaned_row = {
                "date": row.get("date", ""),
                "start_km": row.get("start_km", 0),
                "close_km": row.get("close_km", 0),
                "total_km": row.get("total_km", 0),
                "start_time": row.get("start_time", "00:00:00"),
                "close_time": row.get("close_time", "00:00:00"),
                "total_hrs": row.get("total_hrs", 0),
                "extra_km": row.get("extra_km", 0),
                "extra_hrs": row.get("extra_hrs", 0),
                "toll": row.get("toll", 0),
                "remarks": row.get("remarks", "")
            }
            cleaned_entries.append(cleaned_row)

        print(f"📊 TOTAL ROWS EXTRACTED: {len(cleaned_entries)}")
        print("✅ EXTRACTION COMPLETE\n")
        return {"data": cleaned_entries}

    except Exception as error:
        print("\n🔥 GEMINI ERROR")
        traceback.print_exc()
        error_text = str(error).upper()
        if "503" in error_text or "UNAVAILABLE" in error_text:
            return {"error": "Gemini is temporarily busy. Please try again after a few seconds."}
        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
            return {"error": "Gemini request limit reached. Please try again shortly."}
        return {"error": str(error)}

    finally:
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
                print("🗑️ Gemini temporary file deleted.")
            except:
                pass

# =========================================================
# ROUTES
# =========================================================
@app.route("/")
def home():
    return jsonify({"status": "Gemini AI Engine is running perfectly!", "endpoint": "/upload"})

@app.route("/upload", methods=["POST"])
def upload_file():
    print("🔥 BOOM! PDF received by Flask.")
    if "pdf_file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["pdf_file"]
    if not file or file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed."}), 400

    original_filename = secure_filename(file.filename)
    unique_filename = str(uuid.uuid4()) + "_" + original_filename
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

    try:
        file.save(file_path)
        print(f"💾 Temporary file saved: {file_path}")
        
        result = process_with_gemini(file_path, original_filename)

    except Exception as error:
        print("\n🔥 UPLOAD ERROR")
        traceback.print_exc()
        result = {"error": str(error)}

    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print("🗑️ Local temporary PDF deleted.")
            except:
                pass

    if "error" in result:
        return jsonify({"error": result["error"]}), 500

    return jsonify({"success": True, "log_entries": result.get("data", [])})

# =========================================================
# START SERVER
# =========================================================
if __name__ == "__main__":
    print("\n========================================")
    print("🚚 AI LOGSHEET EXTRACTOR (Powered by Gemini)")
    print("========================================")
    print(f"🤖 Model: {MODEL_NAME}")
    print("🌐 Server: http://127.0.0.1:5000")
    print("========================================\n")
    app.run(debug=True, port=5000)
