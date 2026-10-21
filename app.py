import os
import json
import uuid
import traceback
import requests
import re
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

# =========================================================
# FLASK APP CONFIGURATION
# =========================================================
app = Flask(__name__)
UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================================================
# MISTRAL CONFIGURATION 
# =========================================================
MISTRAL_API_KEY = "mstrl_UVMS4UUWJNCHTjX7a2aSfxyPyP7tNIVw_3ka9kq"
HEADERS = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
    "Content-Type": "application/json"
}

# =========================================================
# EXTRACTION PROMPT 
# =========================================================
EXTRACTION_PROMPT = """
You are an expert vehicle logsheet data extraction system.
Read the following OCR text/markdown from an Indian vehicle logsheet.
Your task is to extract EVERY SINGLE LOG ENTRY and output it in strict JSON format.

CRITICAL RULES:
1. PROCESS THE ENTIRE TEXT. DO NOT skip or combine rows.
2. DATE FORMAT: Extract and return the date STRICTLY as DD-MM-YYYY (e.g., 25-08-2026).
3. NUMBERS: Return numerical values as strings or numbers (e.g., 1250, 0).
4. TIME: Return as HH:MM:SS (e.g., "09:30:00") ONLY for start_time and close_time.
5. MISSING VALUES: If a cell is blank, return 0 for numbers, "00:00:00" for time, and "" for text.
6. HOURS/DURATIONS (total_hrs & extra_hrs): If the text says "12:00", return the number 12. If it says "10:30", return 10.5. DO NOT remove colons to make large numbers like 12000.

You must return a JSON object with a single key "log_entries" which is an array of objects containing these exact keys:
["date", "start_km", "close_km", "total_km", "start_time", "close_time", "total_hrs", "extra_km", "extra_hrs", "toll", "remarks"]
"""

# =========================================================
# PROCESS PDF WITH MISTRAL OCR (Handles Scanned PDFs)
# =========================================================
def process_with_mistral_ocr(file_path, filename):
    try:
        print("\n========================================")
        print("📄 MISTRAL OCR & EXTRACTION")
        print("========================================")

        # 1. Upload PDF to Mistral Files API
        print("⬆️ Uploading file to Mistral OCR...")
        upload_url = "https://api.mistral.ai/v1/files"
        
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, "application/pdf")}
            data = {"purpose": "ocr"}
            
            upload_res = requests.post(
                upload_url, 
                headers={"Authorization": f"Bearer {MISTRAL_API_KEY}"}, 
                files=files, 
                data=data
            )
            
            if upload_res.status_code != 200:
                return {"error": f"Mistral File Upload Failed ({upload_res.status_code}): {upload_res.text}"}
                
            file_id = upload_res.json()["id"]

        # 2. Get Signed URL
        url_res = requests.get(f"https://api.mistral.ai/v1/files/{file_id}/url", headers=HEADERS)
        url_res.raise_for_status()
        signed_url = url_res.json()["url"]

        # 3. Run Mistral OCR
        print("🔍 Running Mistral OCR on PDF...")
        ocr_payload = {
            "model": "mistral-ocr-latest",
            "document": {"type": "document_url", "document_url": signed_url}
        }
        ocr_headers = HEADERS.copy()
        ocr_headers["Content-Type"] = "application/json"
        
        ocr_res = requests.post("https://api.mistral.ai/v1/ocr", headers=ocr_headers, json=ocr_payload)
        ocr_res.raise_for_status()
        ocr_result = ocr_res.json()

        # 4. Combine Markdown from all pages
        full_markdown = ""
        for page in ocr_result.get("pages", []):
            full_markdown += page.get("markdown", "") + "\n\n"

        if not full_markdown.strip():
            return {"error": "Mistral OCR found no text in the PDF."}

        print("🧠 Sending OCR markdown to Mistral Chat Model...")
        
        # 5. Send to Chat Model for Structured JSON Extraction
        chat_payload = {
            "model": "mistral-small-latest",
            "response_format": {"type": "json_object"}, 
            "messages": [
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": f"OCR TEXT:\n{full_markdown}"}
            ],
            "temperature": 0.1,
            "max_tokens": 8000
        }
        
        chat_res = requests.post("https://api.mistral.ai/v1/chat/completions", headers=ocr_headers, json=chat_payload)
        chat_res.raise_for_status()
        
        ai_response = chat_res.json()['choices'][0]['message']['content']

        # --- FIX: Clean Markdown backticks from AI Response ---
        ai_response = re.sub(r'^```json\s*', '', ai_response.strip(), flags=re.IGNORECASE)
        ai_response = re.sub(r'\s*```$', '', ai_response.strip())
        
        parsed_json = json.loads(ai_response)
        log_entries = parsed_json.get("log_entries", [])

        # 6. Clean and format entries
        cleaned_entries = []
        for row in log_entries:
            if not isinstance(row, dict): continue
            
            def parse_duration_hrs(val):
                if isinstance(val, str) and ":" in val:
                    parts = val.split(":")
                    return float(parts[0]) + (float(parts[1]) / 60 if len(parts) > 1 else 0)
                try:
                    val = float(val)
                    if val > 1000:  
                        return round(val / 1000, 2) 
                    return round(val, 2)
                except:
                    return 0

            cleaned_row = {
                "date": str(row.get("date", "")).strip(),
                "start_km": row.get("start_km", 0),
                "close_km": row.get("close_km", 0),
                "total_km": row.get("total_km", 0),
                "start_time": str(row.get("start_time", "00:00:00")),
                "close_time": str(row.get("close_time", "00:00:00")),
                "total_hrs": parse_duration_hrs(row.get("total_hrs", 0)),
                "extra_hrs": parse_duration_hrs(row.get("extra_hrs", 0)),
                "toll": row.get("toll", 0),
                "remarks": str(row.get("remarks", ""))
            }
            cleaned_entries.append(cleaned_row)

        print(f"✅ TOTAL ROWS EXTRACTED: {len(cleaned_entries)}\n")
        return {"data": cleaned_entries}

    except Exception as error:
        traceback.print_exc()
        return {"error": f"Mistral OCR Error: {str(error)}"}


# =========================================================
# ROUTES
# =========================================================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "pdf_file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["pdf_file"]
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    original_filename = secure_filename(file.filename)
    unique_filename = str(uuid.uuid4()) + "_" + original_filename
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

    result = None  # Safely initializing result
    
    try:
        file.save(file_path)
        result = process_with_mistral_ocr(file_path, original_filename)
    except Exception as error:
        result = {"error": f"File Processing Error: {str(error)}"}
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass

    # Safety check to prevent NoneType error completely
    if result is None:
        return jsonify({"error": "Critical Error: Process returned nothing."}), 500

    if "error" in result:
        return jsonify({"error": result["error"]}), 500

    return jsonify({"success": True, "log_entries": result.get("data", [])})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
