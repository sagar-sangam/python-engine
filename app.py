import os
import requests
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Tumhari Mistral API Key
API_KEY = "NpFWXSweAk42UxShvetW2FPOx05zHcZn"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ye route browser mein check karne ke liye hai taaki 404 na aaye
@app.route('/')
def home():
    return jsonify({"status": "AI Engine is running perfectly!", "endpoint": "/upload"})

def process_with_mistral(file_path, filename):
    try:
        # 1. Upload File to Mistral
        upload_url = "https://api.mistral.ai/v1/files"
        with open(file_path, "rb") as f:
            files = {"file": (filename, f, "application/pdf")}
            data = {"purpose": "ocr"}
            upload_res = requests.post(upload_url, headers=HEADERS, files=files, data=data)
            upload_res.raise_for_status()
            file_id = upload_res.json()["id"]

        # 2. Get Signed URL
        url_res = requests.get(f"https://api.mistral.ai/v1/files/{file_id}/url", headers=HEADERS)
        url_res.raise_for_status()
        signed_url = url_res.json()["url"]

        # 3. Run OCR
        ocr_payload = {
            "model": "mistral-ocr-latest",
            "document": {"type": "document_url", "document_url": signed_url}
        }
        ocr_headers = HEADERS.copy()
        ocr_headers["Content-Type"] = "application/json"
        
        ocr_res = requests.post("https://api.mistral.ai/v1/ocr", headers=ocr_headers, json=ocr_payload)
        ocr_res.raise_for_status()
        result = ocr_res.json()

        # 4. Combine Markdown
        full_markdown = ""
        for page in result.get("pages", []):
            full_markdown += page.get("markdown", "") + "\n\n"

        if not full_markdown.strip():
            return {"error": "No text found in PDF"}

        # 5. Ask AI to format data strictly into JSON Object
        prompt = f"""
        You are an expert data extractor. Read the following OCR Markdown text from an Indian vehicle logsheet.
        Extract the table data and return ONLY a JSON object containing a key "log_entries" with an array of objects inside it.
        
        Required JSON Structure:
        {{
            "log_entries": [
                {{
                    "date": "YYYY-MM-DD",
                    "start_km": 0,
                    "close_km": 0,
                    "total_km": 0,
                    "start_time": "HH:MM:SS",
                    "close_time": "HH:MM:SS",
                    "total_hrs": 0,
                    "extra_km": 0,
                    "extra_hrs": 0,
                    "toll": 0,
                    "remarks": "string"
                }}
            ]
        }}

        Rules:
        1. If a value is missing or not applicable, use 0 for numbers, "00:00:00" for time, and "" for text.
        2. Remove all commas or text characters from numbers.

        OCR TEXT:
        {full_markdown}
        """
        
        chat_payload = {
            "model": "mistral-small-latest",
            "response_format": {"type": "json_object"}, 
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        
        # Yahan URL theek kar diya gaya hai
        chat_res = requests.post("https://api.mistral.ai/v1/chat/completions", headers=ocr_headers, json=chat_payload)
        chat_res.raise_for_status()
        
        ai_response = chat_res.json()['choices'][0]['message']['content']
        
        # Clean response and extract the inner array
        parsed_json = json.loads(ai_response)
        log_entries = parsed_json.get("log_entries", [])
                
        return {"data": log_entries}

    except Exception as e:
        return {"error": str(e)}

@app.route('/upload', methods=['POST'])
def upload_file():
    print("🔥 BOOM! PHP se request aa gayi Flask ke paas!")
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Process File
    result = process_with_mistral(file_path, filename)

    # Cleanup temp file
    if os.path.exists(file_path):
        os.remove(file_path)

    if "error" in result:
        return jsonify({"error": result["error"]}), 500

    return jsonify({"success": True, "log_entries": result["data"]})

if __name__ == "__main__":
    app.run(debug=True, port=5000)