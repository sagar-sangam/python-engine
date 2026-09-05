
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
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. "
        "Please add it to your .env file."
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

client = genai.Client(
    api_key=GEMINI_API_KEY
)

# Model recommended by the error returned by your API
MODEL_NAME = "gemini-3.6-flash"


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

                    "date": {
                        "type": "string"
                    },

                    "start_km": {
                        "type": "number"
                    },

                    "close_km": {
                        "type": "number"
                    },

                    "total_km": {
                        "type": "number"
                    },

                    "start_time": {
                        "type": "string"
                    },

                    "close_time": {
                        "type": "string"
                    },

                    "total_hrs": {
                        "type": "number"
                    },

                    "extra_km": {
                        "type": "number"
                    },

                    "extra_hrs": {
                        "type": "number"
                    },

                    "toll": {
                        "type": "number"
                    },

                    "remarks": {
                        "type": "string"
                    }
                },

                "required": [
                    "date",
                    "start_km",
                    "close_km",
                    "total_km",
                    "start_time",
                    "close_time",
                    "total_hrs",
                    "extra_km",
                    "extra_hrs",
                    "toll",
                    "remarks"
                ]
            }
        }
    },

    "required": [
        "log_entries"
    ]
}


# =========================================================
# EXTRACTION PROMPT
# =========================================================

EXTRACTION_PROMPT = """
You are an expert vehicle logsheet data extraction system.

The attached PDF is an Indian vehicle logsheet.

Your task is to extract EVERY SINGLE LOG ENTRY / TABLE ROW
from the COMPLETE PDF.

=========================================================
VERY IMPORTANT
=========================================================

PROCESS THE ENTIRE PDF.

Read ALL pages.

Do NOT only read the first page.

Every physical row in the vehicle logsheet must become
exactly ONE object inside "log_entries".

DO NOT:

- skip rows
- summarize rows
- combine rows
- invent values
- ignore later pages
- ignore small table rows
- ignore scanned pages

If the same date appears in multiple rows, create multiple
separate objects.

=========================================================
TABLE EXTRACTION
=========================================================

Carefully understand the table columns and their alignment.

Make sure each value belongs to the correct row.

Do not move values from one row to another.

If a table continues on another page, continue extracting
the rows from that page.

=========================================================
NUMBER FIELDS
=========================================================

These fields MUST be numbers:

start_km
close_km
total_km
total_hrs
extra_km
extra_hrs
toll

Remove commas and other formatting characters.

Examples:

"1,250" -> 1250

"2,500.50" -> 2500.50

If a numeric field is genuinely missing:

0

Do not convert a valid number into text.

=========================================================
TIME FIELDS
=========================================================

Fields:

start_time
close_time

Use:

HH:MM:SS

Examples:

09:30 -> 09:30:00

5:45 PM -> 17:45:00

If genuinely missing:

00:00:00

=========================================================
DATE
=========================================================

Return date as:

YYYY-MM-DD

Only convert the date when the actual date is clear.

NEVER invent a date.

=========================================================
TOTALS
=========================================================

If the document already contains:

total_km

or

total_hrs

use the value printed in the document.

Do not unnecessarily recalculate it.

The printed value has priority.

=========================================================
REMARKS
=========================================================

Extract the actual remarks associated with each row.

If there is no remark:

""

=========================================================
MISSING VALUES
=========================================================

If information is genuinely missing:

Number -> 0

Time -> "00:00:00"

Text -> ""

Do not guess.

=========================================================
FINAL VERIFICATION
=========================================================

Before returning the answer:

1. Check every page.
2. Check every table.
3. Count all physical rows.
4. Make sure every row is represented.
5. Make sure no rows are duplicated.
6. Verify date values.
7. Verify KM values.
8. Verify time values.
9. Verify toll values.
10. Verify remarks.

Return ONLY valid JSON matching the supplied schema.

Do not return Markdown.

Do not return explanations.

Do not return code fences.

Do not return any text outside the JSON object.
"""


# =========================================================
# RETRY CONFIGURATION
# =========================================================

MAX_RETRIES = 5

INITIAL_RETRY_DELAY = 2

MAX_RETRY_DELAY = 30


# =========================================================
# HELPER:
# CHECK WHETHER ERROR IS TEMPORARY
# =========================================================

def is_retryable_error(error):

    error_text = str(error).upper()

    retryable_errors = [

        "503",
        "UNAVAILABLE",

        "429",
        "RESOURCE_EXHAUSTED",

        "500",
        "INTERNAL",

        "502",
        "BAD_GATEWAY",

        "504",
        "DEADLINE_EXCEEDED",

        "TIMEOUT",
        "TIMED OUT"
    ]

    return any(
        item in error_text
        for item in retryable_errors
    )


# =========================================================
# GEMINI REQUEST WITH RETRY
# =========================================================

def generate_with_retry(uploaded_file):

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            print(
                f"🧠 Gemini request "
                f"{attempt}/{MAX_RETRIES}"
            )

            response = client.models.generate_content(

                model=MODEL_NAME,

                contents=[
                    uploaded_file,
                    EXTRACTION_PROMPT
                ],

                config=types.GenerateContentConfig(

                    response_mime_type="application/json",

                    response_schema=RESPONSE_SCHEMA
                )
            )

            print(
                "✅ Gemini request successful"
            )

            return response


        except Exception as error:

            last_error = error

            print(
                f"⚠️ Gemini request failed "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

            print(
                str(error)
            )


            # ---------------------------------------------
            # Do not retry permanent errors
            # ---------------------------------------------

            if not is_retryable_error(error):

                raise


            # ---------------------------------------------
            # Maximum retries reached
            # ---------------------------------------------

            if attempt >= MAX_RETRIES:

                print(
                    "❌ Maximum Gemini retries reached."
                )

                raise


            # ---------------------------------------------
            # Exponential backoff
            # ---------------------------------------------

            delay = min(

                INITIAL_RETRY_DELAY * (2 ** (attempt - 1)),

                MAX_RETRY_DELAY
            )

            print(
                f"⏳ Retrying in {delay} seconds..."
            )

            time.sleep(delay)


    raise last_error


# =========================================================
# PROCESS PDF WITH GEMINI
# =========================================================

def process_with_gemini(
    file_path,
    filename
):

    uploaded_file = None

    try:

        print("\n")
        print("========================================")
        print("📄 GEMINI PDF EXTRACTION")
        print("========================================")

        print(
            f"Filename: {filename}"
        )


        # =================================================
        # 1. UPLOAD PDF TO GEMINI
        # =================================================

        print(
            "⬆️ Uploading PDF to Gemini..."
        )

        uploaded_file = client.files.upload(
            file=file_path
        )

        print(
            "✅ PDF uploaded successfully"
        )

        print(
            f"Gemini file: "
            f"{uploaded_file.name}"
        )


        # =================================================
        # 2. GEMINI EXTRACTION
        # =================================================

        response = generate_with_retry(
            uploaded_file
        )


        # =================================================
        # 3. CHECK RESPONSE
        # =================================================

        if not response:

            return {
                "error":
                "Gemini returned no response."
            }


        if not response.text:

            return {
                "error":
                "Gemini returned an empty response."
            }


        ai_response = response.text.strip()


        print(
            f"📥 Response received: "
            f"{len(ai_response)} characters"
        )


        # =================================================
        # 4. PARSE JSON
        # =================================================

        try:

            parsed_json = json.loads(
                ai_response
            )

        except json.JSONDecodeError as error:

            print(
                "\n❌ GEMINI RETURNED INVALID JSON"
            )

            print(
                "--------------------------------"
            )

            print(
                ai_response
            )

            print(
                "--------------------------------"
            )

            return {
                "error":
                "Gemini returned invalid JSON."
            }


        # =================================================
        # 5. GET LOG ENTRIES
        # =================================================

        log_entries = parsed_json.get(
            "log_entries",
            []
        )


        if not isinstance(
            log_entries,
            list
        ):

            return {
                "error":
                "Invalid log_entries format."
            }


        # =================================================
        # 6. CLEAN / NORMALIZE ROWS
        # =================================================

        cleaned_entries = []


        for row in log_entries:

            if not isinstance(
                row,
                dict
            ):

                continue


            cleaned_row = {

                "date":
                    row.get(
                        "date",
                        ""
                    ),

                "start_km":
                    row.get(
                        "start_km",
                        0
                    ),

                "close_km":
                    row.get(
                        "close_km",
                        0
                    ),

                "total_km":
                    row.get(
                        "total_km",
                        0
                    ),

                "start_time":
                    row.get(
                        "start_time",
                        "00:00:00"
                    ),

                "close_time":
                    row.get(
                        "close_time",
                        "00:00:00"
                    ),

                "total_hrs":
                    row.get(
                        "total_hrs",
                        0
                    ),

                "extra_km":
                    row.get(
                        "extra_km",
                        0
                    ),

                "extra_hrs":
                    row.get(
                        "extra_hrs",
                        0
                    ),

                "toll":
                    row.get(
                        "toll",
                        0
                    ),

                "remarks":
                    row.get(
                        "remarks",
                        ""
                    )
            }


            cleaned_entries.append(
                cleaned_row
            )


        # =================================================
        # 7. RESULT
        # =================================================

        print(
            f"📊 TOTAL ROWS EXTRACTED: "
            f"{len(cleaned_entries)}"
        )

        print(
            "========================================"
        )
        print(
            "✅ EXTRACTION COMPLETE"
        )
        print(
            "========================================\n"
        )


        return {
            "data":
            cleaned_entries
        }


    except Exception as error:

        print("\n")
        print("========================================")
        print("🔥 GEMINI ERROR")
        print("========================================")

        traceback.print_exc()


        error_text = str(error)


        # =================================================
        # FRIENDLY ERROR FOR 503
        # =================================================

        if (
            "503" in error_text
            or "UNAVAILABLE" in error_text.upper()
        ):

            return {
                "error":
                "Gemini is temporarily busy. "
                "Please try again after a few seconds."
            }


        # =================================================
        # FRIENDLY ERROR FOR 429
        # =================================================

        if (
            "429" in error_text
            or "RESOURCE_EXHAUSTED"
            in error_text.upper()
        ):

            return {
                "error":
                "Gemini request limit reached. "
                "Please try again shortly."
            }


        # =================================================
        # OTHER ERROR
        # =================================================

        return {
            "error":
            error_text
        }


    finally:

        # =================================================
        # DELETE GEMINI FILE
        # =================================================

        if uploaded_file:

            try:

                client.files.delete(
                    name=uploaded_file.name
                )

                print(
                    "🗑️ Gemini temporary file deleted."
                )

            except Exception as cleanup_error:

                print(
                    "⚠️ Gemini file cleanup failed:"
                )

                print(
                    cleanup_error
                )


# =========================================================
# HOME ROUTE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# UPLOAD ROUTE
# =========================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_file():

    print(
        "🔥 BOOM! PDF received by Flask."
    )


    # =====================================================
    # CHECK FILE
    # =====================================================

    if "pdf_file" not in request.files:

        return jsonify({
            "error":
            "No file uploaded."
        }), 400


    file = request.files[
        "pdf_file"
    ]


    if not file or file.filename == "":

        return jsonify({
            "error":
            "Empty filename."
        }), 400


    # =====================================================
    # CHECK PDF
    # =====================================================

    if not file.filename.lower().endswith(
        ".pdf"
    ):

        return jsonify({
            "error":
            "Only PDF files are allowed."
        }), 400


    # =====================================================
    # SECURE FILENAME
    # =====================================================

    original_filename = secure_filename(
        file.filename
    )


    if not original_filename:

        return jsonify({
            "error":
            "Invalid filename."
        }), 400


    # =====================================================
    # UNIQUE TEMP FILENAME
    # =====================================================

    unique_filename = (

        str(uuid.uuid4())
        + "_"
        + original_filename
    )


    file_path = os.path.join(

        app.config[
            "UPLOAD_FOLDER"
        ],

        unique_filename
    )


    try:

        # =================================================
        # SAVE FILE
        # =================================================

        file.save(
            file_path
        )

        print(
            f"💾 Temporary file saved:"
        )

        print(
            file_path
        )


        # =================================================
        # PROCESS WITH GEMINI
        # =================================================

        result = process_with_gemini(

            file_path,

            original_filename
        )


    except Exception as error:

        print(
            "\n🔥 UPLOAD ERROR"
        )

        traceback.print_exc()


        result = {

            "error":
            str(error)
        }


    finally:

        # =================================================
        # DELETE LOCAL FILE
        # =================================================

        if os.path.exists(
            file_path
        ):

            try:

                os.remove(
                    file_path
                )

                print(
                    "🗑️ Local temporary PDF deleted."
                )

            except Exception as cleanup_error:

                print(
                    "⚠️ Local cleanup failed:"
                )

                print(
                    cleanup_error
                )


    # =====================================================
    # RETURN ERROR
    # =====================================================

    if "error" in result:

        return jsonify({

            "error":
            result["error"]

        }), 500


    # =====================================================
    # RETURN SUCCESS
    # =====================================================

    return jsonify({

        "success":
        True,

        "log_entries":
        result.get(
            "data",
            []
        )

    })


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    print("")
    print("========================================")
    print("🚚 AI LOGSHEET EXTRACTOR")
    print("========================================")
    print(
        f"🤖 Model: {MODEL_NAME}"
    )
    print(
        "🌐 Server: http://127.0.0.1:5000"
    )
    print("========================================")
    print("")

    app.run(
        debug=True,
        port=5000
    )
