"""
Health Prediction Application
=============================
Flask + SQLite + Bootstrap 5 + Gemini API

Run:
    pip install -r requirements.txt
    python app.py
"""

import math
import os
import re
import sqlite3
from datetime import date, datetime
import time
import threading

import requests
from dotenv import load_dotenv
from flask import Flask, flash, g, redirect, render_template, request, url_for
from flask import Flask



# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
load_dotenv()  # reads GEMINI_API_KEY from a local ".env" file (if present)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "database.db")

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# GEMINI_URL = (
#     f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
# )

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "change-this-secret-in-production")  # needed for flash()

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip().strip('"').strip("'")
if GEMINI_MODEL.startswith("models/"):          # tolerate "models/gemini-..." in .env
    GEMINI_MODEL = GEMINI_MODEL[len("models/"):]
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
RETRY_STATUS = {429, 500, 503, 504}   # temporary errors worth retrying
GEMINI_FALLBACKS = [m.strip().removeprefix("models/") for m in
                    os.getenv("GEMINI_FALLBACK_MODELS", "gemini-flash-latest,gemini-flash-lite-latest").split(",")
                    if m.strip()]
# ---------------------------------------------------------------------------
# 2. DATABASE LAYER (SQLite)
# ---------------------------------------------------------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS patients (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name     TEXT    NOT NULL,
    date_of_birth TEXT    NOT NULL,          -- stored as ISO yyyy-mm-dd
    email         TEXT    NOT NULL,
    glucose       REAL    NOT NULL,
    haemoglobin   REAL    NOT NULL,
    cholesterol   REAL    NOT NULL,
    remarks       TEXT,                      -- filled by Gemini
    created_at    TEXT    DEFAULT CURRENT_TIMESTAMP
);
"""


def get_db():
    """Open one SQLite connection per request and reuse it."""
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row  # access columns by name
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    """Close the connection automatically when the request ends."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create the patients table automatically on first run."""
    with sqlite3.connect(DATABASE) as conn:
        conn.execute(CREATE_TABLE_SQL)


# ---------------------------------------------------------------------------
# 3. VALIDATION LOGIC
# ---------------------------------------------------------------------------
def parse_number(raw, label, errors, key):
    """Convert text to a non-negative finite float or record an error."""
    raw = (raw or "").strip()
    if not raw:
        errors[key] = f"{label} is required."
        return None
    try:
        value = float(raw)
    except ValueError:
        errors[key] = f"{label} must be a number."
        return None
    if not math.isfinite(value) or value < 0:
        errors[key] = f"{label} must be a positive number."
        return None
    return value


def validate_patient(form):
    """
    Validate the submitted form.
    Returns (cleaned_data, errors). If errors is empty the data is safe to save.
    """
    errors = {}

    # Full name – required
    full_name = (form.get("full_name") or "").strip()
    if not full_name:
        errors["full_name"] = "Full name is required."

    # Email – format check
    email = (form.get("email") or "").strip()
    if not email:
        errors["email"] = "Email address is required."
    elif not EMAIL_REGEX.match(email):
        errors["email"] = "Enter a valid email address, e.g. name@example.com."

    # Date of birth – dd/mm/yyyy and not in the future
    dob_raw = (form.get("date_of_birth") or "").strip()
    dob_iso = None
    if not dob_raw:
        errors["date_of_birth"] = "Date of birth is required."
    else:
        try:
            dob = datetime.strptime(dob_raw, "%d/%m/%Y").date()
            if dob > date.today():
                errors["date_of_birth"] = "Date of birth cannot be in the future."
            else:
                dob_iso = dob.isoformat()
        except ValueError:
            errors["date_of_birth"] = "Use the dd/mm/yyyy format, e.g. 25/12/1990."

    # Numeric lab values
    glucose = parse_number(form.get("glucose"), "Glucose", errors, "glucose")
    haemoglobin = parse_number(form.get("haemoglobin"), "Haemoglobin", errors, "haemoglobin")
    cholesterol = parse_number(form.get("cholesterol"), "Cholesterol", errors, "cholesterol")

    cleaned = {
        "full_name": full_name,
        "date_of_birth": dob_iso,
        "email": email,
        "glucose": glucose,
        "haemoglobin": haemoglobin,
        "cholesterol": cholesterol,
    }
    return cleaned, errors


def iso_to_display(iso_text):
    """2000-05-17  ->  17/05/2000 (for showing dates to the user)."""
    try:
        return datetime.strptime(iso_text, "%Y-%m-%d").strftime("%d/%m/%Y")
    except (TypeError, ValueError):
        return iso_text or ""


def row_to_dict(row):
    """Turn a DB row into a dict and add a dd/mm/yyyy date for templates."""
    d = dict(row)
    d["dob_display"] = iso_to_display(d["date_of_birth"])
    return d


# ---------------------------------------------------------------------------
# 4. GEMINI AI INTEGRATION
# ---------------------------------------------------------------------------
# def get_ai_assessment(glucose, haemoglobin, cholesterol):
#     """
#     Send the blood values to Gemini and return (text, success_flag).
#     Never raises: on any failure it returns a readable fallback message.
#     """
#     if not GEMINI_API_KEY:
#         return "AI assessment unavailable: GEMINI_API_KEY is not set.", False

#     prompt = (
#         "Patient blood test values:\n"
#         f"Glucose: {glucose}\n"
#         f"Haemoglobin: {haemoglobin}\n"
#         f"Cholesterol: {cholesterol}\n"
#         "Provide a short health risk assessment in one sentence."
#     )
#     payload = {"contents": [{"parts": [{"text": prompt}]}]}
#     headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

#     try:
#         resp = requests.post(GEMINI_URL, json=payload, headers=headers, timeout=30)
#         resp.raise_for_status()
#         data = resp.json()
#         text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
#         return (text or "AI assessment unavailable: empty response."), bool(text)
#     except requests.HTTPError as exc:
#         code = exc.response.status_code if exc.response is not None else "?"
#         return f"AI assessment unavailable: Gemini returned HTTP {code}.", False
#     except (requests.RequestException, KeyError, IndexError, ValueError):
#         return "AI assessment unavailable: could not reach Gemini.", False
# 
def get_ai_assessment(glucose, haemoglobin, cholesterol):
    """
    Call Gemini and return (text, success_flag).
    Retries temporary errors (503 etc.) with a short wait, then tries fallback models.
    """
    if not GEMINI_API_KEY:
        return "AI assessment unavailable: GEMINI_API_KEY is not set.", False

    prompt = (
        "Patient blood test values:\n"
        f"Glucose: {glucose}\n"
        f"Haemoglobin: {haemoglobin}\n"
        f"Cholesterol: {cholesterol}\n"
        "Provide a short health risk assessment in one sentence."
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    models_to_try = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACKS if m != GEMINI_MODEL]
    last_error = "unknown error"

    for model in models_to_try:
        url = f"{GEMINI_BASE}/models/{model}:generateContent"

        for attempt in range(3):                      # up to 3 tries per model
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=30)
            except requests.RequestException as exc:
                last_error = f"network error ({type(exc).__name__})"
                print(f"[Gemini] {model}: {last_error}", flush=True)
                time.sleep(2 ** attempt)              # wait 1s, 2s, 4s
                continue

            if resp.ok:
                try:
                    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                except (KeyError, IndexError, ValueError):
                    last_error = "unexpected response from Gemini"
                    print(f"[Gemini] {model}: {resp.text}", flush=True)
                    break                             # try the next model
                if text:
                    return text, True
                last_error = "empty response"
                break

            # Non-200: log the full body, remember a short message
            print(f"[Gemini ERROR] {model} HTTP {resp.status_code}: {resp.text}", flush=True)
            try:
                detail = resp.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                detail = resp.text
            last_error = f"HTTP {resp.status_code} - {detail[:200]}"

            if resp.status_code in RETRY_STATUS and attempt < 2:
                time.sleep(2 ** attempt)              # temporary: wait, then retry same model
                continue
            break                                     # permanent error or retries used up -> next model

   
    return f"AI assessment unavailable: {last_error}", False

PENDING_TEXT = "Generating AI assessment…"


def _ai_job(patient_id, glucose, haemoglobin, cholesterol):
    """Runs in a background thread: call Gemini, then save the result to the patient row."""
    remarks, _ok = get_ai_assessment(glucose, haemoglobin, cholesterol)
    # A thread can't use Flask's request-bound connection, so open its own.
    with sqlite3.connect(DATABASE, timeout=10) as conn:
        conn.execute("UPDATE patients SET remarks = ? WHERE id = ?", (remarks, patient_id))


def start_ai_job(patient_id, glucose, haemoglobin, cholesterol):
    threading.Thread(
        target=_ai_job, args=(patient_id, glucose, haemoglobin, cholesterol), daemon=True
    ).start()
# ---------------------------------------------------------------------------
# 5. ROUTES
# ---------------------------------------------------------------------------
def load_patients():
    # rows = get_db().execute("SELECT * FROM patients ORDER BY id DESC").fetchall()
    rows = get_db().execute("SELECT * FROM patients ORDER BY id ASC").fetchall()
    return [row_to_dict(r) for r in rows]


def dashboard_stats(patients):
    """Small summary numbers for the dashboard header."""
    n = len(patients)
    avg = lambda key: round(sum(p[key] for p in patients) / n, 1) if n else "–"
    return {
        "total": n,
        "avg_glucose": avg("glucose"),
        "avg_haemoglobin": avg("haemoglobin"),
        "avg_cholesterol": avg("cholesterol"),
    }


@app.route("/debug/models")
def debug_models():
    """TEMPORARY: lists models your API key can call with generateContent. Delete after fixing."""
    r = requests.get(f"{GEMINI_BASE}/models", params={"pageSize": 100},
                     headers={"x-goog-api-key": GEMINI_API_KEY}, timeout=30)
    if not r.ok:
        return {"status": r.status_code, "body": r.text}, r.status_code
    names = [m["name"].replace("models/", "")
             for m in r.json().get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    return {"configured_model": GEMINI_MODEL, "available_for_generateContent": names}

@app.route("/")
def index():
    """READ: dashboard with the empty form and the patient table."""
    patients = load_patients()
    return render_template(
        "index.html", patients=patients, stats=dashboard_stats(patients), form={}, errors={}
    )


@app.route("/add", methods=["POST"])
def add_patient():
    """CREATE: validate -> ask Gemini -> insert."""
    cleaned, errors = validate_patient(request.form)

    if errors:  # show form again with messages and the user's typed values
        patients = load_patients()
        flash("Please fix the highlighted fields.", "danger")
        return render_template(
            "index.html",
            patients=patients,
            stats=dashboard_stats(patients),
            form=request.form,
            errors=errors,
        ), 400

    # remarks, ai_ok = get_ai_assessment(
    #     cleaned["glucose"], cleaned["haemoglobin"], cleaned["cholesterol"]
    # )

    # db = get_db()
    # db.execute(
    #     """INSERT INTO patients
    #        (full_name, date_of_birth, email, glucose, haemoglobin, cholesterol, remarks)
    #        VALUES (?, ?, ?, ?, ?, ?, ?)""",
    #     (
    #         cleaned["full_name"], cleaned["date_of_birth"], cleaned["email"],
    #         cleaned["glucose"], cleaned["haemoglobin"], cleaned["cholesterol"], remarks,
    #     ),
    # )
    # db.commit()

    # if ai_ok:
    #     flash("Patient added and data is saved.", "success")
    # else:
    #     flash(f"Patient added, but {remarks[0].lower() + remarks[1:]}", "warning")
    # return redirect(url_for("index"))
    
    db = get_db()
    cur = db.execute(
        """INSERT INTO patients
           (full_name, date_of_birth, email, glucose, haemoglobin, cholesterol, remarks)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            cleaned["full_name"], cleaned["date_of_birth"], cleaned["email"],
            cleaned["glucose"], cleaned["haemoglobin"], cleaned["cholesterol"], PENDING_TEXT,
        ),
    )
    db.commit()

    # Gemini runs in the background, so the page responds immediately
    start_ai_job(cur.lastrowid, cleaned["glucose"], cleaned["haemoglobin"], cleaned["cholesterol"])
    flash("Patient added. The AI assessment is being generated and will appear shortly.", "success")
    return redirect(url_for("index"))


@app.route("/edit/<int:patient_id>", methods=["GET", "POST"])
def edit_patient(patient_id):
    """UPDATE: GET shows the pre-filled form, POST saves changes."""
    db = get_db()
    row = db.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
    if row is None:
        flash("Patient not found.", "danger")
        return redirect(url_for("index"))
    patient = row_to_dict(row)

    if request.method == "GET":
        form = {**patient, "date_of_birth": patient["dob_display"]}
        return render_template("edit.html", patient=patient, form=form, errors={})

    cleaned, errors = validate_patient(request.form)
    if errors:
        flash("Please fix the highlighted fields.", "danger")
        return render_template(
            "edit.html", patient=patient, form=request.form, errors=errors
        ), 400

    # Ask Gemini again only if lab values changed or the earlier AI call failed.
    # values_changed = (
    #     cleaned["glucose"] != patient["glucose"]
    #     or cleaned["haemoglobin"] != patient["haemoglobin"]
    #     or cleaned["cholesterol"] != patient["cholesterol"]
    # )
    # previous_failed = (patient["remarks"] or "").startswith("AI assessment unavailable")
    # remarks, ai_ok = patient["remarks"], True
    # if values_changed or previous_failed or not remarks:
    #     remarks, ai_ok = get_ai_assessment(
    #         cleaned["glucose"], cleaned["haemoglobin"], cleaned["cholesterol"]
    #     )

    # db.execute(
    #     """UPDATE patients SET full_name=?, date_of_birth=?, email=?, glucose=?,
    #        haemoglobin=?, cholesterol=?, remarks=? WHERE id=?""",
    #     (
    #         cleaned["full_name"], cleaned["date_of_birth"], cleaned["email"],
    #         cleaned["glucose"], cleaned["haemoglobin"], cleaned["cholesterol"],
    #         remarks, patient_id,
    #     ),
    # )
    # db.commit()

    # if ai_ok:
    #     flash("Patient updated successfully.", "success")
    # else:
    #     flash(f"Patient updated, but {remarks[0].lower() + remarks[1:]}", "warning")
    # return redirect(url_for("index"))
    
        # Regenerate the AI remark only if lab values changed or the last attempt failed/is stuck.
    values_changed = (
        cleaned["glucose"] != patient["glucose"]
        or cleaned["haemoglobin"] != patient["haemoglobin"]
        or cleaned["cholesterol"] != patient["cholesterol"]
    )
    old = patient["remarks"] or ""
    needs_ai = (
        values_changed or not old
        or old.startswith("AI assessment unavailable")
        or old.startswith("Generating")
    )
    remarks = PENDING_TEXT if needs_ai else old

    db.execute(
        """UPDATE patients SET full_name=?, date_of_birth=?, email=?, glucose=?,
           haemoglobin=?, cholesterol=?, remarks=? WHERE id=?""",
        (
            cleaned["full_name"], cleaned["date_of_birth"], cleaned["email"],
            cleaned["glucose"], cleaned["haemoglobin"], cleaned["cholesterol"],
            remarks, patient_id,
        ),
    )
    db.commit()

    if needs_ai:
        start_ai_job(patient_id, cleaned["glucose"], cleaned["haemoglobin"], cleaned["cholesterol"])
    flash("Patient updated successfully." + (" The AI assessment is being refreshed." if needs_ai else ""), "success")
    return redirect(url_for("index"))


@app.route("/delete/<int:patient_id>", methods=["POST"])
def delete_patient(patient_id):
    """DELETE: POST only, so a link or crawler can't delete by accident."""
    db = get_db()
    cur = db.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
    db.commit()
    if cur.rowcount:
        flash("Patient deleted.", "success")
    else:
        flash("Patient not found.", "danger")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# 6. ENTRY POINT
# ---------------------------------------------------------------------------
init_db()  # table is created automatically whenever the app starts

if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY is not set - remarks will show a fallback message.")
    app.run(debug=True)
