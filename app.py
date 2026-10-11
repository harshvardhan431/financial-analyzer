import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from analyzer import load_file, analyze

app = Flask(__name__)
app.secret_key = "bi-analyzer-secret-key-enterprise"

# In-memory store for reports so output is separated on its own page
REPORTS = {}


def get_or_create_sample_report():
    if "sample" not in REPORTS:
        sample_path = os.path.join(os.path.dirname(__file__), "sample_data.csv")
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                df = load_file(f, "sample_data.csv")
                res = analyze(df)
                REPORTS["sample"] = {
                    "result": res,
                    "filename": "sample_data.csv (Demo Dataset)",
                    "time": datetime.now().strftime("%b %d, %Y &bull; %I:%M %p")
                }
    return REPORTS.get("sample")


@app.route("/", methods=["GET", "POST"])
def index():
    # If a legacy POST was made to root, redirect to submit handler logic
    if request.method == "POST":
        return submit()
    return render_template("index.html", active_page="home")


@app.route("/submit", methods=["GET", "POST"])
def submit():
    # Quick 1-click load for demo dataset
    if request.method == "GET" and request.args.get("demo"):
        sample_report = get_or_create_sample_report()
        if sample_report:
            return redirect(url_for("output", id="sample"))

    if request.method == "POST":
        f = request.files.get("file")
        if not f or not f.filename:
            return render_template("submit.html", active_page="submit", error="Please choose a valid CSV or Excel file to analyze.")
        
        try:
            df = load_file(f, f.filename)
            res = analyze(df)
            if "error" in res:
                return render_template("submit.html", active_page="submit", error=res["error"])
            
            report_id = str(uuid.uuid4())[:8]
            REPORTS[report_id] = {
                "result": res,
                "filename": f.filename,
                "time": datetime.now().strftime("%b %d, %Y &bull; %I:%M %p")
            }
            return redirect(url_for("output", id=report_id))
        except Exception as e:
            return render_template("submit.html", active_page="submit", error=f"Could not read or parse file: {str(e)}")

    return render_template("submit.html", active_page="submit", error=None)


@app.route("/output")
def output():
    report_id = request.args.get("id")
    report = None

    if report_id and report_id in REPORTS:
        report = REPORTS[report_id]
    else:
        # Default or fallback to sample dataset so dashboard is immediately rich & populated
        report = get_or_create_sample_report()

    if not report:
        return redirect(url_for("submit"))

    return render_template(
        "output.html",
        active_page="output",
        result=report["result"],
        filename=report.get("filename", "Business Dataset"),
        report_time=report.get("time", datetime.now().strftime("%b %d, %Y &bull; %I:%M %p"))
    )


@app.route("/about")
def about():
    return redirect(url_for("index") + "#about")


if __name__ == "__main__":
    app.run(debug=True, port=5000)