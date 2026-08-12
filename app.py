"""
app.py

Browser-accessible web app for the trained c-Met (PDB 4R1V) binding
affinity model. Reuses the exact same prediction logic as main.py.

Run:
    pip install flask
    python3 app.py

Then open: http://localhost:5000

JSON API:
    curl "http://localhost:5000/api/predict?smiles=CCOc1ccc2nc(S(N)(=O)=O)sc2c1"

For access from OTHER machines on your network:
    python3 app.py --host 0.0.0.0
"""

import argparse
import traceback
from flask import Flask, request, render_template_string, jsonify

import main as model_backend

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    # Allows the standalone TABPred frontend (opened as a local file, or
    # served from a different port) to call this API from the browser.
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, ngrok-skip-browser-warning"
    return response

print("Loading model + receptor (this happens once at startup)...")
_bundle, _rec_coords, _rec_types, _rec_res_idx, _num_res, _bundle_path = model_backend.load_resources()
print(f"Ready. Using model: {_bundle_path}")


PAGE_TEMPLATE = """
<!doctype html>
<html>
<head>
  <title>TabPred - Binding Affinity Predictor</title>
  <style>
    body { font-family: sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }
    input[type=text] { width: 100%; padding: 8px; font-size: 15px; box-sizing: border-box; }
    button { padding: 8px 20px; font-size: 15px; margin-top: 10px; cursor: pointer; }
    .result { margin-top: 20px; padding: 14px; border-radius: 6px; }
    .ok { background: #e7f6e7; border: 1px solid #a3d9a3; }
    .err { background: #fbeaea; border: 1px solid #e0a3a3; }
    .note { color: #666; font-size: 13px; margin-top: 24px; }
  </style>
</head>
<body>
  <h2>c-Met (PDB 4R1V) Binding Affinity Predictor</h2>
  <p>Enter a SMILES string. Docks it (fast, exhaustiveness=3) and predicts
     the corrected binding affinity (kcal/mol).</p>
  <form method="POST">
    <input type="text" name="smiles" placeholder="e.g. CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
           value="{{ smiles or '' }}" required>
    <button type="submit">Predict</button>
  </form>

  {% if result is not none %}
    <div class="result ok">
      <strong>Predicted affinity:</strong> {{ result }} kcal/mol
    </div>
  {% endif %}
  {% if error %}
    <div class="result err">
      <strong>Error:</strong> {{ error }}
    </div>
  {% endif %}

  <p class="note">
    This predicts Vina's own docking score (delta-corrected toward
    exhaustive-search quality) for the c-Met target used in training --
    not experimental/wet-lab binding affinity.
  </p>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    smiles = None
    result = None
    error = None

    if request.method == "POST":
        smiles = request.form.get("smiles", "").strip()
        if not smiles:
            error = "Please enter a SMILES string."
        else:
            try:
                affinity = model_backend.predict(
                    smiles, _bundle, _rec_coords, _rec_types, _rec_res_idx, _num_res
                )
                result = f"{affinity:.3f}"
            except Exception as e:
                error = str(e)

    return render_template_string(PAGE_TEMPLATE, smiles=smiles, result=result, error=error)


@app.route("/api/predict", methods=["GET"])
def api_predict():
    smiles = request.args.get("smiles", "").strip()
    if not smiles:
        return jsonify({"error": "missing 'smiles' query parameter"}), 400

    try:
        affinity = model_backend.predict(
            smiles, _bundle, _rec_coords, _rec_types, _rec_res_idx, _num_res
        )
        return jsonify({"smiles": smiles, "predicted_affinity_kcal_mol": round(affinity, 3)})
    except Exception as e:
        return jsonify({"smiles": smiles, "error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)
