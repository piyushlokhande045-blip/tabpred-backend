# TABPred backend — Docker image for Render.com (or any Docker-based host)

FROM python:3.12-slim

# System packages some of the chemistry libs need at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- AutoDock Vina (Linux binary) ---
RUN wget -q https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/vina_1.2.7_linux_x86_64 \
    -O /usr/local/bin/vina \
    && chmod +x /usr/local/bin/vina

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "gdown>=5.1.0"
# App code (small files only — the model is fetched separately below)
COPY . .

# --- Download the compressed model bundle from Google Drive at build time ---
RUN mkdir -p ml_prediction_data
RUN gdown "1MORnoTxf5YRuC1JfQbNcGmuyvpdwyeGU" -O ml_prediction_data/tabpred_tuned_bundle_compressed.pkl
RUN ls -la ml_prediction_data/
RUN python3 -c "import os,sys; p='ml_prediction_data/tabpred_tuned_bundle_compressed.pkl'; s=os.path.getsize(p); print('Downloaded size:', s, 'bytes'); sys.exit(0 if s>1000000 else 1)"
ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "python app.py --host 0.0.0.0 --port ${PORT:-10000}"]
