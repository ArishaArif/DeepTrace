# Build from the REPO ROOT so the ml/ package is included:
#   docker build -f backend/Dockerfile -t deeptrace .
#   docker run -p 8000:8000 -e ALLOWED_ORIGINS=https://your-frontend deeptrace
FROM python:3.11-slim
WORKDIR /srv
COPY backend/requirements.txt backend/requirements.txt
# ML roles: add torch / open_clip / opencv etc. to ml/requirements.txt (optional file)
COPY ml/requirements.tx[t] ml/
RUN pip install --no-cache-dir -r backend/requirements.txt \
 && if [ -f ml/requirements.txt ]; then pip install --no-cache-dir -r ml/requirements.txt; fi
COPY backend backend
COPY ml ml
ENV PYTHONPATH=/srv PORT=8000
WORKDIR /srv/backend
EXPOSE 8000
# One worker: the model loads once and is shared. Concurrency is capped inside the app.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
