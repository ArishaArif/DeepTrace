# Backend

```
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Docs UI: http://localhost:8000/docs

Runs in dummy mode until `ml/predict.py` exists (force it with `USE_DUMMY=1`).
Tests: `pytest`
