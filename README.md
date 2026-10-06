# Urgent message desk

The officer desk only. One saved model: `models/distilbert_optimized.joblib`. The tokenizer files in `models/distilbert_p1/` are not a second model. The duplicate weight file is not here.

Render start command:

```text
gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 180 app:app
```
