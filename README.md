Paperless OCR Service
=====================

FastAPI service that runs the Dolphin-v2 OCR pipeline against PDF documents
reachable by URL or S3. The server downloads each PDF, renders pages to images,
runs layout + element extraction, and returns normalized OCR results grouped by
page.

Highlights
----------
- PDF -> image rendering via PyMuPDF.
- Layout + element routing (code, equations, tables, figures, text).
- Figures returned as base64-encoded PNG crops.
- Stable response schema designed for downstream indexing/UX.

Requirements
------------
- Python 3.10+
- PyTorch and CUDA optional (uses GPU if available).
- Internet access on first run to download `ByteDance/Dolphin-v2` weights.

Setup
-----
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ./requirements.txt --no-build-isolation
```

Run locally
-----------
```bash
uvicorn server:server --reload
```

API
---
`POST /v1/ocr`

HTTP(S) file example:
```json
{
  "model": "dolphin2",
  "documents": [
    {
      "type": "url",
      "url": "https://example.com/document.pdf"
    }
  ]
}
```

Local file example:
```json
{
  "model": "dolphin2",
  "documents": [
    {
      "type": "url",
      "url": "file:///var/data/document.pdf"
    }
  ]
}
```

S3 example:
```json
{
  "model": "dolphin2",
  "documents": [
    {
      "type": "url",
      "url": "s3://my-bucket/path/to/document.pdf"
    }
  ]
}
```

Exclude elements example:
```json
{
  "model": "dolphin2",
  "documents": [
    {
      "type": "url",
      "url": "https://example.com/document.pdf",
      "exclude": ["figures", "tables"]
    }
  ]
}
```

Notes:
- The server downloads the document from the provided URL or S3 URI. `file://`
  URLs are supported only when the file is accessible on the server host.
- The `type` field must be `url` (alias for `DocumentPart.kind`).
- Use `exclude` to skip element types you do not need (e.g., `["figures"]`).

Response shape:
```json
{
  "pages": [
    {
      "page": 1,
      "codes": [
        {
          "label": "code",
          "bbox": [0, 0, 100, 100],
          "text": "print(\"hello\")",
          "reading_order": 0,
          "tags": []
        }
      ],
      "equations": [],
      "figures": [
        {
          "label": "fig",
          "bbox": [0, 0, 100, 100],
          "text": "iVBORw0KGgoAAAANSUhEUgAA...",
          "reading_order": 1,
          "tags": []
        }
      ],
      "tables": [],
      "text": []
    }
  ]
}
```
