Paperless OCR Service
=====================

FastAPI service that runs the Dolphin-v2 OCR pipeline against PDF documents
reachable by URL. The server downloads each PDF, renders pages to images, runs
layout + element extraction, and returns normalized OCR results grouped by page.

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
pip install -r requirements.txt
```

Run locally
-----------
```bash
uvicorn server:server --reload
```

API
---
`POST /v1/ocr`

Request body:
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

Notes:
- The server downloads the document from the provided URL. Local file paths are
  not supported by this endpoint.
- The `type` field must be `url` (alias for `RemoteDocumentPart.kind`).

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
