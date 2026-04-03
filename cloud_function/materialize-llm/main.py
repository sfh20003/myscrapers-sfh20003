import csv
import io
import json
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, Iterable

from flask import Request, jsonify
from google.cloud import storage

# -------------------- ENV --------------------
BUCKET_NAME        = os.getenv("GCS_BUCKET")
STRUCTURED_PREFIX  = os.getenv("STRUCTURED_PREFIX", "structured")

storage_client = storage.Client()

RUN_ID_ISO_RE   = re.compile(r"^\d{8}T\d{6}Z$")
RUN_ID_PLAIN_RE = re.compile(r"^\d{14}$")

CSV_COLUMNS = [
    "post_id", "run_id", "scraped_at",
    "price", "year", "make", "model", "mileage",
    "color", "fuel_type", "cylinder", "transmission",
    "city", "state",
    "source_txt"
]

def _run_id_to_dt(rid: str) -> datetime:
    try:
        if RUN_ID_ISO_RE.match(rid):
            return datetime.strptime(rid, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        if RUN_ID_PLAIN_RE.match(rid):
            return datetime.strptime(rid, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return datetime.now(timezone.utc)

def _list_recent_run_ids(bucket: str, structured_prefix: str, hours_back: int = 1) -> list[str]:
    it = storage_client.list_blobs(bucket, prefix=f"{structured_prefix}/", delimiter="/")
    for _ in it:
        pass
    
    now = datetime.now(timezone.utc)
    recent_runs = []
    
    for p in getattr(it, "prefixes", []):
        tail = p.rstrip("/").split("/")[-1]
        if tail.startswith("run_id="):
            rid = tail.split("run_id=", 1)[1]
            run_dt = _run_id_to_dt(rid)
            if now - run_dt <= timedelta(hours=hours_back):
                recent_runs.append(rid)
                
    return sorted(recent_runs)

def _jsonl_records_for_run(bucket: str, structured_prefix: str, run_id: str):
    b = storage_client.bucket(bucket)
    prefix = f"{structured_prefix}/run_id={run_id}/jsonl_llm/"
    for blob in b.list_blobs(prefix=prefix):
        if not blob.name.endswith(".jsonl"):
            continue
        data = blob.download_as_text()
        for line in data.splitlines():
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                rec.setdefault("run_id", run_id)
                yield rec
            except Exception:
                continue

def _write_csv(records: Iterable[Dict], dest_key: str, columns=CSV_COLUMNS) -> int:
    b = storage_client.bucket(BUCKET_NAME)
    blob = b.blob(dest_key)
    n = 0
    with blob.open("w") as out:
        w = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            row = {c: rec.get(c, None) for c in columns}
            w.writerow(row)
            n += 1
    return n

def materialize_http(request: Request):
    """
    HTTP POST: Processes only the LAST HOUR to prevent memory/timeout 'bombing'.
    Saves to listings_master_llm_V1.csv
    """
    try:
        if not BUCKET_NAME:
            return jsonify({"ok": False, "error": "missing GCS_BUCKET env"}), 500

        run_ids = _list_recent_run_ids(BUCKET_NAME, STRUCTURED_PREFIX, hours_back=1)
        
        if not run_ids:
            return jsonify({
                "ok": True, 
                "message": "No new runs found in the last hour.",
                "runs_scanned": 0
            }), 200

        latest_by_post: Dict[str, Dict] = {}
        for rid in run_ids:
            for rec in _jsonl_records_for_run(BUCKET_NAME, STRUCTURED_PREFIX, rid):
                pid = rec.get("post_id")
                if not pid: continue
                
                prev = latest_by_post.get(pid)
                if (prev is None) or (_run_id_to_dt(rec.get("run_id", rid)) > _run_id_to_dt(prev.get("run_id", ""))):
                    latest_by_post[pid] = rec

        base = f"{STRUCTURED_PREFIX}/datasets"
        # Renamed as requested: V1 added
        final_key = f"{base}/listings_master_llm_V1.csv"
        rows = _write_csv(latest_by_post.values(), final_key)

        return jsonify({
            "ok": True,
            "runs_scanned": len(run_ids),
            "unique_listings_in_window": len(latest_by_post),
            "rows_written": rows,
            "output_csv": f"gs://{BUCKET_NAME}/{final_key}"
        }), 200

    except Exception as e:
        return jsonify({"ok": False, "error": f"{type(e).__name__}: {e}"}), 500
