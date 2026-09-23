"""Orkid scorer — endpoint conformance dashboard + history.

The hackathon judges panel generalized for internal use: score any
HTTP endpoint against any rubric file, keep results in SQLite, and
optionally run it on a schedule (cron/systemd) as a regression probe.

Run:
    export SV_DATA_ROOT=<dataset root or https://dataserver>
    export SV_DATA_TOKEN=<token>            # if remote
    export SCORER_TOKEN=<bearer>            # optional: gate the UI/API
    uvicorn app:app --port 9000

Endpoints:
    GET  /            dashboard (scores, history)
    GET  /score?team=X&url=Y&rubric=Z   run rubric, persist result
    GET  /api/scores  JSON scoreboard + history
    GET  /api/scores/{team}             full history for one target
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

HERE = Path(__file__).parent
DEFAULT_RUBRIC = HERE.parent / "rubric" / "check.py"
RUBRIC_DIR = HERE.parent / "rubric"
DB = Path(os.environ.get("SCORER_DB", HERE / "scores.db"))
GATE = os.environ.get("SCORER_TOKEN")

app = FastAPI(title="orkid scorer", version="0.2.0")


def db() -> sqlite3.Connection:
    c = sqlite3.connect(DB)
    c.execute("""CREATE TABLE IF NOT EXISTS scores(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        team TEXT, url TEXT, rubric TEXT,
        score REAL, max REAL, detail TEXT, ts REAL)""")
    return c


def run_rubric(url: str, check_py: Path, rubric_yaml: Path | None) -> dict:
    cmd = [sys.executable, str(check_py), "--base-url", url]
    if rubric_yaml:
        cmd += ["--rubric", str(rubric_yaml)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if out.returncode != 0 or not out.stdout.strip().startswith("{"):
        raise RuntimeError(out.stderr[-500:] or out.stdout[-500:] or "no output")
    return json.loads(out.stdout)


@app.middleware("http")
async def gate(request: Request, call_next):
    if GATE and request.headers.get("authorization") != f"Bearer {GATE}" \
            and request.query_params.get("key") != GATE:
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


def _rows(limit: int = 50):
    c = db()
    try:
        return c.execute(
            "SELECT team,url,rubric,score,max,ts FROM scores "
            "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    finally:
        c.close()


PAGE = """<!doctype html><meta charset="utf-8"><title>orkid scorer</title>
<style>
 body{{font-family:ui-monospace,monospace;background:#0d1117;color:#e6edf3;max-width:1000px;margin:2em auto;padding:0 1em}}
 input,select{{background:#161b22;color:#e6edf3;border:1px solid #30363d;padding:.5em}}
 button{{background:#238636;color:#fff;border:0;padding:.55em 1.2em;cursor:pointer}}
 table{{border-collapse:collapse;width:100%;margin-top:1.5em}}
 td,th{{border:1px solid #30363d;padding:.4em .6em;text-align:left}}
 .ok{{color:#3fb950}} .bad{{color:#f85149}} .dim{{color:#8b949e}}
</style>
<h1>Orkid scorer</h1>
<form method=get action=/score>
 <input name=team placeholder="label (team/service)" required>
 <input name=url placeholder="https://endpoint" required style="width:26em">
 <input name=rubric placeholder="rubric.yaml (blank = hackathon public)" style="width:24em">
 <button>Score</button>
</form>
{body}
"""


def history_html() -> str:
    rows = "".join(
        f"<tr><td>{t}</td><td class=dim>{u}</td><td class=dim>{Path(r).name}</td>"
        f"<td class={'ok' if s == m else 'bad'}>{s:g}/{m:g}</td>"
        f"<td class=dim>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}</td></tr>"
        for t, u, r, s, m, ts in _rows())
    if not rows:
        return "<p class=dim>no scores yet</p>"
    return ("<h2>History</h2><table><tr><th>label</th><th>endpoint</th>"
            f"<th>rubric</th><th>score</th><th>when</th></tr>{rows}</table>")


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.format(body=history_html())


@app.get("/score", response_class=HTMLResponse)
def score(team: str = Query(...), url: str = Query(...),
          rubric: str = Query("")):
    t0 = time.time()
    check_py = DEFAULT_RUBRIC
    rub_yaml = None
    if rubric:
        rub_yaml = Path(rubric)
        if not rub_yaml.suffix:
            rub_yaml = rub_yaml.with_suffix(".yaml")
        # sibling check.py next to the rubric yaml, else the public one
        sib = rub_yaml.with_name("check.py")
        check_py = sib if sib.exists() else DEFAULT_RUBRIC
    try:
        result = run_rubric(url, check_py, rub_yaml)
    except Exception as e:
        return PAGE.format(
            body=f"<p class=bad>{team}: rubric failed — {e}</p>" + history_html())

    c = db()
    try:
        c.execute(
            "INSERT INTO scores(team,url,rubric,score,max,detail,ts) "
            "VALUES(?,?,?,?,?,?,?)",
            (team, url, rubric or "rubric.yaml", result["score"],
             result["max"], json.dumps(result["checks"]), t0))
        c.commit()
    finally:
        c.close()

    rows = "".join(
        f"<tr><td>{c['id']}</td><td class={'ok' if c['earned']==c['max'] else 'bad'}>"
        f"{c['earned']:g}/{c['max']:g}</td><td class=dim>{c['note']}</td></tr>"
        for c in result["checks"])
    body = (f"<h2>{team} — {result['score']:g}/{result['max']:g}</h2>"
            f"<table><tr><th>check</th><th>pts</th><th>note</th></tr>{rows}</table>"
            + history_html()
            + f"<p class=dim>ran in {time.time()-t0:.1f}s</p>")
    return PAGE.format(body=body)


@app.get("/api/scores")
def api_scores(limit: int = 50):
    return [{"team": t, "url": u, "rubric": r, "score": s, "max": m, "ts": ts}
            for t, u, r, s, m, ts in _rows(limit)]


@app.get("/api/scores/{team}")
def api_team(team: str):
    c = db()
    try:
        rows = c.execute(
            "SELECT url,rubric,score,max,ts FROM scores WHERE team=? "
            "ORDER BY ts DESC", (team,)).fetchall()
    finally:
        c.close()
    return [{"url": u, "rubric": r, "score": s, "max": m, "ts": ts}
            for u, r, s, m, ts in rows]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
