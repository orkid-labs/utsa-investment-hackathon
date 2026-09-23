"""Judges dashboard — run the public rubric against a team's endpoint
and see the scorecard. Operator-side tool; the checks it runs are the
public ones in launchpad/rubric/check.py.

Run:
    export SV_DATA_ROOT=<dataset root or https://dataserver>
    export SV_DATA_TOKEN=<token>            # if remote
    uvicorn app:app --port 9000

Then open http://localhost:9000 — paste a team's endpoint URL, hit
Score, get the per-check breakdown.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

CHECK = Path(__file__).parent.parent / "rubric" / "check.py"
RUBRIC = Path(__file__).parent.parent / "rubric" / "rubric.yaml"

app = FastAPI(title="hackathon judges panel", version="0.1.0")

# session scoreboard — team name -> latest result
BOARD: dict[str, dict] = {}

PAGE = """<!doctype html><meta charset="utf-8"><title>judges panel</title>
<style>
 body{{font-family:ui-monospace,monospace;background:#0d1117;color:#e6edf3;max-width:960px;margin:2em auto;padding:0 1em}}
 input{{background:#161b22;color:#e6edf3;border:1px solid #30363d;padding:.5em;width:20em}}
 button{{background:#238636;color:#fff;border:0;padding:.55em 1.2em;cursor:pointer}}
 table{{border-collapse:collapse;width:100%;margin-top:1.5em}}
 td,th{{border:1px solid #30363d;padding:.4em .6em;text-align:left}}
 .ok{{color:#3fb950}} .bad{{color:#f85149}} .dim{{color:#8b949e}}
</style>
<h1>Judges panel <span class=dim>— public rubric</span></h1>
<form method=get action=/score>
 <input name=team placeholder="team name" required>
 <input name=url placeholder="https://team-endpoint" required style="width:26em">
 <button>Score</button>
</form>
{body}
"""

def scorecard(team: str, url: str, result: dict | None, err: str | None) -> str:
    if err:
        return f"<p class=bad>{team}: {err}</p>"
    rows = "".join(
        f"<tr><td>{c['id']}</td><td class={'ok' if c['earned']==c['max'] else 'bad'}>"
        f"{c['earned']:g}/{c['max']:g}</td><td class=dim>{c['note']}</td></tr>"
        for c in result["checks"])
    return (f"<h2>{team} — {result['score']:g}/{result['max']:g}</h2>"
            f"<table><tr><th>check</th><th>pts</th><th>note</th></tr>{rows}</table>")


def board_html() -> str:
    if not BOARD:
        return ""
    rows = "".join(
        f"<tr><td>{t}</td><td class=dim>{b['url']}</td>"
        f"<td class={'ok' if b['score']>=100 else ''}>{b['score']:g}/100</td>"
        f"<td class=dim>{b['when']}</td></tr>"
        for t, b in sorted(BOARD.items(), key=lambda kv: -kv[1]["score"]))
    return ("<h2>Scoreboard</h2><table><tr><th>team</th><th>endpoint</th>"
            f"<th>score</th><th>scored at</th></tr>{rows}</table>")


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE.format(body=board_html())


@app.get("/score", response_class=HTMLResponse)
def score(team: str = Query(...), url: str = Query(...)):
    t0 = time.time()
    out = subprocess.run(
        [sys.executable, str(CHECK), "--base-url", url],
        capture_output=True, text=True, timeout=180)
    if out.returncode != 0 or not out.stdout.strip().startswith("{"):
        body = f"<p class=bad>{team}: rubric failed — {out.stderr[-500:] or out.stdout[-500:]}</p>"
    else:
        result = json.loads(out.stdout)
        BOARD[team] = {"url": url, "score": result["score"],
                       "when": time.strftime("%H:%M:%S")}
        body = scorecard(team, url, result, None) + board_html()
    body += f"<p class=dim>rubric ran in {time.time()-t0:.1f}s</p>"
    return PAGE.format(body=body)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9000)
