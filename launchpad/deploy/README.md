# Deploying a winner app

How a contestant service goes from repo to live endpoint on the
dataserver host. Same pattern `utsa-template.service` uses for the
reference app.

## What the app must expose

FastAPI on `127.0.0.1`, the five judged endpoints:

- `GET /health` → `200`
- `POST /portfolio/holdings`
- `POST /backtest`
- `POST /screen`
- `POST /asof` (if implemented)
- plus `POST /decisions` if the team shipped the adaptive format

The service reads data through the SDK (`statevector.Dataset`) —
it never touches panel files directly and never gets a Massive key.

## Steps

1. **Clone + venv** on the host:

   ```bash
   git clone <repo-url> ~/workspace/winners/<team>
   cd ~/workspace/winners/<team>
   ~/.venv-app/bin/pip install -r requirements.txt   # or reuse the
                                                    # shared sdk venv
   ```

2. **Grant a data token.** Append `<token>,<team>` to the
   dataserver token file — it reloads on mtime, no restart needed.
   Contestant tokens get the standard dataset scope; internal keys
   stay separate.

3. **Env file** `/etc/…/utsa-<team>.env` (mode 600):

   ```
   SV_DATA_ROOT=http://127.0.0.1:8000
   SV_DATA_TOKEN=<token>
   ```

4. **systemd unit** `/etc/systemd/system/utsa-<team>.service` —
   copy `utsa-template.service`, change three lines:
   `WorkingDirectory`, `EnvironmentFile`, `--port <next free>`.

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now utsa-<team>
   ```

5. **Caddy route** if the app gets a public URL — one site block
   per team subdomain, TLS automatic.

6. **Verify**: `curl 127.0.0.1:<port>/health` → 200, then the
   rubric `check.py` against the bound port.

## Constraints

- `NoNewPrivileges` + `PrivateTmp` on every unit — read-only data
  client, no write surface.
- Port map lives in this file's sibling `ports.tsv` — take the
  next free, never reuse.
- A team's token stays live 30 days post-contest unless renewed.
