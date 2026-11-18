#!/usr/bin/env python3
"""Safely prompt and store judge API credentials in workspace .env (chmod 0600)."""
from __future__ import annotations

import argparse
from getpass import getpass
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT / ".env"
DEFAULT_BASE_URL = "https://api.site/v1"
DEFAULT_MODEL = "gpt-6.1-terra"


def load_env(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        data[k.strip()] = v.strip().strip('"').strip("'")
    return data


def save_env(path: Path, data: dict[str, str]) -> None:
    lines = [f"{k}={v}" for k, v in data.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def verify_endpoint(base_url: str, api_key: str) -> None:
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        print(f"Endpoint check OK: HTTP {resp.status} ({url})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--verify", action="store_true", help="Verify /models connectivity without printing the key")
    args = ap.parse_args()

    env = load_env(ENV_PATH)
    if args.verify and env.get("JUDGE_API_KEY"):
        verify_endpoint(env.get("JUDGE_BASE_URL", args.base_url), env["JUDGE_API_KEY"])
        return

    print(f"Target .env : {ENV_PATH}")
    print(f"Base URL    : {args.base_url}")
    print(f"Judge Model : {args.model}")
    token = getpass("Enter API token (input hidden): ").strip()
    if not token:
        raise SystemExit("Aborted: empty token.")

    env["JUDGE_BASE_URL"] = args.base_url
    env["JUDGE_MODEL"] = args.model
    env["JUDGE_API_KEY"] = token
    env["OPENAI_API_BASE"] = args.base_url
    env["OPENAI_API_KEY"] = token
    save_env(ENV_PATH, env)
    print(f"Saved credentials to {ENV_PATH} (permissions 0600). Token not echoed.")

    if args.verify:
        verify_endpoint(args.base_url, token)


if __name__ == "__main__":
    main()
