import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path("anviksha.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def init():
    with _conn() as c:
        c.execute(
            "CREATE TABLE IF NOT EXISTS runs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, paper TEXT, created_at TEXT, "
            "summary TEXT, report TEXT)")


def save_run(report):
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO runs (paper, created_at, summary, report) VALUES (?,?,?,?)",
            (report["paper"], time.strftime("%Y-%m-%d %H:%M:%S"),
             json.dumps(report["summary"]), json.dumps(report, default=str)))
        return cur.lastrowid


def list_runs():
    with _conn() as c:
        return c.execute(
            "SELECT id, paper, created_at, summary FROM runs ORDER BY id DESC").fetchall()


def get_run(run_id):
    with _conn() as c:
        row = c.execute("SELECT report FROM runs WHERE id=?", (run_id,)).fetchone()
    return json.loads(row[0]) if row else None
