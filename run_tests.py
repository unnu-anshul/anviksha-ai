import json
import sys
import time
from pathlib import Path

from agent.graph import build_graph

EXPECTED = json.loads(Path("tests/expected.json").read_text())
Path("runs").mkdir(exist_ok=True)


def evaluate(report, rule):
    got = {c["status"] for c in report["claims"]}
    ok_have = any(s in got for s in rule["must_have_any"])
    ok_not = not any(s in got for s in rule["must_not_have"])
    return ok_have and ok_not, sorted(got)


def main():
    graph = build_graph()
    lines = ["# ANVIKSHA - Test Run Summary", "",
             "| Run | Input | Expected | Actual statuses | Result |", "|---|---|---|---|---|"]
    for n, (name, rule) in enumerate(EXPECTED.items(), start=1):
        pdf = Path("samples") / name
        t0 = time.time()
        try:
            final = graph.invoke({"pdf_name": name, "pdf_bytes": pdf.read_bytes(), "events": []})
            report = final["report"]
            passed, got = evaluate(report, rule)
        except Exception as e:           # a crash is a result too - record it honestly
            report, passed, got = {"error": str(e)}, False, ["CRASH"]
        report["elapsed_seconds"] = round(time.time() - t0, 1)
        out = Path(f"runs/RUN-{n}_{Path(name).stem}.json")
        out.write_text(json.dumps(report, indent=2, default=str))
        expected = f"has any of {rule['must_have_any']}; none of {rule['must_not_have']}"
        lines.append(f"| RUN-{n} | {name} | {expected} | {', '.join(got)} | "
                     f"{'PASS' if passed else 'DIFFERENCE - explain in README'} |")
        print(f"RUN-{n} {name}: {'PASS' if passed else 'DIFF'} {got}")
    Path("runs/RUN_SUMMARY.md").write_text("\n".join(lines))
    print("Summary written to runs/RUN_SUMMARY.md")


if __name__ == "__main__":
    sys.exit(main())
