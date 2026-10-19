import json

import pandas as pd
import streamlit as st

import db
from agent.graph import build_graph
from agent.schemas import STATUS_LABELS

st.set_page_config(page_title="ANVIKSHA.ai", layout="wide")
db.init()

st.title("ANVIKSHA.ai")
st.caption("Question the Claim. Test the Evidence. - Autonomous Research Verification Agent")

BOX = {"reproduced": st.success, "discrepancy": st.error,
       "not_checkable": st.info, "unverified": st.warning}


def show_report(rep, scope):
    s = rep["summary"]
    cols = st.columns(4)
    for col, key in zip(cols, STATUS_LABELS):
        col.metric(STATUS_LABELS[key], s.get(key, 0))

    choice = st.selectbox("Filter claims", ["All"] + list(STATUS_LABELS.values()),
                          key=f"{scope}_filter")
    for c in rep["claims"]:
        if choice != "All" and c["status_label"] != choice:
            continue
        title = f"{c['id']} | {c['status_label']} | {c['text'][:90]}"
        with st.expander(title):
            BOX[c["status"]](c["status_label"])
            st.markdown(f"**Original claim:** {c['text']}")
            st.markdown(f"**Type:** {c['claim_type']}   **Where:** {c['location'] or 'n/a'}"
                        f"   **Page:** {c['page']}")
            st.markdown(f"**Agent decision:** {c['classify_reason']}")
            if c["plan"]:
                st.markdown(f"**Verification method:** `{c['plan']['method']}` - "
                            f"{c['plan']['reason']}")
                st.json(c["plan"]["params"])
            if c["result"]:
                r = c["result"]
                st.markdown(f"**Observed result:** {r['summary']}")
                st.dataframe(pd.DataFrame([{
                    "computed": r["computed"], "reported": r["reported"],
                    "difference": r["difference"], "tolerance": r["tolerance"],
                    "match": r["match"]}]), hide_index=True)
                st.markdown("**Values used (evidence):**")
                st.json(r["details"])
            st.markdown(f"**Explanation:** {c['explanation']}")
            if c["limitations"]:
                st.markdown("**Limitations:**")
                for x in c["limitations"]:
                    st.markdown(f"- {x}")
            if c["history"]:
                st.markdown("**Attempt history:**")
                st.json(c["history"])

    st.subheader("Evidence discovery")
    ev = rep.get("evidence", {})
    st.write("Title guess:", ev.get("title_guess"))
    for group in ("crossref", "openalex", "repos_in_paper", "github_search"):
        if ev.get(group):
            st.markdown(f"**{group}**")
            st.dataframe(pd.DataFrame(ev[group]), hide_index=True)
    for err in ev.get("errors", []):
        st.warning(err)

    st.subheader("Human approval gate (external code)")
    repos = ev.get("repos_in_paper", [])
    if not repos:
        st.write("No repository linked in the paper.")
    for i, repo in enumerate(repos):
        st.write(f"Repository: {repo['url']} (HTTP status: {repo['http_status']})")
        if st.checkbox("I approve running this repository's code", key=f"{scope}_ok{i}"):
            st.warning("Approval recorded, but the isolated sandbox is NOT implemented in this "
                       "prototype, so no external code was executed.")
        else:
            st.caption("Not approved. Nothing was executed.")

    st.subheader("Agent log")
    st.code("\n".join(rep.get("agent_log", [])))
    st.download_button("Download report (JSON)", json.dumps(rep, indent=2, default=str),
                       file_name=f"anviksha_run_{rep.get('run_id')}.json", key=f"{scope}_dl")
    st.subheader("Limitations of this prototype")
    for x in rep.get("global_limitations", []):
        st.markdown(f"- {x}")


tab_run, tab_hist = st.tabs(["Verify a paper", "Past runs"])

with tab_run:
    up = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])
    if up and st.button("Run verification", type="primary"):
        graph = build_graph()
        init = {"pdf_name": up.name, "pdf_bytes": up.getvalue(), "events": []}
        final = None
        with st.status("Agent is working...", expanded=True) as status:
            for chunk in graph.stream(init, stream_mode="updates"):
                for node, upd in chunk.items():
                    st.write(f"**{node}**")
                    for line in upd.get("events", []):
                        st.write(f"- {line}")
                    if "report" in upd:
                        final = upd["report"]
            status.update(label="Verification complete", state="complete")
        st.session_state["report"] = final
    if st.session_state.get("report"):
        show_report(st.session_state["report"], "new")

with tab_hist:
    rows = db.list_runs()
    if not rows:
        st.write("No runs yet.")
    else:
        labels = {f"#{r[0]} - {r[1]} - {r[2]}": r[0] for r in rows}
        pick = st.selectbox("Choose a run", list(labels))
        show_report(db.get_run(labels[pick]), "hist")
