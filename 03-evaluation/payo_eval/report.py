"""Markdown reports written by workflow steps."""

from payo_llm.pilot import render_table
from payo_eval.annotation import Finalized


def finalization_report(result: Finalized, cfg: dict, accepted: bool) -> str:
    """The analysis plan §2 record: counts before and after merging, κ before and after, the gate."""
    labels = result.labels
    kept = labels[labels["exclude_flag"] != "yes"]
    lines = [
        "# Label finalization (analysis plan §2)",
        "",
        f"Posts: {len(labels)} · kept: {len(kept)} · excluded: {len(labels) - len(kept)} · "
        f"label sources: {labels['label_source'].value_counts().to_dict()}",
        "",
        "## Exclusions by reason",
        "",
        render_table(result.exclusions),
        "",
        f"## Emotion counts (merge threshold {cfg['labels']['min_label']})",
        "",
        render_table(result.counts),
        "",
        f"Merges applied at scoring time: {result.merges or 'none'}",
        "",
        "## Agreement on the double-annotated posts (R1)",
        "",
        render_table(result.field_kappa),
        "",
        f"One-vs-rest κ per emotion (gate {cfg['labels']['kappa_gate']}):",
        "",
        render_table(result.label_kappa),
        "",
        "## Gate",
        "",
    ]
    if result.gate_failures:
        lines += [f"- {f}" for f in result.gate_failures]
        lines += ["", "Accepted with --accept-kappa: log the reason in analysis-plan.md §11." if accepted
                  else "Not passed: revise and re-adjudicate, add a merge, or accept and log it."]
    else:
        lines.append("Passed.")
    return "\n".join(lines) + "\n"
