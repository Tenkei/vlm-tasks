"""Aggregate saved usage without depending on an inference provider."""
from decimal import Decimal


TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
                "output_reasoning_tokens")
COUNT_FIELDS = ("attempted_requests", "measured_requests", "unmeasured_requests",
                "total_tokens", "costed_requests", *TOKEN_FIELDS)


def check_journal_usage(calls, usage, *, task):
    """Reconcile recorded counts and estimates while allowing unavailable costs."""
    records = [generation for call in calls for generation in call["generations"]]
    attempted = sum(call.get("attempted_requests", max(1, len(call["generations"]))) for call in calls)
    unmeasured = attempted - len(records) + sum(row.get("unmeasured_requests", 0) for row in records)
    expected = {field: sum(row.get(field, 0) for row in records) for field in TOKEN_FIELDS}
    costs = [Decimal(row["estimated_cost_usd"]) for row in records if row.get("estimated_cost_usd") is not None]
    expected.update(attempted_requests=attempted, measured_requests=attempted - unmeasured,
                    unmeasured_requests=unmeasured, costed_requests=len(costs),
                    total_tokens=expected["input_tokens"] + expected["output_tokens"])
    if any(usage[field] != value for field, value in expected.items() if field in usage):
        raise ValueError(f"Segment usage does not match its request journal: {task}")
    cost = usage.get("estimated_cost_usd")
    if (cost is None) != (not costs) or (cost is not None and Decimal(cost) != sum(costs, Decimal(0))):
        raise ValueError(f"Segment cost does not match its request journal: {task}")
    if usage.get("estimated_cost_complete") and (len(costs) != attempted or unmeasured):
        raise ValueError(f"Segment claims complete costs with missing usage: {task}")


def combine_usage(summaries):
    """Sum totals, retaining unknown costs and incomplete coverage explicitly."""
    result = {}
    for field in COUNT_FIELDS:
        if any(field in summary for summary in summaries):
            values = [summary.get(field) for summary in summaries]
            result[field] = sum(values) if all(value is not None for value in values) else None
    for field in ("estimated_cost_usd", "reported_cost_usd"):
        values = [summary.get(field) for summary in summaries]
        known = [Decimal(value) for value in values if value is not None]
        # An invoiced total is unavailable unless every part reports it.
        available = known and (field == "estimated_cost_usd" or len(known) == len(values))
        result[field] = str(sum(known, Decimal(0))) if available else None
    if any("estimated_cost_complete" in summary for summary in summaries):
        result["estimated_cost_complete"] = all(
            summary.get("estimated_cost_complete", False) and summary.get("estimated_cost_usd") is not None
            for summary in summaries)
    elif any(summary.get("estimated_cost_usd") is not None for summary in summaries):
        result["estimated_cost_complete"] = False
    if any("cost_sources" in summary for summary in summaries):
        result["cost_sources"] = sorted({source for summary in summaries for source in summary.get("cost_sources", [])})
    result["cost_note"] = "; ".join(dict.fromkeys(summary["cost_note"] for summary in summaries if summary.get("cost_note")))
    return result
