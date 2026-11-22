"""The mined-field schema: field names and forms, label values and what each area holds.

A copy of the parts of `exploration/slack_extraction/schema.py` that search needs. The
exploration file stays the source of the extraction prompt (its hash, fc8ae8878988, keys the
DeepSeek labels and the SLM training data), so it is not changed; a test checks that the
two agree.

Forms: `name` (copied word for word), `phrase`, `sentence` (embedded); labels are one value
(`thread_kind`, `status`) or one to three (`area`) from fixed lists.
"""

# Field -> form, in the order the extraction prompt lists them.
FIELD_FORMS: dict[str, str] = {
    "topic": "phrase",
    "concept": "phrase",
    "person": "name",
    "team": "name",
    "client": "name",
    "region": "name",
    "product": "name",
    "feature": "name",
    "component": "name",
    "model": "name",
    "version": "name",
    "environment": "name",
    "bug": "phrase",
    "cause": "phrase",
    "fix": "phrase",
    "config": "name",
    "metric": "name",
    "identifier": "name",
    "date_time": "name",
    "doc_date": "name",
    "document": "name",
    "decision": "sentence",
    "action_item": "phrase",
    "question": "sentence",
    "measurement": "phrase",
    "impact": "phrase",
}
# Name fields that go into the names profile; the document's own date isn't a name.
NAME_FIELDS = [name for name, form in FIELD_FORMS.items() if form == "name" and name != "doc_date"]

THREAD_KINDS = (
    "incident", "bug triage", "how-to", "design proposal", "release or deploy",
    "customer support", "planning", "HR or people", "social", "other",
)  # fmt: skip
STATUSES = ("resolved", "mitigated", "in progress", "open", "no action", "not stated")

# What each area holds, for the agent (the extractor saw only the names).
AREA_GUIDE: dict[str, str] = {
    "billing and pricing": "invoices, charges, credits, refunds, plans and price changes",
    "sales and contracts": "deals, quotes, renewals, commitments, partner bundles, contract terms",
    "customer support": "tickets, customer replies, support macros and wording, escalations",
    "security and compliance": "security reviews, questionnaires, SOC 2, access, audits, PII",
    "legal": "contract clauses, DPAs, terms, legal review",
    "incidents and reliability": "outages, SEVs, errors, postmortems, on-call, failover",
    "performance and latency": "latency, throughput, p95/p99, batching, caching, regressions",
    "deploy and release": "rollouts, canaries, config changes, release plans, rollbacks",
    "infrastructure and capacity": "GPUs, clusters, regions, quotas, capacity planning, hosts",
    "models and evaluation": "model quality, benchmarks, evals, quantization, fine-tuning",
    "API and SDK": "endpoints, rate limits, client libraries, retries, API errors",
    "product and UX": "features, UI, onboarding flows, product decisions",
    "docs and developer relations": "documentation, tutorials, community, webinars, examples",
    "observability and logging": "metrics, dashboards, alerts, logs, tracing",
    "data and storage": "databases, retention, backups, data pipelines, storage",
    "finance and procurement": "budgets, vendors, purchase orders, spend",
    "HR and expenses": "hiring, people topics, expense claims, policies for staff",
    "office and social": "office life, events, social chat",
    "other": "anything that fits none of the above",
}
AREAS = tuple(AREA_GUIDE)

# Each source, as the agent reads it.
SOURCE_KINDS: dict[str, str] = {
    "slack": "chat thread",
    "gmail": "email",
    "linear": "ticket (Linear)",
    "jira": "ticket (Jira)",
    "github": "pull request",
    "confluence": "wiki page",
    "google_drive": "shared doc or sheet",
    "fireflies": "meeting transcript",
    "hubspot": "CRM record",
}
