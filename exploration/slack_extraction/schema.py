"""The Slack extraction schema: fields and relations, with the descriptions the models see.

One source for the DeepSeek prompt, the GLiNER trial and the scoring, so they always ask for the
same fields. Drafted on 16 November from 12 review threads and the Slack benchmark questions;
extended the same day for a generative model (fields 20-26 and the last four relations;
"answers" was dropped on 17 November: DeepSeek forced it onto unrelated decisions), see
docs/todo/Understand how data could be bucketed by content.md.

Forms:
- name: a short name or id, copied word for word (matched word for word at search time);
- phrase: a few words, may be shortened from the text (matched by embeddings);
- sentence: one short sentence, may be rewritten (matched by embeddings);
- label: one value from a fixed list (CLASSES), or one or more (MULTI_CLASSES).

Extended again on 17 November after the smoke test: `concept` (generic terms that paraphrased
questions share with documents), `doc_date` (the corpus stores no document date) and the
multi-label `area` (the bucket key for filtering before search).
"""

# Field -> (form, description). Order is the order the prompt lists them.
FIELDS: dict[str, tuple[str, str]] = {
    "topic": ("phrase", "the main subject of the document, in at most 10 words"),
    "concept": (
        "phrase",
        "a general technical or business concept the document deals with, in 1-4 plain words "
        "(session affinity, log retention, rate limiting, expense reimbursement); not a "
        "product, feature or component name",
    ),
    "person": ("name", "a person's name or handle"),
    "team": ("name", "a team, group or department of the company"),
    "client": (
        "name",
        "a customer company, tenant or customer org (never Redwood Inference, our own company)",
    ),
    "region": ("name", "a cloud region, data centre or location"),
    "product": ("name", "something the company sells: a product, plan, tier or SKU"),
    "feature": (
        "name",
        "a capability users see or switch on inside a product (not the product, not an "
        "internal service)",
    ),
    "component": (
        "name",
        "an internal service, module, endpoint, pipeline or tool that users don't buy",
    ),
    "model": ("name", "an ML model name"),
    "version": ("name", "a version, release, build or variant"),
    "environment": ("name", "a deployment environment or stage (prod, staging, canary, ...)"),
    "bug": ("phrase", "a problem, error or failure, as its short symptom"),
    "cause": ("phrase", "the root cause of a problem"),
    "fix": ("phrase", "a fix or mitigation, taken or planned"),
    "config": ("name", "a configuration setting, parameter or feature flag"),
    "metric": ("name", "the name of a metric or alert (without its value)"),
    "identifier": ("name", "a ticket, incident, PR, request or order id"),
    "date_time": ("name", "a date, time, deadline or time window"),
    "doc_date": (
        "name",
        "the date the document itself was written or sent, if stated (one value at most; not "
        "the other dates it mentions)",
    ),
    "document": ("name", "a named document, RFC, template, runbook or dashboard"),
    "decision": ("sentence", "something the people agreed or decided"),
    "action_item": ("phrase", "a task someone takes on or is asked to do"),
    "question": ("sentence", "a main question the document asks"),
    "measurement": ("phrase", "a measured value with what it measures"),
    "impact": ("phrase", "who or what is affected by a problem, and how badly"),
}

# Label fields: exactly one value from the list.
CLASSES: dict[str, tuple[str, tuple[str, ...]]] = {
    "thread_kind": (
        "what kind of conversation or document this is",
        (
            "incident", "bug triage", "how-to", "design proposal", "release or deploy",
            "customer support", "planning", "HR or people", "social", "other",
        ),
    ),
    "status": (
        "how it ends",
        ("resolved", "mitigated", "in progress", "open", "no action", "not stated"),
    ),
}

# Multi-label fields: one or more values from the list (the bucket keys for filtering).
MULTI_CLASSES: dict[str, tuple[str, tuple[str, ...]]] = {
    "area": (
        "the business areas the document belongs to (1-3, most important first)",
        (
            "billing and pricing", "sales and contracts", "customer support",
            "security and compliance", "legal", "incidents and reliability",
            "performance and latency", "deploy and release", "infrastructure and capacity",
            "models and evaluation", "API and SDK", "product and UX",
            "docs and developer relations", "observability and logging", "data and storage",
            "finance and procurement", "HR and expenses", "office and social", "other",
        ),
    ),
}

# The span-like fields, for the GLiNER trial and the scoring (name: description).
ENTITY_TYPES: dict[str, str] = {name: description for name, (_, description) in FIELDS.items()}

# Relation -> (description, allowed head fields, allowed tail fields).
RELATIONS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "owns": (
        "a person owns or is assigned a fix or task",
        ("person",),
        ("fix", "action_item"),
    ),
    "reported": ("a person or client reported a problem", ("person", "client"), ("bug",)),
    "works_for": ("a person works for a customer company", ("person",), ("client",)),
    "member_of": ("a person is a member of a team", ("person",), ("team",)),
    "affects": (
        "a problem affects a component, feature, product or client",
        ("bug",),
        ("component", "feature", "product", "client"),
    ),
    "occurs_in": (
        "a problem occurs in a region or environment",
        ("bug",),
        ("region", "environment"),
    ),
    "caused_by": ("a problem is caused by a root cause", ("bug",), ("cause",)),
    "fixed_by": ("a problem is fixed or mitigated by a fix", ("bug",), ("fix",)),
    "changes": ("a fix changes a configuration setting or flag", ("fix",), ("config",)),
    "tracked_in": (
        "a problem, fix or task is tracked in a ticket or id",
        ("bug", "fix", "action_item"),
        ("identifier",),
    ),
    "part_of": (
        "a feature or component is part of a product",
        ("feature", "component"),
        ("product",),
    ),
    "decided_by": ("a decision was made by a person", ("decision",), ("person",)),
    "decision_about": (
        "a decision is about a feature, component or product",
        ("decision",),
        ("feature", "component", "product"),
    ),
    "due": ("a task or fix is due at a date or time", ("action_item", "fix"), ("date_time",)),
    "measures": ("a measurement is of a metric", ("measurement",), ("metric",)),
}
