# train-rag

## Git workflow

- `main`: production. Only the user deploys to it, manually, after testing on `dev`.
- `dev`: integration branch. Every PR targets `dev`, never `main`.
- `feature-<name>`: one branch per feature, created from `dev`.

Rules for Claude:

- Do feature work on a `feature-<name>` branch branched from the latest `dev`.
- Open a PR from the feature branch into `dev` with `gh pr create --base dev`.
- Never merge a PR, never push to `dev` or `main` directly, and never open a PR into `main`.
- Commit or push only when the user asks; when they ask to review first, leave changes
  uncommitted.

## Checks

```bash
uv run ruff check . && uv run mypy src tests && uv run pytest
```
