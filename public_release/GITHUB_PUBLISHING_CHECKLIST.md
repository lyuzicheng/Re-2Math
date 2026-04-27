# GitHub Publishing Checklist

Use this checklist before pushing the public release to an anonymous GitHub repository.

## Recommended publish unit

- Publish the contents of `public_release/` as the repository root.
- Do not publish the full local research workspace.

## Already checked in this release layer

- no machine-specific absolute paths in public-facing text artifacts
- no raw API keys
- no downloaded PDF caches
- no raw search shard outputs
- final aggregated tables included
- no human-audit artifacts in the public release layer

## Still worth deciding manually

1. Choose a repository license.
   - `LICENSE` is intentionally not auto-generated here because the correct choice depends on your code/data release policy.
2. Add a citation file if desired.
   - Recommended filename: `CITATION.cff`
3. Decide whether to make the repository public immediately or keep it private until submission timing allows.

## Suggested local publish flow

```bash
cd public_release
git init
git add .
git commit -m "Initial anonymous benchmark release"
git branch -M main
git remote add origin <anonymous-github-repo-url>
git push -u origin main
```

## Final spot checks before push

- open `README.md`
- open `data/README.md`
- open `results/paper_main_tables.md`
- search once more for institution names or personal paths
- confirm the repository name itself is anonymous
