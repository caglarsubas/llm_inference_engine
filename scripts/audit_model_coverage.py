"""Audit popular Hub releases, direct quantizations and candidate distillations."""

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Iterator
from datetime import date, datetime, timezone

import httpx

from inference_engine.portfolio import load_portfolio

# Some publishers host unrelated families. Mistral includes its specialist brands.
PUBLISHERS = {
    "qwen": ("Qwen", "qwen"),
    "deepseek-ai": ("DeepSeek", "deepseek"),
    "moonshotai": ("Kimi", "kimi"),
    "zai-org": ("GLM", "glm"),
    "thudm": ("GLM", "glm"),
    "meta-llama": ("Llama", "llama"),
    "meta-models": ("Muse", "muse"),
    "nvidia": ("Nemotron", "nemotron"),
    "google": ("Gemma", "gemma"),
    "mistralai": ("Mistral", ""),
    "minimaxai": ("MiniMax", "minimax"),
    "ibm-granite": ("Granite", "granite"),
}


def family_for(repo_id: str) -> str | None:
    org, _, name = repo_id.casefold().partition("/")
    publisher = PUBLISHERS.get(org)
    if name and publisher and publisher[1] in name:
        return publisher[0]
    return None


def popular_repos(client: httpx.Client, minimum: int) -> Iterator[dict]:
    url = "https://huggingface.co/api/models"
    params = {"sort": "downloads", "direction": "-1", "limit": "500",
              "expand": ["downloads", "createdAt", "tags"]}
    while url:
        response = client.get(url, params=params)
        response.raise_for_status()
        for row in response.json():
            if row["downloads"] < minimum:
                return
            yield row
        url = response.links.get("next", {}).get("url")
        params = None  # The Hub supplies the complete cursor URL.


def audit(rows: Iterable[dict], catalog_ids: set[str], since: date, until: date) -> tuple[list, list]:
    covered = {repo.casefold() for repo in catalog_ids}
    results = {}
    distillations = {}
    for row in rows:
        created = date.fromisoformat(row["createdAt"][:10])
        if not since <= created <= until:
            continue
        repo_id = row["id"]
        common = {
            "id": repo_id, "created": created.isoformat(), "downloads_30d": row["downloads"],
            "covered": repo_id.casefold() in covered,
        }
        tags = row.get("tags", [])
        if "distill" in repo_id.partition("/")[2].casefold():
            # Distillations often have an intermediate community parent. A name match
            # is a review candidate; the publisher's card establishes training lineage.
            parents = {tag.removeprefix("base_model:") for tag in tags
                       if tag.startswith("base_model:") and tag.count(":") == 1}
            distillations[repo_id.casefold()] = {**common, "parents": sorted(parents)}
        family = family_for(repo_id)
        parents = defaultdict(list)
        if family:
            parents[family] = []
        else:
            for tag in tags:
                if tag.startswith("base_model:quantized:"):
                    parent = tag.removeprefix("base_model:quantized:")
                    if parent_family := family_for(parent):
                        parents[parent_family].append(parent)
        for name, upstream in parents.items():
            results[(name, repo_id.casefold())] = {
                "family": name, "kind": "quantization" if upstream else "publisher",
                **common, "parents": sorted(set(upstream)),
            }
    return (
        sorted(results.values(), key=lambda r: (-r["downloads_30d"], r["id"], r["family"])),
        sorted(distillations.values(), key=lambda r: (-r["downloads_30d"], r["id"])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", type=date.fromisoformat, required=True,
                        help="Earliest UTC repository creation date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--until", type=date.fromisoformat, default=datetime.now(timezone.utc).date(),
                        help="Last UTC repository creation date, inclusive (default: today)")
    parser.add_argument("--min-downloads", type=int, default=100_000,
                        help="Minimum current trailing 30-day downloads (default: 100000)")
    parser.add_argument("--json", action="store_true", help="Print every matching repository")
    args = parser.parse_args()
    if args.since > args.until or args.min_downloads < 1:
        parser.error("use an ordered date range and a positive download threshold")

    entries = load_portfolio().entries
    ids = {e.id for e in entries} | {v.repo_id for e in entries for v in e.variants}
    with httpx.Client(timeout=45, follow_redirects=True) as client:
        rows, distillations = audit(popular_repos(client, args.min_downloads), ids,
                                    args.since, args.until)
    if args.json:
        print(json.dumps({"since": str(args.since), "until": str(args.until),
                          "observed_on": str(datetime.now(timezone.utc).date()),
                          "min_downloads": args.min_downloads, "repositories": rows,
                          "distillation_candidates": distillations}, indent=2))
        return
    print(f"Repository creation window {args.since} through {args.until} (UTC)")
    print(f"Current trailing 30-day downloads >= {args.min_downloads:,}")
    print(f"{'Family':<12} {'Publisher IDs':>15} {'Direct quant IDs':>18}   (covered / found)")
    for name in dict.fromkeys(f for f, _ in PUBLISHERS.values()):
        counts = []
        for kind in ("publisher", "quantization"):
            group = [r for r in rows if r["family"] == name and r["kind"] == kind]
            counts.append(f"{sum(r['covered'] for r in group)} / {len(group)}")
        print(f"{name:<12} {counts[0]:>15} {counts[1]:>18}")
    print("\nHighest-download missing repositories (use --json for the complete list)")
    for row in [r for r in rows if not r["covered"]][:12]:
        print(f"{row['downloads_30d']:>12,}  {row['kind']:<12}  {row['id']}")
    matched = sum(row["covered"] for row in distillations)
    print(f"\nDistillation candidates by repository name: {matched} / {len(distillations)} covered")
    for row in distillations:
        if not row["covered"]:
            print(f"{row['downloads_30d']:>12,}  {row['id']}")


if __name__ == "__main__":
    main()
