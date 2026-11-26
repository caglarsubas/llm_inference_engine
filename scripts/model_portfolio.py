"""Browse the model portfolio offline or download one pinned Hugging Face snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from inference_engine.portfolio import PortfolioVariant, load_portfolio


def download_variant(variant: PortfolioVariant, cache_dir: Path | None) -> list[Path]:
    paths = []
    for artifact in variant.files:
        for repair in (False, True):
            path = Path(hf_hub_download(
                repo_id=variant.repo_id, revision=variant.revision,
                filename=artifact.path, cache_dir=cache_dir, force_download=repair,
            ))
            if path.stat().st_size == artifact.size_bytes:
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() == artifact.sha256:
                        break
        else:
            raise OSError(f"File size or SHA-256 mismatch after re-downloading {path}")
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("list", "variants", "download"), default="list")
    parser.add_argument("repo_id", nargs="?", help="Exact case-sensitive model repository ID")
    parser.add_argument("--task", help="List only entries for this task")
    parser.add_argument("--json", action="store_true", help="Print the portfolio as JSON")
    parser.add_argument("--cache-dir", type=Path, help="Override the Hugging Face download cache")
    parser.add_argument("--variant", help="Exact quantization to download from a distribution repository")
    args = parser.parse_args(argv)
    catalog = load_portfolio()
    if args.command == "download":
        if not args.repo_id or args.task or args.json:
            parser.error("download requires one repository ID and accepts --cache-dir and --variant")
        choices = [v for e in catalog.entries for v in e.variants if v.repo_id == args.repo_id]
        entry = next((item for item in catalog.entries if item.id == args.repo_id), None)
        if choices and (args.variant is not None or entry is None):
            variant = next((v for v in choices if v.quantization == args.variant), None)
            if variant is None:
                parser.error("choose --variant from: " + ", ".join(v.quantization for v in choices))
            for path in download_variant(variant, args.cache_dir):
                print(path)
            return 0
        if args.variant:
            parser.error("--variant requires an exact distribution repository ID")
        if entry is None:
            parser.error(f"unknown portfolio ID: {args.repo_id}")
        patterns = entry.artifact_files
        if patterns is not None and not patterns:
            parser.error(f"{args.repo_id} has an empty artifact_files list")
        path = snapshot_download(
            repo_id=entry.id,
            revision=entry.revision,
            allow_patterns=patterns,
            cache_dir=args.cache_dir,
        )
        print(path)
        return 0

    if args.cache_dir or args.variant:
        parser.error("--cache-dir and --variant apply only to download")
    entries = catalog.entries
    if args.task:
        entries = [entry for entry in entries if args.task in entry.tasks]
        if not entries:
            tasks = sorted({task for entry in catalog.entries for task in entry.tasks})
            parser.error(f"unknown task: {args.task}. Available tasks: {', '.join(tasks)}")
    if args.command == "variants":
        if args.repo_id:
            if not any(entry.id == args.repo_id for entry in catalog.entries):
                parser.error(f"unknown parent ID: {args.repo_id}")
            entries = [entry for entry in entries if entry.id == args.repo_id]
        variants = [(entry.id, variant) for entry in entries for variant in entry.variants]
        if args.json:
            print(json.dumps([
                {"parent_id": parent, **variant.model_dump()} for parent, variant in variants
            ], indent=2))
        else:
            print("Parent\tDistribution\tFormat\tQuantization\tDownload (GB)\tRepo downloads (30d)")
            for parent, variant in variants:
                downloads = f"{variant.downloads_30d:,}" if variant.downloads_30d is not None else "—"
                print(f"{parent}\t{variant.repo_id}\t{variant.format}\t{variant.quantization}"
                      f"\t{variant.size_bytes / 1e9:.2f}\t{downloads}")
        return 0
    if args.repo_id:
        parser.error("repo_id applies only to variants or download")
    if args.json:
        subset = catalog.model_copy(update={"entries": entries})
        print(subset.model_dump_json(indent=2, exclude_none=True, exclude_unset=True))
    else:
        print(f"Portfolio reviewed {catalog.reviewed_on}")
        if catalog.popularity_observed_on:
            print(f"30-day Hugging Face downloads observed {catalog.popularity_observed_on}")
        print("Repository\tFamily\tSize\tTasks\tLicense\tDownloads (30d)")
        for entry in entries:
            row = (
                entry.id, entry.family, entry.size, ", ".join(entry.tasks),
                entry.license, f"{entry.downloads_30d:,}" if entry.downloads_30d is not None else "—",
            )
            print("\t".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
