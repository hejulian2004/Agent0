"""Audit, deduplicate, and proportionally sample a fixed-size SFT set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from .dedup import record_hash
from .merge_sft import audit_record


def _read_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield row


def _rank(seed: int, source: str, digest: str) -> str:
    return hashlib.sha256(f"{seed}:{source}:{digest}".encode("utf-8")).hexdigest()


def _proportional_counts(available: Dict[str, int], target: int) -> Dict[str, int]:
    total = sum(available.values())
    if total < target:
        raise RuntimeError(f"Only {total} valid unique rows are available; required {target}")

    exact = {source: target * count / total for source, count in available.items()}
    selected = {source: min(available[source], int(value)) for source, value in exact.items()}
    remaining = target - sum(selected.values())
    order = sorted(
        available,
        key=lambda source: (exact[source] - selected[source], available[source], source),
        reverse=True,
    )
    while remaining:
        changed = False
        for source in order:
            if selected[source] >= available[source]:
                continue
            selected[source] += 1
            remaining -= 1
            changed = True
            if not remaining:
                break
        if not changed:
            raise RuntimeError("Could not allocate the requested proportional sample")
    return selected


def _mulberry_source(row: Dict[str, Any]) -> Optional[str]:
    """Return the normalized Mulberry sub-dataset encoded in an image path."""

    for image in row.get("images", []):
        if not isinstance(image, str):
            continue
        normalized = image.replace("\\", "/")
        marker = "/mulberry_images/"
        if marker not in normalized:
            continue
        suffix = normalized.split(marker, 1)[1]
        parts = [part for part in suffix.split("/") if part]
        if not parts:
            continue
        # Cauldron wraps the underlying benchmark one directory deeper.
        source = parts[1] if parts[0].casefold() == "cauldron" and len(parts) > 1 else parts[0]
        return f"mulberry:{source.casefold()}"
    return None


def _has_repair_prompt(row: Dict[str, Any]) -> bool:
    return any(
        message.get("role") == "user"
        and "Now switch to the Self-Repair role." in message.get("content", "")
        for message in row.get("messages", [])
        if isinstance(message, dict)
    )


def _normalize_form_feed(row: Dict[str, Any]) -> bool:
    """Convert OCR page-break characters in message text to ordinary newlines."""

    changed = False
    for message in row.get("messages", []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and "\f" in content:
            message["content"] = content.replace("\f", "\n")
            changed = True
    return changed


def _select_rows(
    buckets: Dict[str, list[tuple[str, Dict[str, Any]]]],
    target: int,
    seed: int,
    *,
    require_all_repairs: bool,
    ensure_source_coverage: bool,
) -> list[tuple[str, Dict[str, Any]]]:
    """Select a proportional sample after reserving required repairs/sources."""

    available_total = sum(len(rows) for rows in buckets.values())
    if available_total < target:
        raise RuntimeError(f"Only {available_total} valid unique rows are available; required {target}")

    selected_by_source: Dict[str, list[tuple[str, Dict[str, Any]]]] = {
        source: [] for source in buckets
    }
    selected_digests: set[str] = set()

    if require_all_repairs:
        for source, rows in buckets.items():
            for digest, row in rows:
                if _has_repair_prompt(row):
                    selected_by_source[source].append((digest, row))
                    selected_digests.add(digest)

    selected_count = sum(len(rows) for rows in selected_by_source.values())
    if selected_count > target:
        raise RuntimeError(
            f"Required repair rows ({selected_count}) exceed the requested sample size ({target})"
        )

    if ensure_source_coverage:
        for source in sorted(buckets):
            if selected_by_source[source]:
                continue
            candidates = [
                item for item in buckets[source] if item[0] not in selected_digests
            ]
            if not candidates:
                continue
            chosen = min(candidates, key=lambda item: _rank(seed, source, item[0]))
            selected_by_source[source].append(chosen)
            selected_digests.add(chosen[0])

    selected_count = sum(len(rows) for rows in selected_by_source.values())
    if selected_count > target:
        raise RuntimeError(
            f"Required rows and source coverage ({selected_count}) exceed sample size ({target})"
        )

    remaining_target = target - selected_count
    residual_available = {
        source: len(rows) - len(selected_by_source[source])
        for source, rows in buckets.items()
    }
    additional_counts = (
        _proportional_counts(residual_available, remaining_target)
        if remaining_target
        else {source: 0 for source in buckets}
    )

    for source, rows in buckets.items():
        candidates = [item for item in rows if item[0] not in selected_digests]
        candidates.sort(key=lambda item: _rank(seed, source, item[0]))
        chosen = candidates[: additional_counts[source]]
        selected_by_source[source].extend(chosen)
        selected_digests.update(digest for digest, _ in chosen)

    selected = [
        (source, digest, row)
        for source, rows in selected_by_source.items()
        for digest, row in rows
    ]
    selected.sort(key=lambda item: _rank(seed, item[0], item[1]))
    return [(source, row) for source, _, row in selected]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=int, choices=(1, 2))
    parser.add_argument("--input", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--allow-stage2-images", action="store_true")
    parser.add_argument(
        "--stratify-mulberry-sources",
        action="store_true",
        help="Group rows by the Mulberry sub-dataset encoded in their image path.",
    )
    parser.add_argument(
        "--ensure-source-coverage",
        action="store_true",
        help="Reserve at least one selected row from every non-empty source bucket.",
    )
    parser.add_argument(
        "--require-all-repairs",
        action="store_true",
        help="Include every valid row containing a Self-Repair prompt.",
    )
    parser.add_argument(
        "--normalize-form-feed",
        action="store_true",
        help="Normalize OCR form-feed characters in message text to newlines before audit.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.count <= 0:
        raise SystemExit("--count must be positive")

    buckets: Dict[str, list[tuple[str, Dict[str, Any]]]] = {}
    rejected: Dict[str, Dict[str, int]] = {}
    duplicate_rows = 0
    form_feed_normalized_rows = 0
    seen: set[str] = set()
    input_rows = 0

    for path in args.input:
        source = str(path)
        source_rejected = rejected.setdefault(source, {})
        for row in _read_rows(path):
            input_rows += 1
            if args.normalize_form_feed and _normalize_form_feed(row):
                form_feed_normalized_rows += 1
            reason = audit_record(
                row,
                args.stage,
                allow_stage2_images=args.allow_stage2_images,
            )
            if reason is not None:
                source_rejected[reason] = source_rejected.get(reason, 0) + 1
                continue
            digest = record_hash(row)
            if digest in seen:
                duplicate_rows += 1
                continue
            seen.add(digest)
            source_bucket = (
                _mulberry_source(row) if args.stratify_mulberry_sources else None
            ) or source
            buckets.setdefault(source_bucket, []).append((digest, row))

    available = {source: len(rows) for source, rows in buckets.items()}
    repair_rows_available = sum(
        _has_repair_prompt(row) for rows in buckets.values() for _, row in rows
    )
    selected = _select_rows(
        buckets,
        args.count,
        args.seed,
        require_all_repairs=args.require_all_repairs,
        ensure_source_coverage=args.ensure_source_coverage,
    )
    selected_counts: Dict[str, int] = {}
    for source, _ in selected:
        selected_counts[source] = selected_counts.get(source, 0) + 1
    selected_repair_rows = sum(_has_repair_prompt(row) for _, row in selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for _, row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_path = args.manifest or Path(str(args.output) + ".manifest.json")
    manifest = {
        "stage": args.stage,
        "inputs": [str(path) for path in args.input],
        "output": str(args.output),
        "input_rows": input_rows,
        "valid_unique_pool": sum(available.values()),
        "available_rows_by_source": available,
        "selected_rows": len(selected),
        "selected_rows_by_source": selected_counts,
        "target_rows": args.count,
        "seed": args.seed,
        "allow_stage2_images": args.allow_stage2_images,
        "mulberry_sources_stratified": args.stratify_mulberry_sources,
        "source_coverage_enforced": args.ensure_source_coverage,
        "all_repairs_required": args.require_all_repairs,
        "form_feed_normalization_enabled": args.normalize_form_feed,
        "form_feed_normalized_rows": form_feed_normalized_rows,
        "available_repair_rows": repair_rows_available,
        "selected_repair_rows": selected_repair_rows,
        "duplicate_rows": duplicate_rows,
        "rejected_by_source": rejected,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
