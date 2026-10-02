"""Local balanced construction; shared inputs are always read-only."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime

from agent0_protocol.schema import CanonicalTrajectory, ProtocolError
from agent0_protocol.local_prompts import render_system_prompt
from tools.local_profile import load_config
from tools.data_builder.sft_stream import run_stream, _prepare, _write_state
from tools.data_builder.sft_quality import content_hash, verify_sft_semantics

QUOTAS = dict(geometry3k=112, geoqa=111, mulberry=111, llava_ov_image=111,
              mm_rlhf=111, smr=111, mmeureka=111, retool=111, arxivqa=111)


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare(root, config):
    """Reuse selected IDs/order/images/references, normalize only in memory."""
    shared = Path(config["assets"]["source_root"]) / "data/sft/balanced_1000_v2"
    destination = root / config["local_data"]["sft_root"]
    if (destination / "manifest.json").exists():
        raise ProtocolError("Candidates already prepared; use preflight/resume")
    candidates = {}
    hashes = {}
    for source, quota in QUOTAS.items():
        path = shared / f"{source}_candidates.jsonl"
        candidates[source] = read_rows(path)
        if len(candidates[source]) < quota:
            raise ProtocolError(f"Insufficient candidates: {source}")
        hashes[source] = hashlib.sha256(path.read_bytes()).hexdigest()
        for task in candidates[source]:
            normalized = _prepare(task)
            if not normalized.get("question") or not str(normalized.get("ground_truth", "")).strip():
                raise ProtocolError(f"Missing question/reference: {source}")
            for image in task.get("images", []):
                if not Path(image).is_file():
                    raise FileNotFoundError(image)
    destination.mkdir(parents=True, exist_ok=True)
    for source, tasks in candidates.items():
        path = destination / f"{source}_candidates.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tasks))
    shared_manifest = json.loads((shared / "manifest.json").read_text())
    _write_state(destination / "manifest.json", {
        "quotas": QUOTAS, "seed": shared_manifest["seed"], "shared_candidates": str(shared),
        "input_hashes": hashes, "candidate_rows": {s: len(t) for s, t in candidates.items()},
        "protocol": "agent0.responses.v1", "normalization": "read-time; source_dataset rules",
    })


def preflight(root, config):
    source = Path(config["assets"]["source_root"])
    local = root / config["local_data"]["sft_root"]
    candidates = local if (local / "manifest.json").exists() else source / "data/sft/balanced_1000_v2"
    counts = {s: len(read_rows(candidates / f"{s}_candidates.jsonl")) for s in QUOTAS}
    if any(counts[s] < QUOTAS[s] for s in QUOTAS):
        raise ProtocolError("Insufficient SFT candidates")
    required = [Path(config["assets"]["teacher_model"]), Path(config["assets"]["base_model"]),
                Path(config["assets"]["validation"]),
                source / "data/processed/global_partition_v1/assigned_rl.jsonl"]
    from tools.data_builder.multisource_selection import EVAL_TRAIN_PATHS
    required.extend(source / p for p in EVAL_TRAIN_PATHS.values())
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing read-only inputs: " + ", ".join(missing))
    from tools.sft_environment import readiness
    sft_environment = readiness(root)
    result = {"quotas": QUOTAS, "candidate_rows": counts, "prepared": candidates == local,
            "sft_python_ready": sft_environment["ready"], "sft_environment": sft_environment,
            "prompt_sha256": hashlib.sha256(render_system_prompt().encode()).hexdigest()}
    if 'checkpointed' in config:
        from agent0_protocol.checkpointed import MODES, role_prompt, validate_config
        validate_config(config['checkpointed'])
        result['protocol_version'] = config['checkpointed']['protocol_version']
        result['role_prompt_sha256'] = {mode: hashlib.sha256(role_prompt(mode).encode()).hexdigest() for mode in MODES}
    return result


def reset(root, config):
    """Explicit archival only; fail if any owned stream is still locked."""
    local = root / config["local_data"]["sft_root"]
    streams = [local / f"{s}_generated.jsonl" for s in QUOTAS]
    locks = []
    try:
        for stream in streams:
            stream.parent.mkdir(parents=True, exist_ok=True)
            handle = Path(str(stream) + ".lock").open("a")
            locks.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths = [p for stream in streams for p in (stream, Path(str(stream) + ".state.json")) if p.exists()]
        final = root / config["local_data"]["sft_output"]
        paths.extend(p for p in (final, final.with_suffix(".manifest.json")) if p.exists())
        if 'checkpointed' in config:
            paths.extend(p for p in (final.parent / 'roles', root / config['checkpointed']['audit_root']) if p.exists())
        if not paths:
            return None
        archive = root / "logs/data_reset" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        archive.mkdir(parents=True)
        for path in paths:
            target = archive / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            path.rename(target)
        _write_state(archive / "inventory.json", {"archived": [str(p.relative_to(root)) for p in paths],
                                                  "status": "archived", "candidates_retained": True})
        (archive / "README.md").write_text("Explicit local generation reset. Candidates, inputs and tool images retained. See inventory.json.\n")
        return str(archive)
    except BlockingIOError as exc:
        raise ProtocolError("Stop the active builder before resetting generation") from exc
    finally:
        for handle in locks:
            handle.close()


def build_sft(root, config):
    from agent0_protocol.responses_runtime import ResponsesConfig, ResponsesRuntime
    from scripts.build_sft_dataset import SFTTrajectoryBuilder
    local = root / config["local_data"]["sft_root"]
    if not (local / "manifest.json").is_file():
        raise ProtocolError("Run prepare-balanced explicitly before first generation")
    preflight(root, config)
    runtime = ResponsesRuntime(ResponsesConfig.from_env(), probe_on_init=True)
    builder = SFTTrajectoryBuilder()
    merged, proofs, seen = [], set(), set()
    try:
        for source, quota in QUOTAS.items():
            stream = local / f"{source}_generated.jsonl"
            trajectories, stats = run_stream(read_rows(local / f"{source}_candidates.jsonl"), runtime,
                builder, stream, concurrency=config["sft_data_generation"]["concurrency"],
                min_steps=config["sft_data_generation"]["min_steps"], target_exported=quota)
            if len(trajectories) != quota:
                raise ProtocolError(f"{source}: accepted {len(trajectories)}/{quota}; exhausted candidates")
            proofs.update(builder.answer_judge_hashes)
            for trajectory in trajectories:
                digest = trajectory.metadata["sample_hash"]
                if digest in seen:
                    raise ProtocolError("Duplicate sample across SFT sources")
                seen.add(digest)
                merged.append({"trajectory_id": trajectory.trajectory_id, "data_source": source,
                               "trajectory": trajectory.to_dict()})
            print(json.dumps({"source": source, "accepted": quota, "stats": stats.to_dict()}), flush=True)
        for row in merged:
            audit = verify_sft_semantics(CanonicalTrajectory.from_dict(row["trajectory"]), builder.registry, proofs)
            if not audit.valid:
                raise ProtocolError("Final SFT audit: " + "; ".join(audit.issues))
        output = root / config["local_data"]["sft_output"]
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
        _write_state(output.with_suffix(".manifest.json"), {"rows": len(merged), "quotas": QUOTAS,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "answer_judge_accepted_hashes": sorted(proofs), "protocol":
                config.get('checkpointed', {}).get('protocol_version', 'agent0.responses.v1')})
        if 'checkpointed' in config:
            from tools.checkpointed_sft import export
            export(output, output.parent / 'roles')
    finally:
        runtime.client.close()
    build_rl(root, config)


def canonical_rl_row(row, checkpointed=None):
    """Keep image bytes and source fields; add the exact HJL system contract."""
    from agent0_protocol.tools import get_tool_registry
    row = dict(row)
    prompt = list(row["prompt"])
    systems = [m for m in prompt if m["role"] == "system"]
    if systems and any(m["content"] != render_system_prompt() for m in systems):
        raise ProtocolError("Conflicting stored RL system prompt")
    if not systems:
        prompt.insert(0, {"role": "system", "content": render_system_prompt()})
    row["prompt"] = prompt
    row["tools"] = get_tool_registry().definitions()
    trajectory = CanonicalTrajectory(trajectory_id="rl_" + hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest(),
        tools=row["tools"], items=[{"type": "message", **m} for m in prompt], metadata=dict(row.get("extra_info") or {}))
    row["canonical_trajectory_json"] = json.dumps(trajectory.to_dict(), ensure_ascii=False)
    row["schema_version"] = "agent0.responses.v1"
    row["protocol_version"] = "agent0.responses.v1"
    row["extra_info"] = dict(row.get("extra_info") or {})
    row["extra_info"]["smoke_only"] = False
    row["reward_model"] = {"style": "rule", **row["reward_model"]}
    if checkpointed:
        from agent0_protocol.checkpointed import role_prompt
        row['prompt'][0]['content'] = role_prompt('solve')
        trajectory.items[0]['content'] = role_prompt('solve')
        trajectory.metadata['protocol_version'] = checkpointed['protocol_version']
        row['canonical_trajectory_json'] = json.dumps(trajectory.to_dict(), ensure_ascii=False)
        row['protocol_version'] = checkpointed['protocol_version']
    return row


def build_rl(root, config):
    from collections import Counter
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tools.data_builder import multisource_selection as selection
    source = Path(config["assets"]["source_root"])
    final = root / config["local_data"]["sft_output"]
    sft = read_rows(final)
    if len(sft) != 1000:
        raise ProtocolError("RL construction requires the final mixed SFT 1000 rows")
    grouped = selection._metadata(source / "data/processed/global_partition_v1/assigned_rl.jsonl")
    for name in selection.EVAL_TRAIN_PATHS:
        grouped[name] = selection._eval_metadata(source, name)
    downloaded = selection._downloaded_train_counts(source)
    downloaded.update({name: len(grouped[name]) for name in selection.EVAL_TRAIN_PATHS})
    quotas = selection.proportional_quotas(downloaded, 200)
    pool_quotas = selection.proportional_quotas(downloaded, 1000)
    pool_quotas = {s: max(pool_quotas[s], 2 * quotas[s]) for s in selection.DATASETS}
    metadata = selection._select_metadata(grouped, pool_quotas, 20260922)
    for name in selection.EVAL_TRAIN_PATHS:
        metadata[name] = grouped[name]
    loaders = {"chartqa": selection._load_chartqa, "arxivqa": selection._load_arxivqa,
               "thinklite": selection._load_thinklite,
               **{s: selection._load_eval_training for s in selection.EVAL_TRAIN_PATHS}}
    pool = [row for s in selection.DATASETS for row in loaders[s](source, metadata[s], pool_quotas[s], 20260922)]
    validation = pq.read_table(config["assets"]["validation"]).to_pylist()
    excluded = {selection._question_key(row["trajectory"]["metadata"]["question"]) for row in sft}
    excluded.update(selection._question_key(str(m["content"])) for row in validation
                    for m in row["prompt"] if m["role"] == "user")
    seen, eligible, kinds = set(), [], Counter()
    for row in pool:
        question = str(row["prompt"][0]["content"])
        key = selection._question_key(question)
        kind = selection._answer_kind(row)
        kinds[kind] += 1
        if kind == "open_ended" or key in excluded or key in seen:
            continue
        seen.add(key)
        row["extra_info"]["answer_kind"] = kind
        eligible.append(row)
    formal = selection._select_rows(eligible, quotas, seed=20260922, namespace="formal")
    warmup = selection._select_rows(eligible, quotas, seed=20260922, namespace="warmup",
                                    excluded={selection._rl_group_key(r) for r in formal})
    outputs = [(config["local_data"]["rl_formal"], formal),
               (config["local_data"]["rl_warmup"], warmup),
               (config["local_data"]["validation"], validation)]
    for name, rows in outputs:
        path = root / name
        if path.exists():
            raise ProtocolError("Refusing to replace an existing RL artifact: " + str(path))
    for name, rows in outputs:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([canonical_rl_row(row, config.get('checkpointed')) for row in rows]), path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
    inventory_root = (root / config['local_data']['rl_formal']).parent if 'checkpointed' in config else root / 'data/rl'
    _write_state(inventory_root / "local_manifest.json", {"quotas": quotas, "downloaded_counts": downloaded,
        "formal_warmup_disjoint": True, "candidate_answer_kinds": dict(kinds), "seed": 20260922,
        "future_eval_excluded_sources": list(selection.EVAL_TRAIN_PATHS),
        "authorized_repurposed_eval_sources": list(selection.EVAL_TRAIN_PATHS),
        "validation_source": config["assets"]["validation"], "sft_source": str(final)})
    (inventory_root / "README.md").write_text("Local canonical warm-up/formal: 200 rows each. Validation is a canonical copy of the read-only source. See local_manifest.json for split provenance and benchmark exclusions.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "reset", "sft", "rl"))
    parser.add_argument('--profile', default='local_4090')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = load_config(root, args.profile)
    if 'checkpointed' in config:
        from agent0_protocol.checkpointed import validate_config
        validate_config(config['checkpointed'])
        settings = dict(config['checkpointed'])
        settings['teacher_processor_model'] = config['assets']['teacher_model']
        settings['teacher_model_tokens'] = config['sft_data_generation']['local_teacher']['serve']['max_model_len']
        os.environ['AGENT0_CHECKPOINTED_SETTINGS'] = json.dumps(settings)
    if args.action == 'sft':
        # Validate incompatible state before starting any owned GPU service.
        from tools.training.teacher_session import teacher_session
        from agent0_protocol.responses_runtime import ResponsesConfig, ResponsesRuntime
        from tools.data_builder.sft_stream import fingerprint
        local = root / config['local_data']['sft_root']
        runtime = ResponsesRuntime(ResponsesConfig.from_env(), probe_on_init=False)
        try:
            for source, quota in QUOTAS.items():
                state = local / f'{source}_generated.jsonl.state.json'
                if state.exists():
                    tasks = read_rows(local / f'{source}_candidates.jsonl')
                    expected = hashlib.sha256((fingerprint(tasks, runtime, config['sft_data_generation']['min_steps'], True) + str(quota)).encode()).hexdigest()
                    if json.loads(state.read_text())['fingerprint'] != expected:
                        raise ProtocolError('SFT fingerprint mismatch; explicitly reset only after stopping the builder')
        finally:
            runtime.client.close()
        preflight(root, config)
        with teacher_session(root, config):
            result = build_sft(root, config)
    else:
        result = {"prepare": prepare, "preflight": preflight, "reset": reset,
                  "rl": build_rl}[args.action](root, config)
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
