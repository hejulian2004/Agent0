"""Validate canonical SFT without dropping rows and prepare Swift containers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from agent0_protocol.schema import CanonicalTrajectory, ProtocolError
from agent0_protocol.local_prompts import render_system_prompt
from agent0_protocol.tools import get_tool_registry
from tools.canonical_multimodal import render_with_images, assistant_labels
from tools.data_builder.sft_quality import verify_sft_semantics
from tools.data_builder.sft_stream import _write_state


def prepare(data, model, output, max_length=30720, expected_rows=1000, mode_limits=None):
    from transformers import AutoProcessor
    rows = [json.loads(line) for line in Path(data).read_text().splitlines() if line.strip()]
    manifest = json.loads(Path(data).with_suffix(".manifest.json").read_text())
    from agent0_protocol.checkpointed import PROTOCOL, MODES, ADAPTER_LAYOUT, ADAPTER_FOR_MODE
    checkpointed_mode = manifest.get('protocol') == PROTOCOL and manifest.get('mode')
    shared = manifest.get('protocol') == PROTOCOL and manifest.get('adapter') == 'shared'
    if shared and (manifest.get('adapter_layout_version') != ADAPTER_LAYOUT or
                   manifest.get('mode_to_adapter') != ADAPTER_FOR_MODE or
                   manifest.get('modes') != list(MODES)):
        raise ProtocolError('incompatible_shared_sft_manifest')
    if checkpointed_mode or shared:
        expected_rows = manifest['rows']
    if len(rows) != expected_rows or not rows:
        raise ProtocolError(f"Expected {expected_rows} rows; found {len(rows)}")
    manifest = json.loads(Path(data).with_suffix(".manifest.json").read_text())
    if manifest["sha256"] != hashlib.sha256(Path(data).read_bytes()).hexdigest():
        raise ProtocolError("Final SFT manifest hash mismatch")
    proofs = set(manifest.get("answer_judge_accepted_hashes", []))
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    converted, maximum = [], 0
    mode_counts = Counter()
    for row in rows:
        trajectory = CanonicalTrajectory.from_dict(row["trajectory"])
        row_mode = trajectory.metadata.get('mode')
        mode_counts[row_mode] += 1
        if checkpointed_mode or shared:
            from agent0_protocol.checkpointed import role_prompt
            trajectory.validate()
            if row_mode not in MODES or (checkpointed_mode and row_mode != checkpointed_mode) or trajectory.items[0]['content'] != role_prompt(
                    row_mode, checkpointed=trajectory.metadata.get('repair_strategy', 'suffix') != 'full'):
                raise ProtocolError('role_prompt_or_mode_mismatch')
            if trajectory.metadata.get('loss_start_item_index') is None:
                raise ProtocolError('missing_role_loss_boundary')
        else:
            audit = verify_sft_semantics(trajectory, get_tool_registry(), proofs)
            if not audit.valid:
                raise ProtocolError("; ".join(audit.issues))
        text, images = render_with_images(trajectory, processor.tokenizer)
        ids = processor(text=[text], images=images or None, return_tensors="pt")["input_ids"][0].tolist()
        maximum = max(maximum, len(ids))
        row_limit = min(max_length, mode_limits[row_mode]) if mode_limits and (shared or checkpointed_mode) else max_length
        if len(ids) > row_limit:
            raise ProtocolError(f"Overlength row {trajectory.trajectory_id}: {len(ids)} > {row_limit}")
        labels = assistant_labels(ids, processor.tokenizer)
        if checkpointed_mode or shared:
            boundary = trajectory.metadata['loss_start_item_index']
            if type(boundary) is not int or not 0 <= boundary < len(trajectory.items):
                raise ProtocolError('invalid_role_loss_boundary')
            prefix = CanonicalTrajectory(trajectory.trajectory_id, trajectory.tools, items=trajectory.items[:boundary])
            prefix_text, prefix_images = render_with_images(prefix, processor.tokenizer)
            prefix_ids = processor(text=[prefix_text], images=prefix_images or None,
                return_tensors='pt')['input_ids'][0].tolist()
            if ids[:len(prefix_ids)] != prefix_ids:
                raise ProtocolError('role_prefix_token_alignment_failed')
            labels[:len(prefix_ids)] = [-100] * len(prefix_ids)
        if not any(x != -100 for x in labels):
            raise ProtocolError("No supervised assistant tokens")
        if checkpointed_mode or shared:
            trajectory.metadata['sft_max_tokens'] = row_limit
        converted.append({"messages": [{"role": "user", "content": json.dumps(trajectory.to_dict(), ensure_ascii=False)}]})
    if shared and {mode: mode_counts[mode] for mode in MODES} != manifest['rows_by_mode']:
        raise ProtocolError('shared_sft_mode_counts_mismatch')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted))
    temporary.replace(output)
    report = {"rows": len(rows), "maximum_tokens": maximum, "limit": max_length,
              "data_sha256": manifest["sha256"], "model": str(model),
              "system_sha256": hashlib.sha256(render_system_prompt().encode()).hexdigest()}
    if shared:
        report.update(adapter='shared', adapter_layout_version=ADAPTER_LAYOUT,
                      mode_to_adapter=ADAPTER_FOR_MODE, rows_by_mode=dict(mode_counts))
    _write_state(output.with_suffix(".validation.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=30720)
    parser.add_argument("--mode-limits", type=json.loads)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.model, args.output, args.max_length, mode_limits=args.mode_limits), indent=2))


if __name__ == "__main__":
    main()
