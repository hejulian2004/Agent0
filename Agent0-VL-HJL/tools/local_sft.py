"""Validate canonical SFT without dropping rows and prepare Swift containers."""
import argparse
import hashlib
import json
from pathlib import Path

from agent0_protocol.schema import CanonicalTrajectory, ProtocolError
from agent0_protocol.local_prompts import render_system_prompt
from agent0_protocol.tools import get_tool_registry
from tools.canonical_multimodal import render_with_images, assistant_labels
from tools.data_builder.sft_quality import verify_sft_semantics
from tools.data_builder.sft_stream import _write_state


def prepare(data, model, output, max_length=30720, expected_rows=1000):
    from transformers import AutoProcessor
    rows = [json.loads(line) for line in Path(data).read_text().splitlines() if line.strip()]
    manifest = json.loads(Path(data).with_suffix(".manifest.json").read_text())
    checkpointed_mode = manifest.get('protocol') == 'agent0.checkpointed.v1' and manifest.get('mode')
    if checkpointed_mode:
        expected_rows = manifest['rows']
    if len(rows) != expected_rows or not rows:
        raise ProtocolError(f"Expected {expected_rows} rows; found {len(rows)}")
    manifest = json.loads(Path(data).with_suffix(".manifest.json").read_text())
    if manifest["sha256"] != hashlib.sha256(Path(data).read_bytes()).hexdigest():
        raise ProtocolError("Final SFT manifest hash mismatch")
    proofs = set(manifest.get("answer_judge_accepted_hashes", []))
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    converted, maximum = [], 0
    for row in rows:
        trajectory = CanonicalTrajectory.from_dict(row["trajectory"])
        if checkpointed_mode:
            from agent0_protocol.checkpointed import role_prompt
            trajectory.validate()
            if trajectory.metadata.get('mode') != checkpointed_mode or trajectory.items[0]['content'] != role_prompt(
                    checkpointed_mode, checkpointed=trajectory.metadata.get('repair_strategy', 'suffix') != 'full'):
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
        if len(ids) > max_length:
            raise ProtocolError(f"Overlength row {trajectory.trajectory_id}: {len(ids)} > {max_length}")
        if not any(x != -100 for x in assistant_labels(ids, processor.tokenizer)):
            raise ProtocolError("No supervised assistant tokens")
        converted.append({"messages": [{"role": "user", "content": json.dumps(trajectory.to_dict(), ensure_ascii=False)}]})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in converted))
    temporary.replace(output)
    report = {"rows": len(rows), "maximum_tokens": maximum, "limit": max_length,
              "data_sha256": manifest["sha256"], "model": str(model),
              "system_sha256": hashlib.sha256(render_system_prompt().encode()).hexdigest()}
    _write_state(output.with_suffix(".validation.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=30720)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.model, args.output, args.max_length), indent=2))


if __name__ == "__main__":
    main()
