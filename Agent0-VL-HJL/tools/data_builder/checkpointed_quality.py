"""New-protocol teacher construction and replay auditing."""
import copy
import json
from pathlib import Path
import uuid

from agent0_protocol.checkpointed import PROTOCOL, MODES, role_prompt
from agent0_protocol.checkpointed_runtime import CheckpointedResponsesRunner
from agent0_protocol.schema import CanonicalTrajectory, ProtocolError
from tools.canonical_multimodal import render_with_images


def construct(runtime, initial, task, metadata):
    from transformers import AutoProcessor
    from tools.data_builder.sft_quality import judge_answer
    settings = copy.deepcopy(runtime.config.checkpointed_settings)
    model = settings.pop('teacher_processor_model')
    settings['limits']['max_model_tokens'] = settings.pop('teacher_model_tokens')
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)

    def count(trajectory):
        text, images = render_with_images(trajectory, processor.tokenizer)
        return len(processor(text=[text], images=images or None,
                             return_tensors='pt')['input_ids'][0])

    def score(result):
        trajectory = CanonicalTrajectory.from_dict(result.trajectory)
        try:
            judge_answer(runtime, trajectory, task)
            result.metrics['answer_judge_called'] = bool(trajectory.metadata.get('answer_judge_called'))
            result.trajectory = trajectory.to_dict()
            return 1
        except ProtocolError as exc:
            if str(exc) == 'answer_mismatch':
                return 0
            return None

    project = Path(__file__).resolve().parents[2]
    run_id = uuid.uuid4().hex
    runner = CheckpointedResponsesRunner(runtime, settings,
        project / settings['output_root'] / 'teacher' / run_id, count)
    episode = runner.run_episode(metadata['sample_hash'], initial, score, teacher=True)
    flow = episode.to_dict()
    flow['source'] = metadata['source']
    flow['candidate_index'] = metadata['candidate_index']
    audit = project / settings['audit_root'] / metadata['source'] / (run_id + '.json')
    audit.parent.mkdir(parents=True, exist_ok=True)
    temporary = audit.with_suffix('.tmp')
    temporary.write_text(json.dumps(flow, ensure_ascii=False))
    temporary.replace(audit)
    if not episode.accepted or episode.final['correct'] != 1:
        raise ProtocolError(episode.failure_reason or 'final_answer_not_correct')
    trajectory = CanonicalTrajectory.from_dict(episode.final['trajectory'])
    trajectory.metadata.update(metadata, protocol_version=PROTOCOL, checkpointed_flow=flow,
                               audit_path=str(audit))
    audit_flow(flow)
    return trajectory


def audit_flow(flow):
    if flow['protocol_version'] != PROTOCOL:
        raise ProtocolError('incompatible_checkpointed_flow')
    sessions = {}
    for session in flow['sessions']:
        spec = session['spec']
        if spec['mode'] not in MODES or spec['session_id'] in sessions:
            raise ProtocolError('invalid_or_duplicate_session')
        sessions[spec['session_id']] = session
        trajectory = CanonicalTrajectory.from_dict(session['trajectory'])
        if trajectory.items[:len(spec['initial_items'])] != spec['initial_items']:
            raise ProtocolError('session_input_changed')
        if trajectory.items[0]['content'] != role_prompt(spec['mode'],
                checkpointed=spec['mode'] != 'repair' or spec.get('checkpoint') is not None):
            raise ProtocolError('role_prompt_mismatch')
        trajectory.validate(complete=not bool(session['failure_kind']))
    if flow['accepted']:
        final_review = next((s for s in reversed(flow['sessions'])
            if s['spec']['mode'] == 'verify' and s['spec']['branch'] == 'main'), None)
        if final_review is None or (final_review.get('verification') or {}).get('verdict') != 'accept':
            raise ProtocolError('missing_final_accept')
    for checkpoint in flow['checkpoints']:
        review = sessions[checkpoint['parent_id']]
        if review['spec']['mode'] != 'verify':
            raise ProtocolError('checkpoint_parent_is_not_verifier')
        if set(checkpoint['evidence']) != set(checkpoint['evidence_payloads']):
            raise ProtocolError('unresolved_checkpoint_evidence')
    return True
