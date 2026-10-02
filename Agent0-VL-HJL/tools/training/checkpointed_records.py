"""Transport independent role sessions through VERL without mixing loss streams."""
import base64
import copy
import io
import json
from pathlib import Path
import re
from types import SimpleNamespace
import uuid

import torch

from agent0_protocol.checkpointed import Episode, SessionResult, SessionSpec, PROTOCOL, ADAPTER_LAYOUT, digest, text_of
from agent0_protocol.schema import CanonicalTrajectory, ProtocolError
from tools.canonical_multimodal import item_images, load_image
from tools.data_builder.sft_quality import StrictAnswerJudge
from tools.training.canonical_rollout import object_array
from tools.training.checkpointed_rollout import run_episodes
from tools.training.checkpointed_batch import pack_role
from verl import DataProto


def problem_inputs(trajectory, images):
    items = copy.deepcopy(trajectory.items)
    if len(items) != 2 or items[0].get('role') != 'system' or items[1].get('role') != 'user':
        raise ProtocolError('checkpointed_problem_must_be_original_question')
    if any(item_images(item) for item in items):
        return items
    text = items[1]['content']
    if not isinstance(text, str) or text.count('<image>') != len(images):
        raise ProtocolError('original_image_marker_mismatch')
    content = []
    chunks = text.split('<image>')
    for index, chunk in enumerate(chunks):
        if chunk:
            content.append({'type': 'input_text', 'text': chunk})
        if index < len(images):
            buffer = io.BytesIO()
            load_image(images[index]).save(buffer, format='PNG')
            content.append({'type': 'input_image', 'image_url':
                'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()})
    items[1]['content'] = content
    return items


def outcome_scorer(reference, question):
    if reference is None or not str(reference).strip():
        raise ProtocolError('missing_reference')

    def score(result):
        from verl.prompts.agent0_templates import assistant_text
        text = assistant_text(result.trajectory['items']) or ''
        markers = list(re.finditer(r'FINAL_ANSWER:\s*([^\n]+)|\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', text))
        if not markers:
            return 0
        answer = next(value.strip() for value in markers[-1].groups() if value is not None)
        return int(StrictAnswerJudge.is_equivalent(answer, reference, question=question))
    return score


def generate_records(rollout, prompts, settings, adapters, group, rank):
    identity = [uuid.uuid4().hex if rank == 0 else None]
    torch.distributed.broadcast_object_list(identity,
        src=torch.distributed.get_global_rank(group, 0), group=group)
    root = Path(settings['output_root']) / 'rl_tool_images' / identity[0]
    episodes, owners = [], []
    count = len(prompts)
    for index in range(count):
        trajectory = CanonicalTrajectory.from_dict(json.loads(prompts.non_tensor_batch['canonical_trajectory_json'][index]))
        if trajectory.metadata.get('protocol_version') != PROTOCOL:
            raise ProtocolError('old_rl_data_cannot_enter_checkpointed_rollout')
        images = prompts.non_tensor_batch.get('multi_modal_data', [{}] * count)[index].get('image', [])
        problem = problem_inputs(trajectory, images)
        reward = prompts.non_tensor_batch['reward_model'][index]
        scorer = outcome_scorer(reward.get('ground_truth'), text_of(problem[1]))
        # Identity is based on stored problem, never on its reference answer.
        problem_id = digest(problem)
        for sample in range(settings['sampling']['solve_n']):
            episodes.append(Episode(problem_id + ':rollout:' + str(sample),
                problem, trajectory.tools, settings, scorer))
            owners.append(index)
    states = run_episodes(episodes, rollout, settings, adapters, root, group, rank)
    records = [[] for _ in range(count)]
    for episode, index in zip(episodes, owners):
        flow = episode.to_dict()
        pixels = {session.spec.session_id: states[session.spec.session_id].pixel_inputs
                  for session in episode.sessions}
        records[index].append({'flow': flow, 'pixels': pixels})
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        (root / 'README.md').write_text('Checkpointed RL rollout\n\nProtocol: ' + PROTOCOL +
            '\nStatus: completed generation; role rewards and image evidence retained.\n' +
            'Adapter versions: ' + str(adapters.versions) + '\n')
        with (root / 'audit.jsonl').open('w') as stream:
            for episode in episodes:
                stream.write(json.dumps(episode.to_dict(), ensure_ascii=False) + '\n')
    return DataProto(batch=prompts.batch,
        non_tensor_batch={'checkpointed_records': object_array(records)},
        meta_info={'adapter_versions': dict(adapters.versions), 'adapter_layout_version': ADAPTER_LAYOUT})


def pack_records(records, settings, processor, pad_token_id, world_size):
    sessions, states, flows = [], {}, []
    for problem_records in records:
        for record in problem_records:
            flows.append(record['flow'])
            for value in record['flow']['sessions']:
                value = copy.deepcopy(value)
                value['spec'] = SessionSpec(**value['spec'])
                session = SessionResult(**value)
                sessions.append(session)
                states[session.spec.session_id] = SimpleNamespace(
                    pixel_inputs=record['pixels'][session.spec.session_id])
    return {mode: pack_role(sessions, states, mode, settings, processor,
                           pad_token_id, world_size) for mode in ('solve', 'repair', 'verify')}, flows
