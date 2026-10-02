"""Swift/Megatron plugin: canonical HJL conversations and real image tensors."""
import json

from swift.template import register_template
from swift.template.templates.qwen import QwenTemplateMeta, Qwen2_5VLTemplate
from swift.template.base import MaxLengthError
from agent0_protocol.schema import CanonicalTrajectory
from tools.canonical_multimodal import render_with_images, assistant_labels


class HJLCanonicalTemplate(Qwen2_5VLTemplate):
    def _encode_truncated(self, inputs):
        # The JSON container is internal to the loader, never a model prompt.
        trajectory = CanonicalTrajectory.from_dict(json.loads(inputs.messages[0]["content"]))
        text, images = render_with_images(trajectory, self.tokenizer)
        encoded = self.processor(text=[text], images=images or None, padding=False, return_tensors="pt")
        ids = encoded.pop("input_ids")[0].tolist()
        encoded.pop("attention_mask", None)
        if self.max_length is not None and len(ids) > self.max_length:
            raise MaxLengthError(f"Canonical row has {len(ids)} tokens; limit is {self.max_length}")
        labels = assistant_labels(ids, self.tokenizer)
        boundary = trajectory.metadata.get('loss_start_item_index', 0)
        if boundary:
            prefix = CanonicalTrajectory(trajectory.trajectory_id, trajectory.tools,
                                         items=trajectory.items[:boundary])
            prefix_text, prefix_images = render_with_images(prefix, self.tokenizer)
            prefix_ids = self.processor(text=[prefix_text], images=prefix_images or None,
                padding=False, return_tensors='pt')['input_ids'][0].tolist()
            if ids[:len(prefix_ids)] != prefix_ids:
                raise ValueError('Canonical prefix token alignment failed')
            labels[:len(prefix_ids)] = [-100] * len(prefix_ids)
        if not any(label != -100 for label in labels):
            raise ValueError("Canonical row has no assistant actions")
        encoded.update(input_ids=ids, labels=labels, length=len(ids))
        return dict(encoded)


register_template(QwenTemplateMeta("hjl_canonical_vl", template_cls=HJLCanonicalTemplate,
                                   default_system=None, agent_template=None))
