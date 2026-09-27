import pytest

from verl.workers.actor.dp_actor import _micro_batch_loss_scale


def test_micro_batch_loss_scale_uses_samples_not_mapping_fields():
    assert _micro_batch_loss_scale(1, 4) == pytest.approx(0.25)
    assert _micro_batch_loss_scale(2, 4) == pytest.approx(0.5)


@pytest.mark.parametrize("micro_batch_size,mini_batch_size", [(0, 4), (1, 0), (5, 4)])
def test_micro_batch_loss_scale_rejects_invalid_sizes(micro_batch_size, mini_batch_size):
    with pytest.raises(ValueError):
        _micro_batch_loss_scale(micro_batch_size, mini_batch_size)
