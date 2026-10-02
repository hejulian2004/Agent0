"""Phase transition retains the Actor/optimizer and resets only data loading."""
import json
from pathlib import Path


class PhaseLoader:
    def __init__(self, warmup, formal, warmup_epochs, phase):
        self.loaders = {"warmup": warmup, "serc": formal}
        self.warmup_epochs = warmup_epochs
        self.phase_mode = phase
        self.epoch = 0
        self.current_phase = "serc" if phase == "serc" else "warmup"

    def __len__(self):
        return len(self.loaders[self.current_phase])

    def __iter__(self):
        self.current_phase = "serc" if self.phase_mode == "serc" or (self.phase_mode == "full" and self.epoch >= self.warmup_epochs) else "warmup"
        print(f"[RL phase] phase={self.current_phase} epoch={self.epoch}", flush=True)
        yield from self.loaders[self.current_phase]
        self.epoch += 1

    def state_dict(self):
        return {"epoch": self.epoch, "phase": self.current_phase,
                "loaders": {key: loader.state_dict() for key, loader in self.loaders.items()}}

    def load_state_dict(self, state):
        self.epoch, self.current_phase = state["epoch"], state["phase"]
        for key, loader in self.loaders.items():
            loader.load_state_dict(state["loaders"][key])


def install(config):
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    if getattr(RayPPOTrainer, "_local_schedule_installed", False):
        return
    original_create = RayPPOTrainer._create_dataloader
    original_save = RayPPOTrainer._save_checkpoint
    original_load = RayPPOTrainer._load_checkpoint

    def create(self):
        if not self.config.get('local_schedule'):
            return original_create(self)
        original_create(self)
        warmup = self.train_dataloader
        original_files = self.config.data.train_files
        self.config.data.train_files = self.config.local_schedule.formal_data
        original_create(self)
        formal = self.train_dataloader
        self.config.data.train_files = original_files
        self.train_dataloader = PhaseLoader(warmup, formal, self.config.local_schedule.warmup_epochs,
                                            self.config.local_schedule.phase)
        self.total_training_steps = self.config.trainer.total_training_steps
        for name in ("reward_fn", "val_reward_fn"):
            reward = getattr(self, name, None)
            if reward is not None:
                reward.local_loader = self.train_dataloader

    def save(self):
        if not self.config.get('local_schedule'):
            return original_save(self)
        original_save(self)
        path = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}" / "local_protocol.json"
        path.write_text(json.dumps({"fingerprint": self.config.local_schedule.protocol_fingerprint,
                                   "step": self.global_steps, "phase": self.train_dataloader.current_phase, "mode": self.config.local_schedule.phase}))

    def load(self):
        if not self.config.get('local_schedule'):
            return original_load(self)
        if self.config.trainer.resume_mode == "resume_path":
            path = Path(self.config.trainer.resume_from_path) / "local_protocol.json"
            if not path.is_file() or json.loads(path.read_text())["fingerprint"] != self.config.local_schedule.protocol_fingerprint:
                raise ValueError("Refusing legacy/incompatible RL resume before loading weights")
        fresh = self.train_dataloader.state_dict() if self.config.local_schedule.get('formal_from_checkpoint', False) else None
        result = original_load(self)
        if fresh is not None:
            self.train_dataloader.load_state_dict(fresh)
        return result

    RayPPOTrainer._create_dataloader = create
    RayPPOTrainer._save_checkpoint = save
    RayPPOTrainer._load_checkpoint = load
    RayPPOTrainer._local_schedule_installed = True
