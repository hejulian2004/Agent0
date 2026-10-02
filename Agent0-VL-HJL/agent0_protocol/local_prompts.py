"""Load the authoritative local prompt file without importing training libraries.

The prompt source remains verl/prompts/agent0_templates.py. Isolated SFT
workers only need its pure rendering functions, not VERL's torch/ray imports.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "verl/prompts/agent0_templates.py"
_spec = spec_from_file_location("_agent0_local_prompt_templates", _path)
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)

SOLVER_SYSTEM_PROMPT = _module.SOLVER_SYSTEM_PROMPT
render_system_prompt = _module.render_system_prompt
render_solver_request = _module.render_solver_request
render_verifier_request = _module.render_verifier_request
render_repair_request = _module.render_repair_request
assistant_text = _module.assistant_text
