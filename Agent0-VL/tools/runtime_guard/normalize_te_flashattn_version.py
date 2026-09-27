"""Let Transformer Engine recognize locally built FlashAttention 2.8.3 wheels.

The CUDA 13 wheel is tagged ``2.8.3+cu130torch2.13``. TE compares that PEP 440
local version directly with its upper bound ``2.8.3``, so it incorrectly treats
the same 2.8.3 API as unsupported. Normalize only the version used for that
capability check; the installed binary and its metadata are left untouched.
"""

from importlib import metadata, util
from pathlib import Path


def main() -> None:
    raw_version = metadata.version("flash-attn")
    if "+" not in raw_version:
        print(f"[sft-pipeline] flash-attn version needs no normalization: {raw_version}")
        return

    spec = util.find_spec("transformer_engine")
    if spec is None or spec.origin is None:
        raise SystemExit("Transformer Engine is not installed; cannot apply its compatibility shim")

    backends = (
        Path(spec.origin).parent
        / "pytorch/attention/dot_product_attention/backends.py"
    )
    source = backends.read_text()
    unpatched = 'fa_utils.version = PkgVersion(get_pkg_version("flash-attn"))'
    patched = 'fa_utils.version = PkgVersion(get_pkg_version("flash-attn").split("+", 1)[0])'

    if patched in source:
        print(f"[sft-pipeline] Transformer Engine flash-attn version shim already applied ({raw_version})")
        return
    if unpatched not in source:
        raise SystemExit(
            "Transformer Engine backend source differs from the expected version check; "
            "refusing to patch it automatically"
        )

    backends.write_text(source.replace(unpatched, patched, 1))
    print(
        f"[sft-pipeline] normalized flash-attn local build suffix for TE compatibility "
        f"({raw_version} -> {raw_version.split('+', 1)[0]}); binary unchanged"
    )


if __name__ == "__main__":
    main()
