"""Fast, method-faithful SFT data construction for Agent0-VL.

The builder deliberately lives outside the released runtime.  It only adds
files and reads the runtime's solver prompt and sandbox implementation.
"""

from .sources import SOURCE_STAGES, iter_source_samples, load_source

__all__ = ["SOURCE_STAGES", "iter_source_samples", "load_source"]
