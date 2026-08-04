"""Adapter from the retained CUDA validator to the released software model.

The historical validator passed simple case objects with ``a``, ``b``, ``c``,
and optional scale fields.  The public model deliberately accepts that same
duck-typed interface, so no research model implementation is duplicated here.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tcgen05_model.model import (  # noqa: E402
    Tcgen05BlockScaledMmaModel,
    Tcgen05BlockScaledMxf4E2M1MmaModel,
    Tcgen05BlockScaledMxf4Nvfp4E2M1Scale2XMmaModel,
    Tcgen05BlockScaledMxf4Nvfp4E2M1Scale4XMmaModel,
    Tcgen05BlockScaledMxf8f6f4E2M1MmaModel,
    Tcgen05BlockScaledMxf8f6f4E2M3MmaModel,
    Tcgen05BlockScaledMxf8f6f4E3M2MmaModel,
    Tcgen05BlockScaledMxf8f6f4E4M3MmaModel,
    Tcgen05BlockScaledMxf8f6f4E5M2MmaModel,
    Tcgen05I8S32MmaModel,
    Tcgen05MixedRawWindowMmaModel,
    Tcgen05RawWindowBf16MmaModel,
    Tcgen05RawWindowF16MmaModel,
    Tcgen05RawWindowFp4E2M1MmaModel,
    Tcgen05RawWindowFp6E2M3MmaModel,
    Tcgen05RawWindowFp6E3M2MmaModel,
    Tcgen05RawWindowFp8E4M3MmaModel,
    Tcgen05RawWindowFp8E5M2MmaModel,
    Tcgen05RawWindowNvfp4MmaModel,
    Tcgen05RawWindowTf32MmaModel,
    _tcgen05_decoder_for_format,
)


class _ExploratoryModelRemoved:
    """Fail clearly if an old exploratory CLI mode is invoked by hand."""

    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "exploratory model variants are not part of the release; "
            "use run_full_validation.py or a raw-window release model"
        )


# The retained command dispatcher historically imported these names for probe
# modes that the release driver never invokes.  Stubs keep argument parsing
# compatible without shipping the discarded research implementations.
FixedAlignTf32MmaModel = _ExploratoryModelRemoved
SimpleTf32MmaModel = _ExploratoryModelRemoved
Tcgen05ExactTf32MmaModel = _ExploratoryModelRemoved
Tcgen05ProductSumThenCModel = _ExploratoryModelRemoved
