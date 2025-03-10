import torch
from functools import wraps
from contextlib import contextmanager

# Create a version-compatible autocast function
@contextmanager
def compatible_autocast():
    """
    A compatibility wrapper for torch.cuda.amp.autocast that works with both
    older PyTorch versions (pre-1.10) and newer versions.
    """
    if hasattr(torch.cuda.amp, 'autocast'):
        try:
            # Try modern syntax with device_type (PyTorch 1.10+)
            with torch.cuda.amp.autocast(device_type='cuda'):
                yield
        except TypeError:
            # Fallback to older syntax without device_type
            with torch.cuda.amp.autocast():
                yield
    else:
        # Really old versions or CPU-only PyTorch
        yield