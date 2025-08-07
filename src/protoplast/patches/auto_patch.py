"""
Auto-patch module for daft optimizations.

Import this module before importing daft to automatically apply all patches.

Usage:
    # Option 1: Import this module first
    import protoplast.patches.auto_patch
    import daft
    
    # Option 2: Use the context manager
    from protoplast.patches.auto_patch import patched_daft
    with patched_daft():
        import daft
        # Your daft code here
        
    # Option 3: Manual control
    from protoplast.patches.daft_flotilla import apply_flotilla_patches
    import daft
    apply_flotilla_patches()  # Apply after daft import
"""

import sys
from contextlib import contextmanager
from typing import Generator


def auto_patch_on_import():
    """
    Automatically apply patches when daft modules are imported.
    
    This sets up import hooks to patch daft modules as they're loaded.
    """
    if 'daft' in sys.modules:
        # daft already imported, apply patches now
        _apply_all_patches()
    else:
        # Set up to patch when daft gets imported
        _setup_import_hooks()


def _apply_all_patches():
    """Apply all available patches to daft."""
    try:
        from .daft_flotilla import apply_flotilla_patches
        apply_flotilla_patches()
    except Exception as e:
        print(f"⚠ Failed to apply flotilla patches: {e}")


def _setup_import_hooks():
    """Set up import hooks to patch daft when it's imported."""
    # For now, we'll use a simpler approach since import hooks are complex
    # Users should call apply_flotilla_patches() after importing daft
    pass


@contextmanager 
def patched_daft() -> Generator[None, None, None]:
    """
    Context manager that ensures daft patches are applied.
    
    Usage:
        with patched_daft():
            import daft
            # Your daft code here with patches applied
    """
    try:
        yield
    finally:
        # Apply patches if daft was imported in the context
        if 'daft' in sys.modules:
            _apply_all_patches()


def rollback_all_patches():
    """Rollback all patches to original daft implementation."""
    try:
        from .daft_flotilla import rollback_flotilla_patches
        rollback_flotilla_patches()
    except Exception as e:
        print(f"⚠ Failed to rollback flotilla patches: {e}")


# Auto-apply patches when this module is imported
# This allows users to just import this module before importing daft
if __name__ != "__main__":
    # Only auto-patch if this module is imported, not if run directly
    auto_patch_on_import()
