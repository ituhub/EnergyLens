#!/usr/bin/env python3
"""
EnergyLens Deployment Fix — patches ml/training.py for:
  1. EnhancedInformer state_dict key remapping (short→long)
  2. XGBoost pickle class alias (XGBoostModel→XGBoostTimeSeriesModel)

Run from repo root:  python fix_deployment.py
Creates backups:     ml/training.py.bak2
"""

import re
from pathlib import Path

def patch_training_py():
    path = Path("ml/training.py")
    source = path.read_text()

    # ── Backup ──
    backup = Path("ml/training.py.bak2")
    backup.write_text(source)
    print(f"✓ Backup saved to {backup}")

    # ═══════════════════════════════════════════
    # FIX 2: XGBoost class alias in _KaggleUnpickler
    # ═══════════════════════════════════════════
    # Current code:
    #     class _KaggleUnpickler(pickle.Unpickler):
    #         def find_class(self, module, name):
    #             if module == "__main__" or module == "builtins":
    #                 from . import models as ml_models
    #                 cls = getattr(ml_models, name, None)
    #                 if cls is not None:
    #                     return cls
    #             return super().find_class(module, name)
    #
    # Need to add: alias XGBoostModel → XGBoostTimeSeriesModel

    old_find_class = '''    class _KaggleUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "__main__" or module == "builtins":
                from . import models as ml_models
                cls = getattr(ml_models, name, None)
                if cls is not None:
                    return cls
            return super().find_class(module, name)'''

    new_find_class = '''    # Class aliases: notebook class names → production class names
    _CLASS_ALIASES = {
        "XGBoostModel": "XGBoostTimeSeriesModel",
    }

    class _KaggleUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            name = _CLASS_ALIASES.get(name, name)
            if module == "__main__" or module == "builtins":
                from . import models as ml_models
                cls = getattr(ml_models, name, None)
                if cls is not None:
                    return cls
            return super().find_class(module, name)'''

    if old_find_class in source:
        source = source.replace(old_find_class, new_find_class)
        print("✓ FIX 2: Added XGBoostModel → XGBoostTimeSeriesModel alias")
    else:
        print("⚠ FIX 2: Could not find exact _KaggleUnpickler block — check manually")

    # ═══════════════════════════════════════════
    # FIX 1: Informer state_dict key remapping
    # ═══════════════════════════════════════════
    # Current code:
    #     model = nn_constructors[name]()
    #     state = torch.load(pt_file, map_location="cpu", weights_only=True)
    #     model.load_state_dict(state)
    #
    # Need: remap short keys to long before load_state_dict

    old_load = '''            if name in nn_constructors and pt_file.exists():
                model = nn_constructors[name]()
                state = torch.load(pt_file, map_location="cpu", weights_only=True)
                model.load_state_dict(state)'''

    new_load = '''            if name in nn_constructors and pt_file.exists():
                model = nn_constructors[name]()
                state = torch.load(pt_file, map_location="cpu", weights_only=True)
                # Remap short checkpoint keys (Kaggle notebook) to production names
                _KEY_REMAP = {
                    "proj": "input_projection",
                    "enc": "encoder",
                    "out": "output_projection",
                    "norm": "input_norm",
                    "pos": "pos_encoding",
                }
                if any(k.split(".")[0] in _KEY_REMAP for k in state):
                    state = {
                        ".".join([_KEY_REMAP.get(parts[0], parts[0])] + parts[1:]): v
                        for k, v in state.items()
                        for parts in [k.split(".")]
                    }
                    logger.info(f"Remapped {name} state_dict keys (notebook → production)")
                model.load_state_dict(state)'''

    if old_load in source:
        source = source.replace(old_load, new_load)
        print("✓ FIX 1: Added Informer state_dict key remapping")
    else:
        print("⚠ FIX 1: Could not find exact load block — check manually")

    # ── Write ──
    path.write_text(source)
    print(f"\n✓ Wrote patched {path}")


def verify_patches():
    """Quick verify the patches landed."""
    source = Path("ml/training.py").read_text()

    checks = [
        ("XGBoostModel", "XGBoostModel alias"),
        ("_KEY_REMAP", "Informer key remap"),
        ("_CLASS_ALIASES", "Class aliases dict"),
    ]

    print("\nVerification:")
    for marker, label in checks:
        if marker in source:
            print(f"  ✓ {label} — present")
        else:
            print(f"  ✗ {label} — MISSING")


if __name__ == "__main__":
    patch_training_py()
    verify_patches()

    print("\n" + "═" * 50)
    print("Next steps:")
    print("  1. Test locally:  python -c \"from ml.training import load_models; print(load_models('models', 'DK1'))\"")
    print("  2. Deploy:        gcloud run deploy energylens --source . --region europe-north1")
    print("═" * 50)
