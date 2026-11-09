"""
test_trainer.py

Local CPU test of nnUNetTrainerEmidec / nnUNetTrainerEmidecStrongAug on the preprocessed
datasets (needs nnUNet_raw / nnUNet_preprocessed / nnUNet_results pointing at a local
nnU-Net work dir and nnUNet_n_proc_DA=0). For each arm it builds the real trainer, checks
batch composition over many batches, runs two real training steps and prints the setup.

    nnUNet_n_proc_DA=0 python experiments/nnunet/test_trainer.py
"""

import os
import shutil
import sys

import numpy as np
import torch

import nnunetv2
from nnunetv2.run.run_training import get_trainer_from_args

HERE = os.path.dirname(os.path.abspath(__file__))
dst = os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer", "nnUNetTrainerEmidec.py")
shutil.copy(os.path.join(HERE, "nnUNetTrainerEmidec.py"), dst)

ARMS = [("arm 1  real only", 501, "nnUNetTrainerEmidec", 0.0),
        ("arm 2  real + 197", 502, "nnUNetTrainerEmidec", 1 / 3),
        ("arm 3  real, strong aug", 501, "nnUNetTrainerEmidecStrongAug", 0.0),
        ("arm 4  real + 154 bright", 504, "nnUNetTrainerEmidec", 1 / 3),
        ("arm 5  real + random 154", 505, "nnUNetTrainerEmidec", 1 / 3)]
N_BATCHES = 40


def build(did, trainer, frac, seed):
    os.environ["EMIDEC_SEED"], os.environ["EMIDEC_SYN_FRACTION"] = str(seed), str(frac)
    t = get_trainer_from_args(str(did), "2d", "all", trainer_name=trainer, plans_identifier="nnUNetPlans",
                              device=torch.device("cpu"))
    t.initialize()
    return t


def main():
    ok = True
    for name, did, trainer, frac in ARMS:
        t = build(did, trainer, frac, seed=1)
        w1 = next(iter(t.network.parameters())).detach().clone()   # initial weights, before any step
        dl, _ = t.get_dataloaders()
        keys_syn = []
        for _ in range(N_BATCHES):
            b = next(dl)
            keys_syn.append(sum("__ins" in k for k in b["keys"]))
        last8 = np.mean([sum("__ins" in k for k in next(dl)["keys"][-8:]) for _ in range(200)])
        n_real = len([k for k in dl.data_loader.indices if "__ins" not in k])
        n_syn = len(dl.data_loader.indices) - n_real
        want = int(round(t.batch_size * frac))
        good = all(s == want for s in keys_syn)
        b = next(dl)
        out = t.train_step(b)
        out2 = t.train_step(next(dl))
        finite = np.isfinite(out["loss"]) and np.isfinite(out2["loss"])
        # seed check: same seed -> identical initial weights, different seed -> different
        w = lambda tr: next(iter(tr.network.parameters())).detach().clone()
        same = torch.equal(w1, w(build(did, trainer, frac, seed=1)))
        diff = not torch.equal(w1, w(build(did, trainer, frac, seed=2)))
        print(f"\n=== {name}: {trainer}, Dataset50{did % 10}")
        print(f"  cases in training set: {n_real} real + {n_syn} synthetic; batch {t.batch_size}, patch {list(t.configuration_manager.patch_size)}")
        print(f"  synthetic per batch over {N_BATCHES} batches: {sorted(set(keys_syn))} (expected {want}) -> {'ok' if good else 'WRONG'}; "
              f"epochs {t.num_epochs} x {t.num_iterations_per_epoch} iterations; mean inserts in the last 8 positions "
              f"(nnU-Net's forced-foreground slots): {last8:.2f} (random order: {want * 8 / t.batch_size:.2f} expected)")
        print(f"  batch tensor {tuple(b['data'].shape)}; train_step losses {out['loss']:.3f}, {out2['loss']:.3f} -> {'finite' if finite else 'NOT FINITE'}")
        print(f"  seed: same seed -> same init weights {same}; different seed -> different {diff}")
        outer = [type(x).__name__ for x in dl.data_loader.transforms.transforms]
        print(f"  training transform chain: {outer}")
        ok = ok and good and finite and same and diff
    # unit check of the jitter transform
    sys.path.insert(0, os.path.dirname(dst))
    from nnUNetTrainerEmidec import RandomShiftTransform
    x = torch.arange(36.).reshape(1, 6, 6); s = -torch.ones(1, 6, 6, dtype=torch.int16)
    out = RandomShiftTransform._shift(x, 2, -1, 0)
    ref = torch.zeros_like(x); ref[0, 2:, :5] = x[0, :4, 1:]
    print("\njitter unit check (dy=+2, dx=-1):", torch.equal(out, ref))
    ok = ok and torch.equal(out, ref)
    print("ALL OK" if ok else "FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
