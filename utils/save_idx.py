import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pickle
import os


def generate_indices_three_way(total_samples: int,
                               per_sample: int,
                               first_n: int,
                               train_k: int,
                               seed: int = 42):
    """Generate three-way indices: train / val / test.

    Convention (consistent with current script logic):
    - Each sample has per_sample entries (local index 0..per_sample-1).
    - The first first_n entries participate in the train+val split, from which train_k are randomly selected for train, and the rest for val.
    - The last per_sample-first_n entries are fixed as test.
    Returns (train_idx, val_idx, test_idx), all of which are global flattened indices.
    """
    rng = np.random.RandomState(seed)
    train_idx = []
    val_idx = []
    test_idx = []

    for s in range(total_samples):
        base = s * per_sample
        local_candidates = list(range(base, base + first_n))
        chosen = rng.choice(local_candidates, size=train_k, replace=False).tolist()
        train_idx.extend(chosen)
        val_local = [i for i in local_candidates if i not in chosen]
        val_idx.extend(val_local)
        test_idx.extend(list(range(base + first_n, base + per_sample)))

    train_idx.sort()
    val_idx.sort()
    test_idx.sort()
    return train_idx, val_idx, test_idx


def generate_indices_two_way(total_samples: int,
                             per_sample: int,
                             first_n: int,
                             train_k: int,
                             seed: int = 42):
    """Generate two-way indices: train / test.

    Convention similar to three-way:
    - Each sample has per_sample entries (local index 0..per_sample-1).
    - The first first_n entries participate in the train+test split, from which train_k are randomly selected for train.
    - The remaining (from the first first_n not chosen) and the last per_sample-first_n entries are all used as test.
    Returns (train_idx, test_idx), all of which are global flattened indices.
    """
    rng = np.random.RandomState(seed)
    train_idx = []
    test_idx = []

    for s in range(total_samples):
        base = s * per_sample
        local_candidates = list(range(base, base + first_n))
        chosen = rng.choice(local_candidates, size=train_k, replace=False).tolist()
        train_idx.extend(chosen)
        # Remaining candidates + tail unified as test
        remain_local = [i for i in local_candidates if i not in chosen]
        test_idx.extend(remain_local)
        test_idx.extend(list(range(base + first_n, base + per_sample)))

    train_idx.sort()
    test_idx.sort()
    return train_idx, test_idx


def save_indices_three_way(out_path: str,
                           total_samples: int,
                           per_sample: int,
                           first_n: int,
                           train_k: int,
                           seed: int = 42):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    train_idx, val_idx, test_idx = generate_indices_three_way(
        total_samples=total_samples,
        per_sample=per_sample,
        first_n=first_n,
        train_k=train_k,
        seed=seed,
    )
    with open(out_path, 'wb') as f:
        pickle.dump({'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}, f)
    print(f'Wrote indices to {out_path}')
    print('counts:', 'train=', len(train_idx), 'val=', len(val_idx), 'test=', len(test_idx))


def save_indices_two_way(out_path: str,
                         total_samples: int,
                         per_sample: int,
                         first_n: int,
                         train_k: int,
                         seed: int = 42):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    train_idx, test_idx = generate_indices_two_way(
        total_samples=total_samples,
        per_sample=per_sample,
        first_n=first_n,
        train_k=train_k,
        seed=seed,
    )
    with open(out_path, 'wb') as f:
        pickle.dump({'train_idx': train_idx, 'test_idx': test_idx}, f)
    print(f'Wrote indices to {out_path}')
    print('counts:', 'train=', len(train_idx), 'test=', len(test_idx))


def generate_indices_tail_test_with_val(total_samples: int,
                                        per_sample: int,
                                        num_test_samples: int = 10,
                                        tail_steps: int = 20,
                                        val_ratio: float = 0.1,
                                        seed: int = 42):
    """Randomly select num_test_samples samples, using their last tail_steps time steps as test;

    For each sample, take val_ratio from its non-test indices as val, and the rest as train.

    Returns (train_idx, val_idx, test_idx), all of which are global flattened indices.
    """
    rng = np.random.RandomState(seed)

    chosen_samples = rng.choice(np.arange(total_samples, dtype=int),
                                size=num_test_samples,
                                replace=False)

    test_idx = []
    for s in chosen_samples:
        base = int(s) * per_sample
        start = base + (per_sample - tail_steps)
        end = base + per_sample
        test_idx.extend(range(start, end))

    test_idx = sorted(test_idx)

    # For each sample, take val_ratio from its non-test indices as val
    val_idx = []
    train_idx = []

    for s in range(total_samples):
        base = s * per_sample
        if s in chosen_samples:
            # test sample: indices available from 0 to per_sample - tail_steps - 1
            available = list(range(base, base + per_sample - tail_steps))
        else:
            # non-test sample: all indices available
            available = list(range(base, base + per_sample))

        # Take val_ratio from available indices as val
        val_size = int(len(available) * val_ratio)
        val_chosen = rng.choice(available, size=val_size, replace=False).tolist()
        val_idx.extend(val_chosen)
        train_local = [i for i in available if i not in val_chosen]
        train_idx.extend(train_local)

    val_idx.sort()
    train_idx.sort()

    return train_idx, val_idx, test_idx


def save_indices_tail_test_with_val(out_path: str,
                                    total_samples: int,
                                    per_sample: int,
                                    num_test_samples: int = 10,
                                    tail_steps: int = 20,
                                    val_ratio: float = 0.1,
                                    seed: int = 42):
    """Wrapper for saving:

    - Randomly select num_test_samples samples, using their last tail_steps steps as test;
    - For each sample, take val_ratio from its non-test indices as val, the rest as train.
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    train_idx, val_idx, test_idx = generate_indices_tail_test_with_val(
        total_samples=total_samples,
        per_sample=per_sample,
        num_test_samples=num_test_samples,
        tail_steps=tail_steps,
        val_ratio=val_ratio,
        seed=seed,
    )
    with open(out_path, 'wb') as f:
        pickle.dump({'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx}, f)
    print(f'Wrote indices to {out_path}')
    print('counts:', 'train=', len(train_idx), 'val=', len(val_idx), 'test=', len(test_idx))


if __name__ == '__main__':
    # New scheme:
    # - Randomly select 10 samples;
    # - The last 20 time steps of these 10 samples serve as test;
    # - For each sample, 10% of its non-test indices are taken as val, the rest as train.

    total_samples = 10
    per_sample = 200      # Each sample has per_sample time steps
    num_test_samples = 10
    tail_steps = 20
    val_ratio = 0.1

    out_path = './indices/indices_tailtest_100x200_10samples_20steps_val0.1.pkl'
    save_indices_tail_test_with_val(
        out_path=out_path,
        total_samples=total_samples,
        per_sample=per_sample,
        num_test_samples=num_test_samples,
        tail_steps=tail_steps,
        val_ratio=val_ratio,
        seed=42,
    )