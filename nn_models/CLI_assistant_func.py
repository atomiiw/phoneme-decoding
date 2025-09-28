import os

def folds_exist(path, n_folds):
    for i in range(n_folds):
        f = os.path.join(path, f"fold_data/fold_{i}.h5")
        if not os.path.exists(f):
            print("❌ Missing:", f)
            return False
    return True