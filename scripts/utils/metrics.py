import gc
import torch
import numpy as np
def get_tensors_in_memory():
    tensor_count = 0
    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                total_size += obj.size()
                tensor_count += 1
        except:
            pass
    return tensor_count, total_size


def softmax(t):
    return t.exp() / t.exp().sum(-1).unsqueeze(-1)


def roc_auc_compute_fn(y_preds, y_targets):
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")

    y_true = y_targets#.detach().numpy()
    nunique = len(np.unique(y_true)) 
    y_pred = np.nan_to_num(y_preds)

    if nunique > 2:
        from sklearn.metrics import accuracy_score
        return accuracy_score(y_true, y_pred)
    else:
        return roc_auc_score(y_true, y_pred)

