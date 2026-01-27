import os
import argparse 

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

import metrics
from tasks import get_models
import utils
from pprint import pprint

from measure_alignment import prepare_features, compute_score



    
def compute_alignment_sweep(x_feat_paths, y_feat_paths, metric, topk_vec, precise=True):
    """
    Args:
        x_feat_paths: list of paths to x features
        y_feat_paths: list of paths to y features
        metric: the metric to use
        topk_vec: vector of the number of nearest neighbors to use (specific to knn metrics)
        precise: if true use exact quantiling. (helpful to set to false if running on cpu)
            this is more of a feature to speed up matmul if using float32 
            used in measure_alignment.py
    Returns:
        alignment_scores: a numpy array of shape len(x_feat_paths) x len(y_feat_paths) x len(topk_vec)
        alignment_indices: a numpy array of shape len(x_feat_paths) x len(y_feat_paths) x len(topk_vec) x 2
    """
    
    os.makedirs(args.output_dir, exist_ok=True)

    symmetric_metric = (x_feat_paths == y_feat_paths)
    if metric == "cycle_knn":
        symmetric_metric = False

    alignment_scores = np.zeros((len(x_feat_paths), len(y_feat_paths), len(topk_vec)))
    alignment_indices = np.zeros((len(x_feat_paths), len(y_feat_paths), len(topk_vec), 2))

    pbar = tqdm(total=len(y_feat_paths) * len(x_feat_paths))
    
    # sweep over topk values and compute alignment for each between all models 
    for k_idx, topk in enumerate(topk_vec):
        for i, x_fp in enumerate(x_feat_paths):
            raw_x = torch.load(x_fp, map_location="cuda:0")["feats"]
            if isinstance(raw_x, torch.Tensor):
                x_feats = prepare_features(raw_x.float(), exact=precise)
            else:
                x_feats = [prepare_features(layer.float(), exact=precise) for layer in raw_x]
            
            # x_feats = prepare_features(torch.load(x_fp, map_location="cuda:0")["feats"].float(), exact=precise)
                
            for j, y_fp in enumerate(y_feat_paths):
                if symmetric_metric:
                    if i > j:
                        pbar.update(1)
                        continue           

                raw_y = torch.load(y_fp, map_location="cuda:0")["feats"]
                if isinstance(raw_y, torch.Tensor):
                    y_feats = prepare_features(raw_y.float(), exact=precise)
                else:
                    y_feats = [prepare_features(layer.float(), exact=precise) for layer in raw_y]
                best_score, best_indices = compute_score(y_feats, x_feats, metric=metric, topk=topk)
                
                alignment_scores[i, j, k_idx] = best_score
                alignment_indices[i, j, k_idx] = best_indices
                
                if symmetric_metric:
                    alignment_scores[j, i, k_idx] = best_score
                    alignment_indices[j, i, k_idx] = best_indices[::-1]

                pbar.update(1)

                del y_feats
                torch.cuda.empty_cache()

    return alignment_scores, alignment_indices


if __name__ == "__main__":
    """
    recommended to use llm as modality_x since it will load each LLM features once
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=str, default="prh/minhuh")
    parser.add_argument("--subset",         type=str, default="wit_1024")

    parser.add_argument("--modality_x",     type=str, default="all", choices=["vision", "language", "all"])
    parser.add_argument("--prompt_x",       action="store_true")
    parser.add_argument("--pool_x",         type=str, default=None, choices=['avg', 'cls'])
    
    parser.add_argument("--modality_y",     type=str, default="all", choices=["vision", "language", "all"])
    parser.add_argument("--prompt_y",       action="store_true")
    parser.add_argument("--pool_y",         type=str, default=None, choices=['avg', 'cls'])

    parser.add_argument("--modelset",       type=str, default="val", choices=["val", "test", "mini"])
    parser.add_argument("--metric",         type=str, default="mutual_knn", choices=metrics.AlignmentMetrics.SUPPORTED_METRICS)
    parser.add_argument("--topk_len",           type=int, default=10)

    parser.add_argument("--input_dir",      type=str, default="./results/features")
    parser.add_argument("--output_dir",     type=str, default="./results/alignment")
    parser.add_argument("--precise",        action="store_true")
    parser.add_argument("--force_remake",   action="store_true")

    args = parser.parse_args()
    
    if not args.precise:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    save_path = utils.to_alignment_filename(
            args.output_dir, args.dataset, args.modelset,
            args.modality_x, args.pool_x, args.prompt_x,
            args.modality_y, args.pool_y, args.prompt_y,
            args.metric, args.topk_len
    )
    
    if os.path.exists(save_path) and not args.force_remake:
        print(f"alignment already exists at {save_path}")
        exit()
    
    llm_models, lvm_models = get_models(args.modelset, modality='all')
    models_x = llm_models if args.modality_x == "language" else lvm_models
    models_y = llm_models if args.modality_y == "language" else lvm_models
    
    models_x_paths = [utils.to_feature_filename(args.input_dir, args.dataset, args.subset, m, args.pool_x, args.prompt_x) for m in models_x]
    models_y_paths = [utils.to_feature_filename(args.input_dir, args.dataset, args.subset, m, args.pool_y, args.prompt_y) for m in models_y]
    
    for fn in models_x_paths + models_y_paths:
        assert os.path.exists(fn), fn
    
    topk_vec = torch.linspace(5, 300, steps=args.topk_len).round().long().tolist()
    print(f"dataset:\t{args.dataset}")
    print(f"metric: \t{args.metric}")
    if 'knn' in args.metric:
        print(f"topk:\t{topk_vec}")
    
    print(f"models_x_paths:")    
    pprint(models_x_paths)
    print("\nmodels_y_paths:")
    pprint(models_y_paths)
    
    print('\nmeasuring alignment')
    alignment_scores, alignment_indices = compute_alignment_sweep(models_x_paths, models_y_paths, args.metric, topk_vec, args.precise)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, {"scores": alignment_scores, "indices": alignment_indices, 'topk_vec': topk_vec})
    print(f"saved to {save_path}")
    