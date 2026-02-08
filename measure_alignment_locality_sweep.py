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



    
def compute_alignment_sweep(x_feat_paths, y_feat_paths, metric, param_vec, precise=True, layer_mode='max'):
    """
    Args:
        x_feat_paths: list of paths to x features
        y_feat_paths: list of paths to y features
        metric: the metric to use
        param_vec: vector of parameter values to sweep over (e.g., topk values or rbf_sigma values)
        precise: if true use exact quantiling. (helpful to set to false if running on cpu)
            this is more of a feature to speed up matmul if using float32 
            used in measure_alignment.py
        layer_mode: 'max' to find max alignment across all layers, 'final' to use only final layer
    Returns:
        alignment_scores: a numpy array of shape len(x_feat_paths) x len(y_feat_paths) x len(param_vec)
        alignment_indices: a numpy array of shape len(x_feat_paths) x len(y_feat_paths) x len(param_vec) x 2
    """
    
    os.makedirs(args.output_dir, exist_ok=True)

    symmetric_metric = (x_feat_paths == y_feat_paths)
    if metric == "cycle_knn":
        symmetric_metric = False

    alignment_scores = np.zeros((len(x_feat_paths), len(y_feat_paths), len(param_vec)))
    alignment_indices = np.zeros((len(x_feat_paths), len(y_feat_paths), len(param_vec), 2))

    pbar = tqdm(total=len(y_feat_paths) * len(x_feat_paths) * len(param_vec))
    
    # Get the parameter name for this metric
    sweep_config = metrics.AlignmentMetrics.SWEEP_PARAMS[metric]
    param_name = sweep_config['param']
    
    # sweep over parameter values and compute alignment for each between all models 
    for param_idx, param_value in enumerate(param_vec):
        for i, x_fp in enumerate(x_feat_paths):
            raw_x = torch.load(x_fp, map_location="cuda:0")["feats"]
            if isinstance(raw_x, torch.Tensor):
                x_feats = prepare_features(raw_x.float(), exact=precise)
            else:
                x_feats = [prepare_features(layer.float(), exact=precise) for layer in raw_x]
                
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
                
                # Build kwargs with the appropriate parameter
                kwargs = {param_name: param_value} if param_name else {}
                kwargs['layer_mode'] = layer_mode
                best_score, best_indices = compute_score(y_feats, x_feats, metric=metric, **kwargs)
                
                alignment_scores[i, j, param_idx] = best_score
                alignment_indices[i, j, param_idx] = best_indices
                
                if symmetric_metric:
                    alignment_scores[j, i, param_idx] = best_score
                    alignment_indices[j, i, param_idx] = (best_indices[1], best_indices[0])

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
    parser.add_argument("--metric",         type=str, default="mutual_knn", 
                        choices=metrics.AlignmentMetrics.SUPPORTED_METRICS)
    parser.add_argument("--sweep_len",      type=int, default=10, help="Number of steps in parameter sweep")
    parser.add_argument("--layer_mode",     type=str, default="max", choices=["max", "final"], 
                        help="'max' finds best alignment across all layers, 'final' uses only final layer")
    parser.add_argument("--run_name",      type=str, default="test", help="Subdirectory name for this run's results")

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
    
    # Get sweep configuration for the metric
    sweep_config = metrics.AlignmentMetrics.SWEEP_PARAMS[args.metric]
    param_name = sweep_config['param']
    
    # Generate parameter sweep vector based on metric
    if param_name:
        param_min = sweep_config['min']
        param_max = sweep_config['max']
        if param_name == 'topk':
            # For topk, use integer steps
            param_vec = torch.linspace(param_min, param_max, steps=args.sweep_len).round().long().tolist()
            # param_vec = np.geomspace(param_min, param_max, num=args.sweep_len).round().long().tolist()
        elif param_name == 'temperature':
            # For temperature, use linear steps
            param_vec = torch.linspace(param_min, param_max, steps=args.sweep_len).tolist()
        elif param_name == 'rbf_sigma':
            # For rbf_sigma, use geometric steps
            # param_vec = torch.linspace(param_min, param_max, steps=args.sweep_len).tolist()
            param_vec = np.geomspace(param_min, param_max, num=args.sweep_len).tolist()
        else:
            raise ValueError(f"Unknown parameter name {param_name} for metric {args.metric}")
    else:
        # No sweep for this metric, just use a single default value
        param_vec = [None]
    
    save_path = utils.to_alignment_filename(
            args.output_dir, args.dataset, args.modelset,
            args.modality_x, args.pool_x, args.prompt_x,
            args.modality_y, args.pool_y, args.prompt_y,
            args.metric, args.sweep_len, args.layer_mode, args.run_name
    )
    
    if os.path.exists(save_path) and not args.force_remake:
        print(f"alignment already exists at {save_path}")
        exit()
    
    llm_models, lvm_models = get_models(args.modelset, modality='all')
    models_x = llm_models if args.modality_x == "language" else lvm_models
    models_y = llm_models if args.modality_y == "language" else lvm_models
    
    models_x_paths = [utils.to_feature_filename(args.input_dir, args.dataset, args.subset, 
                                                m, args.pool_x, args.prompt_x) for m in models_x]
    models_y_paths = [utils.to_feature_filename(args.input_dir, args.dataset, args.subset, m, 
                                                args.pool_y, args.prompt_y) for m in models_y]
    
    for fn in models_x_paths + models_y_paths:
        assert os.path.exists(fn), fn
    
    print(f"dataset:\t{args.dataset}")
    print(f"metric: \t{args.metric}")
    print(f"layer_mode:\t{args.layer_mode}")
    if param_name:
        print(f"{param_name} sweep:\t{param_vec}")
    
    print(f"models_x_paths:")    
    pprint(models_x_paths)
    print("\nmodels_y_paths:")
    pprint(models_y_paths)
    
    print('\nmeasuring alignment')
    alignment_scores, alignment_indices = compute_alignment_sweep(models_x_paths, models_y_paths, 
                                                                  args.metric, param_vec, args.precise, 
                                                                  args.layer_mode)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, {"scores": alignment_scores, "indices": alignment_indices, 'param_vec': param_vec, 'param_name': param_name})
    print(f"saved to {save_path}")
