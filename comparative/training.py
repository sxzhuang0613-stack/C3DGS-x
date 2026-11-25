"""对比实验统一训练入口：逐病例拟合所选方法，保存模型、预测与运行记录。

用法（在 C3DGS 根目录）：
    python comparative/training.py --config comparative/config.yaml --method gsvr
    python comparative/training.py --method dyna3dgr --stage stage1   # 或 --stage full
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from case_data import load_config, iter_cases, seed_everything
from methods import load_method, method_tag

PREDICTION_FILE = 'prediction.npz'


def save_prediction(out_dir, case, prediction, cfg):
    """拆分整数层位（计分用，float32）与插层（可视化用，float16）后写入 prediction.npz。"""
    query = np.asarray(prediction['query_index'], dtype=np.float64)
    integer = np.flatnonzero(np.abs(query - np.round(query)) < 1e-6)
    if not np.array_equal(np.round(query[integer]).astype(int), np.arange(case.shape[0])):
        raise ValueError('预测层位未覆盖全部整数层')
    volume = np.asarray(prediction['volume'], dtype=np.float32)
    arrays = dict(
        query_index=query, slices=volume[integer], dense=volume.astype(np.float16),
        frame=int(prediction['frame']), train_index=case.train_index,
        test_index=case.test_index, spacing=np.asarray(case.spacing),
        interp_step_size=cfg.interp_step_size,
    )
    if 'sequence' in prediction:
        arrays['sequence'] = np.asarray(prediction['sequence'], dtype=np.float32)[:, integer]
        arrays['frames'] = np.asarray(prediction['frames'], dtype=int)
    np.savez_compressed(out_dir / (PREDICTION_FILE + '.tmp.npz'), **arrays)
    (out_dir / (PREDICTION_FILE + '.tmp.npz')).replace(out_dir / PREDICTION_FILE)


def main():
    # 读取配置与显式覆盖
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.yaml')))
    parser.add_argument('--method', required=True)
    parser.add_argument('--stage', choices=['stage1', 'full'], default=None,
                        help='仅 dyna3dgr：stage1 只拟合初始帧静态高斯；full 为完整两阶段')
    parser.add_argument('--cases', nargs='*', default=None, help='覆盖配置中的病例列表')
    parser.add_argument('--protocol', choices=['observed', 'holdout'], default=None)
    parser.add_argument('--iters', type=lambda v: v if v == 'preset' else int(v), default=None,
                        help='覆盖迭代数：正整数或 preset（各方法预设）')
    args = parser.parse_args()
    cfg = load_config(args.config)
    for key in ('cases', 'iters'):
        if getattr(args, key) is not None:
            setattr(cfg, key, getattr(args, key))
    if args.protocol is not None:
        cfg.eval_protocol = args.protocol
    method_cfg = dict(cfg.methods[args.method])
    stage = None
    if args.method == 'dyna3dgr':
        stage = args.stage or method_cfg['stage']
    elif args.stage is not None:
        raise ValueError('--stage 只适用于 dyna3dgr')

    # 初始化方法与输出目录
    torch.cuda.set_device(torch.device(cfg.device))
    method = load_method(args.method)
    tag = method_tag(args.method, stage)
    root = Path(cfg.output_root) / cfg.eval_protocol / tag

    # 逐病例拟合并保存
    for case in iter_cases(cfg):
        out_dir = root / case.name
        if cfg.skip_existing and (out_dir / PREDICTION_FILE).is_file():
            print(f'[{tag}] {case.name}: 已有预测，跳过')
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(cfg.random_seed)
        print(f'[{tag}] {case.name}: S={case.shape[0]} train={case.train_index.tolist()}')

        start = time.time()
        prediction = method.fit(case, cfg, method_cfg, out_dir, stage=stage)
        elapsed = prediction.get('elapsed_seconds', time.time() - start)

        save_prediction(out_dir, case, prediction, cfg)
        record = dict(method=args.method, stage=stage, case=case.name, elapsed_seconds=elapsed,
                      eval_protocol=cfg.eval_protocol, iters=cfg.iters,
                      iterations_run=prediction.get('iterations'),
                      slice_thickness_mm=case.thickness,
                      random_seed=cfg.random_seed, method_config=method_cfg,
                      config=vars(cfg) | {'methods': None})
        (out_dir / 'run.json').write_text(json.dumps(record, ensure_ascii=False, indent=2,
                                                     default=str), encoding='utf-8')
        print(f'[{tag}] {case.name}: 完成，用时 {elapsed:.1f}s')
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
