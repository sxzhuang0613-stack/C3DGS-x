"""对比实验共用的配置、病例数据与几何约定。

坐标约定与 C3DGS 一致（单位 mm，病例局部栈坐标）：
面内像素中心 x=(u+.5-W/2)*sx、y=(v+.5-H/2)*sy；第 k 层位于 z=(k-(S-1)/2)*sz。
层位用连续层索引 k 表示，插层位置为非整数 k。
"""
import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

C3DGS_ROOT = Path(__file__).resolve().parents[1]
# 追加到末尾：参考仓库稍后插入首位，其同名顶层包（utils/models 等）优先于 C3DGS
if str(C3DGS_ROOT) not in sys.path:
    sys.path.append(str(C3DGS_ROOT))

from Dataloader import dataset  # noqa: E402


def load_config(path):
    """读取对比配置；拒绝重复键，相对路径以配置文件为基准。"""
    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if key in result:
                raise ValueError(f'配置键重复：{key}')
            result[key] = loader.construct_object(value_node, deep=True)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    config_path = Path(path).resolve()
    with config_path.open(encoding='utf-8') as stream:
        values = yaml.load(stream, Loader=UniqueLoader)

    resolve = lambda p: str((config_path.parent / os.path.expandvars(os.path.expanduser(p))).resolve())
    for key in ('references_root', 'output_root'):
        values[key] = resolve(values[key])
    values['dataset_paths'] = {k: resolve(v) for k, v in values['dataset_paths'].items()}

    # 整数层位必须落在插层网格上，否则观测层与插层输出无法逐层对应
    steps_per_slice = 1 / values['interp_step_size']
    if abs(steps_per_slice - round(steps_per_slice)) > 1e-6:
        raise ValueError('1/interp_step_size 必须为整数')
    if values['eval_protocol'] not in ('observed', 'holdout'):
        raise ValueError('eval_protocol 必须为 observed 或 holdout')
    if values['eval_protocol'] == 'holdout' and values['holdout_stride'] < 2:
        raise ValueError('holdout_stride 至少为 2')
    return argparse.Namespace(**values)


def seed_everything(seed):
    """每个病例开始前重置随机状态，使病例结果不依赖运行顺序。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class CaseData:
    """单个病例的观测数据与协议划分。

    sequence: [S,T,H,W] 归一化到 [0,1] 的 float32 CPU 张量（Dataloader 输出轴序）。
    train_index: 参与训练的层索引（等间隔，间隔 stride）；test_index: 计入指标的层索引。
    """
    name: str
    sequence: torch.Tensor
    spacing: tuple          # (sx, sy, sz)，mm
    thickness: float        # 层面轮廓 FWHM，mm
    reference_phase: int
    time_indices: list      # ED/ES 对应帧
    train_index: np.ndarray
    test_index: np.ndarray
    stride: int

    @property
    def shape(self):
        """返回 (S, H, W)。"""
        return self.sequence.shape[0], self.sequence.shape[2], self.sequence.shape[3]

    def frame(self, t):
        """第 t 帧全部层 [S,H,W]。"""
        return self.sequence[:, t]

    def z_mm(self, k):
        """连续层索引 k → 层法向坐标 z（mm）。"""
        return (np.asarray(k, dtype=np.float64) - (self.shape[0] - 1) / 2) * self.spacing[2]


def query_index(case, step):
    """输出层位：覆盖首末层的等间隔网格 [Q]，整数层位均在其中。"""
    count = int(round((case.shape[0] - 1) / step)) + 1
    return np.arange(count) * step


def split_slices(num_slices, protocol, stride):
    """按测试协议划分训练层与计分层。

    holdout 只对夹在两个训练层之间的留出层计分，避免把外推误当作层间重建。
    """
    all_index = np.arange(num_slices)
    if protocol == 'observed':
        return all_index, all_index, 1
    train = all_index[all_index % stride == 0]
    test = np.setdiff1d(all_index[all_index < train[-1]], train)
    return train, test, stride


def iter_cases(cfg):
    """按配置逐个产出 CaseData；集合与 evaluate.py 相同（仅含有效标签病例）。"""
    name = {'mm': 'mnms', 'mms': 'mnms', 'm&ms': 'mnms'}.get(cfg.dataset_name.lower(),
                                                            cfg.dataset_name.lower())
    data = dataset(mode=cfg.dataset_mode, datasets=name, data_path=cfg.dataset_paths[name],
                   train_fraction=cfg.train_fraction,
                   normalization_epsilon=cfg.normalization_epsilon,
                   dim='3D', require_labels=True)
    wanted = set(cfg.cases or [])
    for index, (patient, _) in enumerate(data.samples):
        case_name = os.path.splitext(patient)[0]
        if wanted and case_name not in wanted:
            continue
        item = data[index]
        sequence = item['cmr']
        spacing = tuple(float(v) for v in item['info']['spacing_xyz'])
        train, test, stride = split_slices(sequence.shape[0], cfg.eval_protocol, cfg.holdout_stride)
        if len(test) == 0:
            print(f'{case_name}: 层数过少，holdout 协议下没有可计分的留出层，跳过')
            continue
        yield CaseData(
            name=case_name, sequence=sequence, spacing=spacing,
            thickness=cfg.slice_thickness_fwhm_spacing * spacing[2],
            reference_phase=cfg.reference_phase,
            time_indices=[int(t) for t in item['info']['time_indices']],
            train_index=train, test_index=test, stride=stride,
        )


def use_reference(repo_dir):
    """把参考仓库放到 sys.path 首位；同一进程只应加载一个参考仓库。

    多个参考仓库都有 utils/models 等同名顶层包，已缓存的同名模块若不来自该仓库则先移除。
    """
    repo_dir = Path(repo_dir).resolve()
    if not repo_dir.is_dir():
        raise FileNotFoundError(f'参考代码目录不存在：{repo_dir}')
    for entry in repo_dir.iterdir():
        name = entry.stem if entry.suffix == '.py' else entry.name
        module = sys.modules.get(name)
        if module is not None and repo_dir not in Path(getattr(module, '__file__', '') or '/').parents:
            for key in [k for k in sys.modules if k == name or k.startswith(name + '.')]:
                del sys.modules[key]
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    return repo_dir
