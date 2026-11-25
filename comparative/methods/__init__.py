"""对比方法注册表。

每个方法模块提供统一接口：
    fit(case, cfg, method_cfg, out_dir, **options) -> dict
在 case.train_index 的层上逐病例拟合并返回预测（逐病例优化不保存权重；out_dir 供方法写中间文件）：
    query_index: [Q] 连续层索引（覆盖 0..S-1，步长 cfg.interp_step_size）
    volume:      [Q,H,W] 静态方法在对应层位、按各自前向（层面轮廓/PSF）模型渲染的预测
    frame:       volume 对应的帧
    sequence:    可选 [F,Q,H,W]，frames 为对应帧列表（Dyna3DGR full）
    iterations:  实际迭代数；elapsed_seconds 可选，覆盖 fit 计时（C3DGS 训练在自身入口完成）
模块延迟导入：各参考仓库的同名顶层包只在被选中的方法里加载。
"""
import importlib

METHODS = {
    'c3dgs': 'methods.c3dgs',          # 本方法：读取 C3DGS 自身训练入口按同一协议保存的最终 PLY
    'gaussiansvr': 'methods.gaussiansvr',
    'gsvr': 'methods.gsvr',
    'nesvor': 'methods.nesvor',
    'eclare': 'methods.eclare',
    'dyna3dgr': 'methods.dyna3dgr',
}


def load_method(name):
    """按名称导入方法模块。"""
    if name not in METHODS:
        raise ValueError(f'未知方法 {name}，可选：{sorted(METHODS)}')
    return importlib.import_module(METHODS[name])


def method_tag(name, stage=None):
    """输出目录名；Dyna3DGR 按阶段关键字区分。"""
    return f'{name}_{stage}' if name == 'dyna3dgr' else name
