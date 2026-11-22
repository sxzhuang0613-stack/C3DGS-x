"""病例局部 spacing 坐标；PLY 几何记录与 checkpoint 使用同一契约。"""
import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from config_utils import render_options, RENDER_MODEL, ADDITIVE_RENDER_MODEL, RENDER_SEMANTICS

# 层面轮廓 FWHM 与高斯标准差之比 2·sqrt(2 ln 2)；与 diff_gaussian_rasterization_mri.FWHM_TO_SIGMA 相同，
# 此处不导入 CUDA 扩展，使 CPU 几何检查不依赖编译库
FWHM_TO_SIGMA = 1.0/(2.0*np.sqrt(2.0*np.log(2.0)))


def make_geometry(spacing_xyz, shape_shw, args, legacy=False):
    spacing = np.asarray(spacing_xyz, dtype=float).reshape(3)
    slices, height, width = map(int, shape_shw)
    extent = spacing * [width, height, max(slices-1, 1)]
    scale = float(extent.max()/2)
    bounds = list(args.bounds) if legacy else (extent/(2*scale)).tolist()
    rendering = render_options(args)
    rendering['bounds'] = bounds
    model = getattr(args, 'render_model', RENDER_MODEL)
    semantics = RENDER_SEMANTICS[model]
    if model == ADDITIVE_RENDER_MODEL:
        if legacy:
            raise ValueError('加和切片模型需要 spacing 局部坐标，不支持旧 legacy_box 几何')
        # 层厚 s（模型单位）：FWHM = k·sz，数据无层厚元数据，k 为实现选择；k=0 即零厚度截面
        rendering['render_model'] = model
        rendering['raster_options'].update(
            slice_sigma=float(FWHM_TO_SIGMA*args.slice_thickness_fwhm_spacing*spacing[2]/scale),
            amplitude_cutoff=float(args.amplitude_cutoff))
    return dict(version=0 if legacy else 1, frame='legacy_box' if legacy else 'spacing_stack_local',
                spacing_unit='mm_assumed_from_dataset', spacing_xyz=spacing.tolist(),
                shape_shw=[slices,height,width], scale_mm=None if legacy else scale,
                bounds=bounds, reference_phase=args.reference_phase,
                render_model=model,
                intensity_semantics=semantics['intensity'], opacity_mode=semantics['opacity'],
                rasterizer_api_version=semantics['api'],
                normalization_epsilon=args.normalization_epsilon, rendering=rendering)


def check_geometry(saved, current):
    """只比较坐标/观测条件，不把迭代数或输出路径当作实验几何。"""
    def same(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(same(a[k],b[k]) for k in a)
        if isinstance(a, (list,tuple)) and isinstance(b, (list,tuple)):
            return len(a) == len(b) and all(same(x,y) for x,y in zip(a,b))
        if isinstance(a, (int,float)) and isinstance(b, (int,float)):
            return bool(np.isclose(a,b,rtol=1e-8,atol=1e-10))
        return a == b
    if not same(saved,current):
        raise ValueError('模型几何与当前数据/配置不一致：请核对 spacing、shape、参考相位、归一化和渲染参数')


def attach_geometry(model, geometry):
    model.geometry = copy.deepcopy(geometry)
    model.bounds = list(geometry['bounds'])
    # 强度参数化随渲染语义确定：加和模型的 _intensity 为 softplus 前的原始参数
    model.intensity_softplus = geometry.get('intensity_semantics') == 'peak_density_softplus'


def save_model_ply(model, path, iteration=None):
    """旧 PLY 字段不变；哈希防止中断后将旧几何记录误配给新 PLY。"""
    model.save_ply(str(path)+'.tmp')
    os.replace(str(path)+'.tmp',path)
    record = dict(geometry=model.geometry,
                  geometry_reference_count=model.geometry_reference_count,
                  iteration=iteration,
                  ply_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    sidecar = str(path)+'.geometry.json'
    with open(sidecar+'.tmp','w',encoding='utf-8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2,allow_nan=False)
    os.replace(sidecar+'.tmp',sidecar)


def read_model_geometry(path):
    sidecar = Path(str(path)+'.geometry.json')
    if not sidecar.is_file():
        return None
    record = json.loads(sidecar.read_text(encoding='utf-8'))
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != record['ply_sha256']:
        raise ValueError(f'PLY 与几何记录不匹配：{sidecar}')
    return record
