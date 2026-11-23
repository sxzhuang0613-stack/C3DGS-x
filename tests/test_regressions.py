"""配置和接口回归可在 CPU 运行；真实张量及 CUDA 验证按依赖明确跳过。"""
import ast
import importlib.util
import math
import numbers
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def source_function(filename, name, namespace, class_name=None):
    tree = ast.parse((ROOT / filename).read_text())
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    function = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]


@unittest.skipUnless(importlib.util.find_spec('yaml'), '需要 PyYAML')
class ConfigTests(unittest.TestCase):
    def test_configs_have_numeric_optimizer_and_renderer_values(self):
        from utils.config_utils import load_config
        args = load_config(ROOT / 'config_train.yaml')
        for key in ('adam_eps', 'stage2_adam_eps', 'init_ridge_relative'):
            self.assertIsInstance(getattr(args, key), float)
        self.assertEqual(args.interpolate_iter, 5000)
        self.assertLess(args.interpolate_iter, args.densify_until_iter)
        self.assertFalse(hasattr(args, 'max_num_pts'))
        self.assertFalse(hasattr(args, 'structural_growth_fraction'))

    def test_test_config_only_contains_stage1_parameters(self):
        from utils.config_utils import load_config
        args = load_config(ROOT / 'config_test.yaml')
        self.assertFalse(any(key.startswith('stage2') for key in vars(args)))
        self.assertFalse(hasattr(args, 'iters'))
        self.assertIn('{case}', args.stage1_ply_path)

    def test_duplicate_keys_are_rejected(self):
        from utils.config_utils import load_config
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text('device: cuda:0\ndevice: cuda:1\n')
            with self.assertRaisesRegex(ValueError, '重复'):
                load_config(path)

    def test_old_structure_options_are_rejected(self):
        """已取消的容量、增长预算和旧 Split 参数不能以静默配置残留。"""
        import yaml
        from utils.config_utils import load_config
        original = yaml.safe_load((ROOT/'config_train.yaml').read_text())
        for key, value in (('max_num_pts', 200000), ('structural_growth_fraction', .05),
                           ('structural_split_fraction', .35),
                           ('structural_algorithm', 'batch_normalized_v1')):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'config.yaml'
                path.write_text(yaml.safe_dump(dict(original, **{key:value})))
                with self.assertRaises(ValueError):load_config(path)

    def test_pruning_thresholds_are_validated(self):
        """过大尺寸参数必须有限非负，启用步数必须为非负整数。"""
        import yaml
        from utils.config_utils import load_config
        original = yaml.safe_load((ROOT/'config_train.yaml').read_text())
        for key, value in (('structural_size_max_screen_radius', -1.),
                           ('structural_size_max_scale_ratio', float('nan')),
                           ('structural_size_split_from_iter', -1)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'config.yaml'
                path.write_text(yaml.safe_dump(dict(original, **{key:value})))
                with self.assertRaises(ValueError):load_config(path)

    def test_config_paths_do_not_depend_on_cwd(self):
        from utils.config_utils import load_config
        args = load_config(ROOT / 'config_test.yaml')
        self.assertEqual(Path(args.output_path), ROOT / 'output/evaluation')

    def test_active_entry_fields_are_all_configured(self):
        from utils.config_utils import load_config
        for config, files in (
            ('config_train.yaml', ['GetInitialGaussian.py', 'Motion.py', 'temp.py',
                                   'config_utils.py',
                                   'Gaussians/initialization.py', 'Gaussians/GaussianModel_structural.py',
                                   'case_geometry.py']),
            ('config_test.yaml', ['evaluate.py', 'config_utils.py', 'case_geometry.py']),
        ):
            args = load_config(ROOT / config)
            used = set()
            for filename in files:
                tree = ast.parse((ROOT / filename).read_text())
                used.update(n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                            and isinstance(n.value, ast.Name) and n.value.id == 'args')
            self.assertFalse(used - set(vars(args)))


from test_stage1_normalized import RendererTests as Stage1RendererTests


class RendererTests(Stage1RendererTests):
    pass


class LegacyGeometryTests(unittest.TestCase):
    def test_continuous_index_validation(self):
        # 运行原始校验；在进入 Tensor 几何构造时停止，不冒充几何数值测试。
        class GeometryReached(Exception): pass
        def reached(*args, **kwargs): raise GeometryReached
        namespace = {'numbers': numbers, 'math': math,
                     'torch': types.SimpleNamespace(as_tensor=reached, float64=None)}
        source_function('diff-gaussian-slice-rendering/diff_gaussian_slice_rendering/__init__.py',
                        '_positive_int', namespace)
        axis = source_function(
            'diff-gaussian-slice-rendering/diff_gaussian_slice_rendering/__init__.py',
            'from_axis', namespace, class_name='GaussianRasterizationSettings')
        for index, slices in [(0., 1), (0.25, 3), (2., 3)]:
            with self.subTest(index=index), self.assertRaises(GeometryReached):
                axis(None, 16, 16, slice_idx=index, num_slices=slices)
        for index in (-0.1, 2.1, float('nan'), float('inf'), True):
            with self.subTest(index=index), self.assertRaises(ValueError):
                axis(None, 16, 16, slice_idx=index, num_slices=3)


HAS_TENSORS = all(importlib.util.find_spec(name) is not None for name in ('torch', 'numpy', 'plyfile', 'yaml'))


@unittest.skipUnless(HAS_TENSORS, '需要 PyTorch、NumPy、plyfile 和 PyYAML')
class TensorTests(unittest.TestCase):
    def model(self, scale=0.1, count=4):
        import torch
        from utils.config_utils import load_config
        from Gaussians.GaussianModel_structural import GaussianModel
        from utils.case_geometry import make_geometry, attach_geometry
        from stage1_test_support import alpha_train_config
        args = alpha_train_config()
        args.device = 'cpu'
        args.structural_min_observations = 1
        args.structural_grad_multiplier = 0.
        model = GaussianModel()
        for name, value in (
            ('_xyz', torch.zeros(count, 3)), ('_intensity', torch.full((count, 1), 0.5)),
            ('_scaling', torch.full((count, 3), math.log(scale))),
            ('_rotation', torch.tensor([[1., 0., 0., 0.]]).repeat(count, 1)),
        ):
            setattr(model, name, torch.nn.Parameter(value))
        attach_geometry(model, make_geometry([1., 1., 8.], [3, 32, 32], args))
        model.training_setup(args)
        return model, args

    def test_split_has_no_capacity_limit_and_small_points_are_not_cloned(self):
        """Split 处理全部大点候选，统计张量随最终点数更新；小点不做 Clone。"""
        import torch
        for scale, expected in ((0.005, 4), (0.2, 8)):
            model, args = self.model(scale=scale)
            model.xyz_gradient_accum.fill_(1)
            model.denom.fill_(1)
            weight = torch.full((4,), 10.)
            with mock.patch.object(model, 'contribution_audit', return_value=(weight, weight)):
                model.structural_step(torch.zeros(3, 32, 32), 1000)
            self.assertEqual(model.get_xyz.shape[0], expected)
            self.assertEqual(model.denom.shape[0], expected)
            self.assertTrue(torch.isfinite(model.get_scaling).all())

    def test_new_ply_round_trip(self):
        import torch
        from Gaussians.GaussianModel_structural import GaussianModel
        model, _ = self.model()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'model.ply')
            model.save_ply(path)
            other = GaussianModel()
            other.load_ply(path, device='cpu')
        self.assertTrue(torch.equal(model.get_intensity, other.get_intensity))
        self.assertTrue(torch.equal(model.get_xyz, other.get_xyz))

    def test_training_checkpoint_restores_optimizer_and_randomness(self):
        import torch
        from GetInitialGaussian import save_training_checkpoint, restore_training_checkpoint
        model, args = self.model()
        model.get_xyz.sum().backward()
        model.optimizer.step()
        model.xyz_gradient_accum.fill_(3)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(torch.cuda, 'get_rng_state_all', return_value=[]), \
                mock.patch.object(torch.cuda, 'set_rng_state_all'):
            path = str(Path(directory) / 'state.pth')
            save_training_checkpoint(path, model, 123, [123], {'loss': [0.5]}, args)
            expected_random = random.random()
            state = torch.load(path, weights_only=False)
            other, _ = self.model(count=2)
            restore_training_checkpoint(state, other, args)
            self.assertEqual(random.random(), expected_random)
            self.assertTrue(torch.equal(model.get_xyz, other.get_xyz))
            self.assertTrue(torch.equal(other.xyz_gradient_accum, model.xyz_gradient_accum))
            self.assertEqual(state['iteration'], 123)
            self.assertTrue(torch.equal(model.optimizer.state[model._xyz]['exp_avg'],
                                        other.optimizer.state[other._xyz]['exp_avg']))

    def test_ssim_identity(self):
        import torch
        from utils.CostFunctions import ssim
        image = torch.rand(2, 1, 16, 16)
        self.assertAlmostEqual(ssim(image, image).item(), 1., places=6)

    def test_cuda_covariance_path_has_gradients(self):
        """当前 Renderer 的预计算协方差能向实际切片几何反传梯度。"""
        import torch
        from stage1_test_support import NATIVE_AVAILABLE
        from utils.Renderer import render
        if not torch.cuda.is_available() or not NATIVE_AVAILABLE:
            self.skipTest('需要 CUDA 与 API_VERSION=2 的切片扩展')
        model,_=self.model(count=1)
        model._xyz=torch.nn.Parameter(model.get_xyz.detach().cuda())
        model._intensity=torch.nn.Parameter(model.get_intensity.detach().cuda())
        covariance=torch.tensor([[.1,0.,0.,.1,0.,.1]],device='cuda',requires_grad=True)
        image=render(model,[16,16],.25,2,2,covariance=covariance)['render']
        image.sum().backward()
        for value in (model._xyz,model._intensity,covariance):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
        self.assertGreater(covariance.grad.abs().sum().item(),0.)


if __name__ == '__main__':
    unittest.main()
