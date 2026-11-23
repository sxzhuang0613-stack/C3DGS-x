"""批量高斯操作回归；CPU 数值/状态检查与短 CUDA 集成分开执行。"""
from pathlib import Path
import math
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from utils.config_utils import load_config
from utils.case_geometry import make_geometry, attach_geometry
from Gaussians.GaussianModel_structural import GaussianModel
from stage1_test_support import NATIVE_AVAILABLE, alpha_train_config


class StructuralTests(unittest.TestCase):
    def setUp(self):
        self.args = alpha_train_config()
        self.args.structural_min_observations = 1
        self.args.structural_grad_multiplier = 0.
        self.args.structural_grad_threshold = .1
        # 测试几何为 1/16 模型单位每像素；沿用 2 像素 Split 门限以保持既有算例语义。
        self.args.structural_split_sigma_pixels = 2.
        self.args.structural_size_max_screen_radius = 0.
        self.args.structural_size_max_scale_ratio = 0.
        self.images = torch.zeros(3, 32, 32)
        # Split 偏移随机；固定种子避免观测框剪枝偶然截掉子点。
        torch.manual_seed(0)
        # 默认审计：每点贡献 10 个像素当量、平均残差 1，增殖不受残差门控与支撑门影响。
        self.audit = None

    def model(self, scales=(.15, .15, .15), count=4, device='cpu'):
        # 默认 0.15 模型单位 = 2.4 输出像素，大于测试 Split 门限 2 像素；0.05 = 0.8 像素为小点。
        model = GaussianModel()
        model._xyz = nn.Parameter(torch.zeros(count, 3, device=device))
        model._scaling = nn.Parameter(torch.tensor(scales, device=device).log().repeat(count, 1))
        model._rotation = nn.Parameter(torch.tensor([[1., 0., 0., 0.]], device=device).repeat(count, 1))
        values = torch.arange(1, count+1, device=device).float()
        values[::2] *= -1
        model._intensity = nn.Parameter(values[:, None])
        attach_geometry(model, make_geometry([1., 1., 8.], self.images.shape, self.args))
        model.training_setup(self.args)
        model.xyz_gradient_accum.fill_(1.)
        model.denom.fill_(1.)
        # 模拟已训练的 Adam 状态，逐行不同以检验索引对应关系。
        for group in model.optimizer.param_groups:
            parameter = group['params'][0]
            moment = torch.arange(1, count+1, device=device).reshape(count, *([1]*(parameter.ndim-1))).expand_as(parameter).clone().float()
            model.optimizer.state[parameter] = dict(step=torch.tensor(7.), exp_avg=moment,
                                                     exp_avg_sq=moment.square())
        return model

    def step(self, model, iteration=1000, densify=True, audit=None):
        audit = audit or self.audit
        if audit is None:
            weight = torch.full((len(model.get_xyz),), 10., device=model.get_xyz.device)
            audit = (weight, weight.clone())
        with mock.patch.object(model, 'contribution_audit', return_value=audit):
            return model.structural_step(self.images.to(model.get_xyz.device), iteration, densify=densify)

    def test_standalone_and_no_candidate_render_or_solver(self):
        self.assertEqual(GaussianModel.__bases__, (object,))
        model = self.model()
        with mock.patch('Renderer.render', side_effect=AssertionError('不应渲染')), \
                mock.patch('torch.linalg.svd', side_effect=AssertionError('不应拟合')), \
                mock.patch('torch.topk', side_effect=AssertionError('不应 Top-K')):
            report = self.step(model)
        self.assertEqual(report['split'], 4)
        self.assertEqual(report['after'], 8)

    def test_small_points_are_not_cloned(self):
        """不做 Clone：通过梯度/残差门的小点（面内 sigma 不超过 Split 门限）不增殖，参数与动量保持不变。"""
        model = self.model(scales=(.05, .05, .05))
        parameter = model._xyz
        report = self.step(model)
        self.assertNotIn('clone', report)
        self.assertEqual((report['selected'], report['split'], report['after']), (0, 0, 4))
        self.assertIs(model._xyz, parameter)
        self.assertEqual(model.optimizer.state[model._xyz]['step'], 7)

    def test_split_matches_classic_sampling_and_all_axis_shrink(self):
        """无局部轴与法向对齐时，Split 为经典随机三轴偏移与全轴缩放，状态继承正确。"""
        model = self.model(scales=(.3, .1, .2))
        with torch.no_grad():
            model._rotation.copy_(torch.tensor([[.9, .3, .3, 0.]]).repeat(4, 1))
            # 父点保持在观测框内，避免框剪枝干扰 Split 公式检查。
            model._xyz.copy_(torch.arange(12).reshape(4, 3)*.02)
        parent_xyz = model.get_xyz.detach().clone()
        scales = model.get_scaling.detach().repeat(2, 1)
        intensity = model.get_intensity.detach().clone()
        # 固定 RNG 独立构造经典公式，检查三个局部轴均参与随机偏移和缩放。
        torch.manual_seed(81)
        state = torch.get_rng_state()
        samples = torch.normal(torch.zeros_like(scales), scales)
        rotation = model._rotation_matrix(model._rotation).detach().repeat(2, 1, 1)
        expected_xyz = parent_xyz.repeat(2, 1)+(rotation@samples[:, :, None]).squeeze(-1)
        torch.set_rng_state(state)
        report = self.step(model)
        self.assertEqual(report['split'], 4)
        torch.testing.assert_close(model.get_xyz, expected_xyz)
        torch.testing.assert_close(model.get_intensity, intensity.repeat(2, 1))
        torch.testing.assert_close(model.get_scaling, scales/1.6)
        for group in model.optimizer.param_groups:
            self.assertEqual(torch.count_nonzero(model.optimizer.state[group['params'][0]]['exp_avg']), 0)

    def test_layer_spacing_does_not_force_normal_split(self):
        """面内小、法向长的点不按层间距拆分。"""
        model = self.model(scales=(.05, .05, .8))
        report = self.step(model)
        self.assertEqual((report['split'], report['after']), (0, 4))

    def test_all_candidates_grow_without_capacity_or_cycle_budget(self):
        """所有合格父点均拆分，配置不再包含容量或周期预算；子点继承强度、动量清零。"""
        self.assertFalse(hasattr(self.args, 'max_num_pts'))
        self.assertFalse(hasattr(self.args, 'structural_growth_fraction'))
        model = self.model(count=64)
        intensity = model.get_intensity.detach().clone()
        report = self.step(model)
        self.assertEqual((report['split'], report['after']), (64, 128))
        torch.testing.assert_close(model.get_intensity, intensity.repeat(2, 1))
        for key in ('_xyz', '_intensity'):
            self.assertEqual(torch.count_nonzero(model.optimizer.state[getattr(model, key)]['exp_avg']), 0)

    def test_dark_and_unobserved_points_are_not_pruned(self):
        model = self.model()
        model.xyz_gradient_accum.zero_()
        with torch.no_grad():model._intensity[:2] = 0.
        model.denom[1] = 0
        before = model._xyz
        report = self.step(model)
        self.assertEqual((report['pruned'], report['after']), (0,4))
        self.assertIs(model._xyz,before)

    def test_invalid_point_is_removed_without_normalizing_bad_rotation(self):
        model = self.model()
        model.xyz_gradient_accum.zero_()
        with torch.no_grad():model._rotation[0] = 0.
        self.assertEqual(self.step(model)['pruned'], 1)
        self.assertTrue(torch.isfinite(model.get_rotation).all())
        with torch.no_grad():model._rotation.zero_()
        with self.assertRaises(FloatingPointError):self.step(model)

    def test_gradient_units_relative_threshold_and_noop_parameter_identity(self):
        model = self.model()
        model.xyz_gradient_accum.zero_();model.denom.zero_()
        model._xyz.grad = torch.ones_like(model._xyz)
        model.add_densification_stats_3d(torch.tensor([True, False, True, False]))
        expected = model._voxel_spacing().norm()
        torch.testing.assert_close(model.xyz_gradient_accum[:, 0], torch.tensor([expected, 0., expected, 0.]))
        model.xyz_gradient_accum[:, 0] = torch.tensor([1., 2., 4., 8.])
        model.denom.fill_(1.)
        self.args.structural_grad_multiplier = 2.
        report = self.step(model)
        self.assertEqual(report['grad_threshold'], 4.)
        self.assertEqual(report['selected'], 2)
        parameter = model._xyz
        model.xyz_gradient_accum.zero_()
        self.assertEqual(self.step(model)['selected'], 0)
        self.assertIs(model._xyz, parameter)
        self.assertEqual(model.denom.sum(), 0)

    def test_checkpoint_resume_after_batch_structure_preserves_rng_and_adam(self):
        from GetInitialGaussian import save_training_checkpoint, restore_training_checkpoint, check_resume_config
        model = self.model();self.step(model)
        self.args.device = 'cpu'
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(torch.cuda, 'get_rng_state_all', return_value=[]), \
                mock.patch.object(torch.cuda, 'set_rng_state_all'):
            path = str(Path(directory)/'state.pth')
            save_training_checkpoint(path, model, 1000, [], {}, self.args)
            expected = torch.rand(3)
            state = torch.load(path, weights_only=False)
            other = GaussianModel();restore_training_checkpoint(state, other, self.args)
            torch.testing.assert_close(torch.rand(3), expected)
            torch.testing.assert_close(other._xyz, model._xyz)
            torch.testing.assert_close(other.optimizer.state[other._scaling]['exp_avg'],
                                       model.optimizer.state[model._scaling]['exp_avg'])
            self.assertEqual(other.geometry_reference_count, 4)
            check_resume_config(state['config'], self.args)
            state['config'].pop('structural_algorithm')
            with self.assertRaisesRegex(ValueError, '结构算法'):check_resume_config(state['config'], self.args)

    def test_voxel_gradient_and_pixel_scale_are_coordinate_scale_invariant(self):
        first, second = self.model(), self.model()
        geometry = dict(second.geometry, scale_mm=second.geometry['scale_mm']/2,
                        bounds=[value for value in second.bounds])
        attach_geometry(second, geometry)
        with torch.no_grad():
            second._xyz.mul_(2)
            second._scaling.add_(torch.tensor(2.).log())
        second.training_setup(self.args)
        first.xyz_gradient_accum.zero_();first.denom.zero_()
        first._xyz.grad = torch.ones_like(first._xyz)
        second._xyz.grad = torch.ones_like(second._xyz)*.5
        visible = torch.ones(4, dtype=torch.bool)
        first.add_densification_stats_3d(visible)
        second.add_densification_stats_3d(visible)
        torch.testing.assert_close(first.xyz_gradient_accum, second.xyz_gradient_accum)
        first_report, second_report = self.step(first), self.step(second)
        self.assertAlmostEqual(first_report['sigma_pixels_median'], second_report['sigma_pixels_median'], places=6)
        self.assertEqual(first_report['selected'], second_report['selected'])

    def test_mixed_split_prune_commits_aligned_states_once(self):
        model = self.model(scales=(.05, .05, .05))
        with torch.no_grad():
            model._scaling[[1, 3]] = torch.tensor([.3, .1, .2]).log()
            model._xyz[0] = float('nan')
        with mock.patch.object(model, '_replace_structure', wraps=model._replace_structure) as commit:
            report = self.step(model)
        self.assertEqual(commit.call_count, 1)
        self.assertEqual((report['split'], report['pruned'], report['after']), (2, 1, 5))
        torch.testing.assert_close(model.optimizer.state[model._scaling]['exp_avg'][:, 0],
                                   torch.tensor([3., 0., 0., 0., 0.]))
        self.assertEqual(len(model.optimizer.state), 4)
        for group in model.optimizer.param_groups:
            parameter = group['params'][0]
            self.assertEqual(parameter.shape, model.optimizer.state[parameter]['exp_avg'].shape)
        # 新参数下一步可继续使用 Adam，不能留下旧参数引用。
        sum(group['params'][0].square().sum() for group in model.optimizer.param_groups).backward()
        model.optimizer.step()

    def test_split_respects_log_scale_floor(self):
        self.args.log_scale_min = -1.21
        model = self.model(scales=(.3, .3, .3))
        self.assertEqual(self.step(model)['selected'], 0)

    def test_split_floor_checks_smallest_axis(self):
        """全轴收缩时，最细轴也必须满足既有 log_scale 下界。"""
        self.args.log_scale_min = math.log(.08)
        model = self.model(scales=(.3, .1, .5))
        self.assertEqual(self.step(model)['selected'], 0)

    def test_screen_radius_statistics_use_observed_maximum(self):
        """累计实际观察到的半径，未观察点和较小的新半径不覆盖历史最大值。"""
        model = self.model()
        visible = torch.tensor([True, False, True, False])
        model.add_densification_stats_3d(visible, torch.tensor([3, 50, 7, 9]))
        model.add_densification_stats_3d(visible, torch.tensor([2, 60, 9, 8]))
        torch.testing.assert_close(model.max_radii2D, torch.tensor([3., 0., 9., 0.]))

    def test_oversize_points_are_split_not_pruned_after_start_iteration(self):
        """世界尺度或屏幕半径超限的点强制 Split（不删除、不受梯度门控），阈值相等不拆，启用步之前不拆。"""
        self.args.structural_size_split_from_iter = 1001
        self.args.structural_size_max_screen_radius = 20.
        self.args.structural_size_max_scale_ratio = .1
        model = self.model(scales=(.05, .05, .05))
        model.xyz_gradient_accum.zero_()
        with torch.no_grad():
            model._scaling[1, 0] = math.log(.11*max(model.bounds))
        model.max_radii2D[2:] = torch.tensor([21., 20.])
        report = self.step(model)
        self.assertEqual((report['split_size'], report['after']), (0, 4))
        # 每周期会清空统计，第二轮重新累计观察半径。
        model.max_radii2D[2:] = torch.tensor([21., 20.])
        self.args.structural_size_split_from_iter = 1000
        with mock.patch.object(model, '_replace_structure', wraps=model._replace_structure) as commit:
            report = self.step(model)
        self.assertEqual((report['split_size'], report['split'], report['pruned'], report['after']), (2, 2, 0, 6))
        self.assertEqual(commit.call_count, 1)
        torch.testing.assert_close(model.optimizer.state[model._xyz]['exp_avg'][:2, 0], torch.tensor([1., 4.]))
        # 超限父点的子点所有轴除以 1.6，针状点最短轴截到尺度下界。
        floor = math.exp(model.log_scale_floor())
        # 子点按 [父 1, 父 2, 父 1, 父 2] 排列。
        torch.testing.assert_close(model.get_scaling[[2, 4]], torch.tensor([.11*max(model.bounds)/1.6, .05/1.6, .05/1.6]
                                                                            ).clamp_min(floor).repeat(2, 1))
        torch.testing.assert_close(model.get_scaling[[3, 5]], torch.full((2, 3), .05/1.6).clamp_min(floor))

    def test_oversize_children_are_kept_and_split_again(self):
        """仍超限的子点保留到下个周期继续拆分，直到降到上限以下；全程不删除。"""
        self.args.structural_size_split_from_iter = 0
        self.args.structural_size_max_scale_ratio = .1
        model = self.model(scales=(.15, .15, .15), count=2)
        model.xyz_gradient_accum.zero_()
        with torch.no_grad():model._scaling[1] = math.log(.3)
        report = self.step(model)
        self.assertEqual((report['split'], report['pruned'], report['after']), (2, 0, 4))
        torch.testing.assert_close(model.get_scaling.max(1).values.sort().values,
                                   torch.tensor([.15, .15, .3, .3])/1.6)
        torch.testing.assert_close(model.get_intensity[:, 0].sort().values, torch.tensor([-1., -1., 2., 2.]))
        report = self.step(model)
        self.assertEqual((report['split_size'], report['pruned'], report['after']), (2, 0, 6))
        for group in model.optimizer.param_groups:
            parameter = group['params'][0]
            self.assertEqual(parameter.shape, model.optimizer.state[parameter]['exp_avg'].shape)

    def test_world_size_limit_ignores_normal_elongation(self):
        """世界尺度只看面内投影标准差：沿法向任意拉长的点不拆，面内过大的点拆分。"""
        self.args.structural_size_split_from_iter = 0
        self.args.structural_size_max_scale_ratio = .1
        model = self.model(scales=(.05, .05, 2.), count=3)
        model.xyz_gradient_accum.zero_()
        with torch.no_grad():model._scaling[2] = torch.tensor([.11*max(model.bounds), .05, .05]).log()
        report = self.step(model)
        self.assertEqual((report['split_size'], report['after']), (1, 4))

    def test_prune_minimum_is_checked_before_commit(self):
        """剪枝不能留下空模型；失败时保留原参数和优化器引用。"""
        model = self.model(count=1)
        model.xyz_gradient_accum.zero_()
        with torch.no_grad():model._xyz[0, 0] = 10.
        parameter = model._xyz
        with self.assertRaisesRegex(FloatingPointError, '剪枝后'):
            self.step(model)
        self.assertIs(model._xyz, parameter)
        self.assertIs(model.optimizer.param_groups[0]['params'][0], parameter)

    def test_split_shrinks_all_axes_even_when_normal_aligned(self):
        """经典 3DGS Split：与法向对齐的轴同样缩小 1.6 倍，不保留层间方向尺度。"""
        model = self.model(scales=(.3, .3, .2))
        report = self.step(model)
        self.assertEqual(report['split'], 4)
        torch.testing.assert_close(model.get_scaling, torch.tensor([[.3, .3, .2]]).repeat(8, 1)/1.6)

    def test_gradient_threshold_is_frozen_after_first_cycle(self):
        """相对中位数只标定一次；后续梯度整体下降时不再按固定比例选点。"""
        self.args.structural_grad_threshold = 0.
        self.args.structural_grad_multiplier = 2.
        model = self.model()
        model.xyz_gradient_accum[:, 0] = torch.tensor([1., 2., 4., 8.])
        self.assertEqual(self.step(model)['grad_threshold'], 4.)
        model = self.model()
        model.structural_grad_reference = 4.
        model.xyz_gradient_accum[:, 0] = torch.tensor([.1, .2, .4, .8])
        report = self.step(model)
        self.assertEqual((report['grad_threshold'], report['selected']), (4., 0))

    def test_residual_gate_stops_growth_after_convergence(self):
        """覆盖区平均残差不超过 tau 时，即使梯度超过门限也不增殖。"""
        self.args.structural_residual_threshold = .01
        model = self.model()
        weight = torch.full((4,), 10.)
        report = self.step(model, audit=(weight, weight*torch.tensor([.005, .009, .02, .5])))
        self.assertEqual((report['residual_high'], report['split'], report['after']), (2, 2, 6))

    def test_low_contribution_points_are_kept_but_not_densified(self):
        """不按贡献删除（层间点可能服务插层）；W 低于支撑门的点保留但不增殖。"""
        model = self.model()
        weight = torch.tensor([0., .4, .5, 3.])
        report = self.step(model, audit=(weight, weight))
        self.assertEqual((report['pruned'], report['split'], report['after']), (0, 2, 6))
        torch.testing.assert_close(model.optimizer.state[model._xyz]['exp_avg'][:2, 0], torch.tensor([1., 2.]))
        # 新子点来自 W >= 0.5 的第 3、4 个父点（父点被拆分删除，W 低的点保留）。
        torch.testing.assert_close(model.get_intensity[2:, 0], torch.tensor([-3., 4., -3., 4.]))

    def test_points_outside_observation_box_are_pruned(self):
        """观测框为 [W*sx,H*sy,S*sz]/2 加半体素；框外旧点和框外子点均剪除。"""
        model = self.model()
        model.xyz_gradient_accum.zero_()
        box = model._observation_box()
        torch.testing.assert_close(box, model._voxel_spacing()*torch.tensor([32., 32., 3.])/2)
        limit = box+.5*model._voxel_spacing()
        with torch.no_grad():
            model._xyz[1, 0] = limit[0]*1.01
            model._xyz[2, 2] = -limit[2]*1.01
            model._xyz[3, 2] = limit[2]*.99
        report = self.step(model)
        self.assertEqual((report['pruned_box'], report['after']), (2, 2))

    def test_prune_only_audit_does_not_grow(self):
        """增殖窗口后的审计不做误差驱动增殖，也不删除低贡献点。"""
        model = self.model()
        weight = torch.tensor([0., 10., 10., 10.])
        report = self.step(model, densify=False, audit=(weight, weight))
        self.assertEqual((report['selected'], report['pruned'], report['after']), (0, 0, 4))

    def test_inplane_gradient_statistic_ignores_z(self):
        """经典模式统计量只含面内分量。"""
        self.args.structural_inplane_gradient = True
        model = self.model()
        model.xyz_gradient_accum.zero_(); model.denom.zero_()
        model._xyz.grad = torch.tensor([[3., 4., 100.]]).repeat(4, 1)
        model.add_densification_stats_3d(torch.ones(4, dtype=torch.bool))
        torch.testing.assert_close(model.xyz_gradient_accum[:, 0], torch.full((4,), 5*float(model._voxel_spacing()[0])))

    def test_classic_mode_skips_audit_and_residual_gate(self):
        """关闭残差门时不调用贡献审计（无额外渲染），仅按梯度门限增殖。"""
        self.args.structural_residual_gate = False
        model = self.model()
        with mock.patch.object(model, 'contribution_audit', side_effect=AssertionError('不应审计')):
            report = model.structural_step(self.images, 1000)
        self.assertEqual((report['split'], report['after'], report['residual_high']), (4, 8, 0))

    def test_split_switch(self):
        """关闭 Split 时误差驱动与尺寸超限拆分都不执行，超限点保留原样。"""
        self.args.structural_size_split_from_iter = 0
        self.args.structural_size_max_scale_ratio = .1
        def mixed():
            model = self.model(scales=(.05, .05, .05))
            with torch.no_grad():
                model._scaling[1] = torch.tensor([.3, .1, .2]).log()     # 误差驱动 Split
                model._scaling[2] = torch.tensor([.11, .05, .05]).log()  # 尺寸超限
            return model
        report = self.step(mixed())
        self.assertEqual((report['split'], report['after']), (2, 6))
        self.args.structural_enable_split = False
        report = self.step(mixed())
        self.assertEqual((report['split'], report['after']), (0, 4))

    def test_scale_floor_is_in_output_pixels(self):
        """尺度下界为 min_sigma_pixels 个面内像素，log_scale_min 只作外层安全边界。"""
        model = self.model()
        self.assertAlmostEqual(model.log_scale_floor(), math.log(.5*float(model._voxel_spacing()[0])))
        self.args.log_scale_min = 0.
        model = self.model()
        self.assertEqual(model.log_scale_floor(), 0.)

    def test_short_cuda_render_update_structure_update(self):
        if not torch.cuda.is_available() or not NATIVE_AVAILABLE:self.skipTest('需要 CUDA 与新库')
        from utils.Renderer import render
        model = self.model(scales=(.3, .1, .2), device='cuda')
        for iteration in range(4):
            output = render(model, [17, 19], 1, 2, 3, **model.geometry['rendering'])
            loss = (output['render']-.1).square().mean()
            loss.backward()
            model.add_densification_stats_3d(output['visibility_filter'], output['radii'])
            self.assertGreater(model.max_radii2D.max().item(), 0.)
            model.optimizer.step();model.optimizer.zero_grad(set_to_none=True)
            if iteration == 1:
                model.xyz_gradient_accum.fill_(1.);model.denom.fill_(1.)
                model.structural_step(self.images.cuda(), 1000, model.geometry['rendering'])
        for group in model.optimizer.param_groups:
            self.assertTrue(torch.isfinite(group['params'][0]).all())
            self.assertEqual(group['params'][0].shape, model.optimizer.state[group['params'][0]]['exp_avg'].shape)
        torch.cuda.synchronize()


if __name__ == '__main__':
    unittest.main()
