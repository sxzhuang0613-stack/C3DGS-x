"""合成 spacing/初始化回归；不读取患者像素、不进行完整训练。"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if not all(importlib.util.find_spec(name) for name in ('torch','numpy','yaml','plyfile')):
    raise unittest.SkipTest('需要已有 PyTorch/NumPy/PyYAML/plyfile')
import numpy as np
import torch
from utils.config_utils import load_config, ssim_options
from utils.case_geometry import make_geometry, check_geometry, attach_geometry, save_model_ply, read_model_geometry
from utils.CostFunctions import anisotropic_scale_loss
from Gaussians.GaussianModel_structural import GaussianModel
from Gaussians.initialization import IntensityOperator
from stage1_test_support import NATIVE_AVAILABLE, reference_render, alpha_train_config


def reference_linear_render(model,img_size,slice_idx,axis,num_slices,**kwargs):
    return reference_render(model,img_size,slice_idx,axis,num_slices,**kwargs)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.args = alpha_train_config()

    def test_spacing_coordinates_agree_with_renderer_including_single_slice(self):
        for slices in (1,3):
            geometry = make_geometry([1.171875,.75,10.],[slices,128,128],self.args)
            for k in range(slices):
                points = torch.tensor([[0.,0.],[127.,127.],[35.,81.]],dtype=torch.float64)
                bx,by,bz = geometry['bounds']
                actual = torch.stack((((points[:,0]+.5)*2/128-1)*bx,
                                      ((points[:,1]+.5)*2/128-1)*by,
                                      torch.full((3,),0. if slices==1 else -bz+2*bz*k/(slices-1))),1)
                expected = torch.stack(((points[:,0]+.5-64)*1.171875,
                    (points[:,1]+.5-64)*.75,torch.full((3,),(k-(slices-1)/2)*10,dtype=torch.float64)),1)/geometry['scale_mm']
                torch.testing.assert_close(actual,expected)

    def test_geometry_rejects_changed_spacing_phase_and_rendering(self):
        geometry = make_geometry([1.,1.,8.],[3,9,13],self.args)
        for field,value in [('spacing_xyz',[1.,1.,10.]),('reference_phase',1)]:
            changed = dict(geometry,**{field:value})
            with self.assertRaisesRegex(ValueError,'几何'):
                check_geometry(geometry,changed)
        self.assertEqual(make_geometry([1.,1.,8.],[3,9,13],self.args,legacy=True)['bounds'],self.args.bounds)

    def test_resume_allows_extension_but_rejects_optimizer_change(self):
        from GetInitialGaussian import check_resume_config
        saved = dict(vars(self.args))
        self.args.iters += 1
        self.args.lambda_volume += .01
        check_resume_config(saved,self.args)
        self.args.intensity_lr *= 2
        with self.assertRaisesRegex(ValueError,'优化条件'):check_resume_config(saved,self.args)

    def test_dense_flow_uses_saved_pixel_centers_and_pixel_units(self):
        from Motion import export_dense_flow
        class Translation(torch.nn.Module):
            def __init__(self):
                super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()))
                self.uses_control_points=False;self.motion_dimension='2D'
                self.control_deformation_mode='kernel';self.control_indices=torch.zeros(1,dtype=torch.long)
                self.through_plane_max_displacement=.1;self.queries=[]
            def deform_query_points(self,points,target_time_t,**kwargs):
                self.queries.append(points.clone())
                return points+points.new_tensor([dx,dy,0.])*target_time_t
        geometry=make_geometry([1.2,.8,8.],[1,3,5],self.args)
        dx=2*geometry['bounds'][0]/5;dy=2*geometry['bounds'][1]/3
        net=Translation()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'flow.npz'
            export_dense_flow(net,2,1,3,5,1,100,str(path),geometry=geometry)
            with np.load(path,allow_pickle=False) as result:
                np.testing.assert_allclose(result['flow_pixel'][1],1.,rtol=1e-6)
                check_geometry(json.loads(result['geometry_json'].item()),geometry)
            y,x=torch.meshgrid(torch.arange(3),torch.arange(5),indexing='ij')
            bx,by,_=geometry['bounds']
            expected=torch.stack((((x.flatten().double()+.5)*2/5-1)*bx,
                                  ((y.flatten().double()+.5)*2/3-1)*by,
                                  torch.zeros(15,dtype=torch.float64)),1)
            torch.testing.assert_close(net.queries[0].double(),expected,atol=1e-7,rtol=1e-6)


class ReconstructionSnapshotTests(unittest.TestCase):
    def test_full_case_snapshot_is_single_png_with_expected_slices(self):
        """每次快照写一张"原图 | 渲染"拼图；插层期按 0.25 连续索引渲染 4(S-1)+1 张，否则只渲染 S 张整数层。"""
        import GetInitialGaussian as stage1
        from PIL import Image
        model = SimpleNamespace(get_scaling=torch.full((2, 3), .1),
                                _rotation=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1))
        indices = []
        def fake_render(_, img_size, slice_idx, **kwargs):
            indices.append(slice_idx)
            return {'render': torch.full(img_size, slice_idx/10.)}
        images = torch.stack([torch.full((12, 16), k/10.) for k in range(5)])
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(stage1, 'render', side_effect=fake_render):
            stage1.reconstruct_full_case(model, images, directory, 2000, False, .25, dpi=40)
            self.assertEqual(indices, [0, 1, 2, 3, 4])
            indices.clear()
            stage1.reconstruct_full_case(model, images, directory, 6000, True, .25, dpi=40)
            self.assertEqual(indices, [i*.25 for i in range(17)])
            files = sorted(p.name for p in (Path(directory)/'reconstructions').iterdir())
            # 只保留最终拼图，后一次调用覆盖前一次
            self.assertEqual(files, ['final.png'])
            Image.open(Path(directory)/'reconstructions'/'final.png').verify()


class InitializationTests(unittest.TestCase):
    def setUp(self):
        self.args = alpha_train_config()
        self.args.num_pts = 48
        self.args.init_fit_steps = 8
        self.args.interpolation = False
        patch = mock.patch('Gaussians.initialization.render',side_effect=reference_linear_render)
        patch.start();self.addCleanup(patch.stop)

    def initialize(self, device='cpu', case='synthetic'):
        y,x = torch.meshgrid(torch.linspace(-1,1,9,device=device),torch.linspace(-1,1,13,device=device),indexing='ij')
        images = torch.stack([.1+.5*torch.exp(-3*((x-.1*k)**2+y*y)) for k in range(3)])
        geometry = make_geometry([1.,1.5,8.],images.shape,self.args)
        model = GaussianModel();attach_geometry(model,geometry)
        report = model.initialize_from_images(images,geometry,self.args,geometry['rendering'],case)
        return model,images,report

    def test_distribution_covariance_reproducibility_and_rng_isolation(self):
        rng = torch.get_rng_state().clone()
        first,images,report = self.initialize()
        second,_,_ = self.initialize()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        torch.testing.assert_close(first.get_xyz,second.get_xyz,rtol=0,atol=0)
        self.assertEqual(first.get_xyz.shape,(48,3))
        # 观测框按每层代表厚度 sz 的薄片定义，z 可越过首末层中心但不超出 S*sz/2。
        self.assertTrue((first.get_xyz.abs() <= first._observation_box()+1e-6).all())
        # 经典 3DGS 初始尺度：各向同性，sigma=sqrt(最近 3 个中心平方距离均值)，下界为 min_sigma_pixels。
        from Gaussians.initialization import _mean_knn_sq_distance
        floor = self.args.min_sigma_pixels*float(first._voxel_spacing()[:2].min())
        expected = _mean_knn_sq_distance(first.get_xyz).sqrt().clamp_min(floor)
        torch.testing.assert_close(first.get_scaling, expected[:,None].repeat(1,3))
        torch.testing.assert_close(first.get_rotation,torch.tensor([[1.,0.,0.,0.]]).repeat(48,1))
        self.assertLess(report['mse_after'],report['mse_before'])
        self.assertTrue(torch.isfinite(first.get_intensity).all())
        self.assertEqual(first.geometry_reference_count,48)

    def test_remaining_invisible_points_are_dropped_after_retries(self):
        """重试后仍无像素贡献的少量点被丢弃（不报错），点数与参考容量同步减少。"""
        original = IntensityOperator.forward
        def forward(operator, coefficients, visibility=False):
            result = original(operator, coefficients, visibility)
            if visibility:
                result[1][0] = False
            return result
        with mock.patch.object(IntensityOperator, 'forward', forward):
            model,_,report = self.initialize()
        self.assertEqual((report['dropped_invisible'], report['count']), (1, 47))
        self.assertEqual((len(model.get_xyz), model.geometry_reference_count), (47, 47))
        self.assertTrue(all(len(getattr(model, name)) == 47 for name in ('_scaling','_rotation','_intensity')))

    def test_knn_scale_matches_brute_force_definition(self):
        """分块 kNN 与 distCUDA2 的定义一致：排除自身后最近 3 点平方距离均值；分块边界不影响结果。"""
        from Gaussians.initialization import _mean_knn_sq_distance
        points = torch.rand(257, 3, generator=torch.Generator().manual_seed(3))
        distance = torch.cdist(points, points).square()
        distance.fill_diagonal_(float('inf'))
        expected = distance.topk(3, dim=1, largest=False).values.mean(1)
        torch.testing.assert_close(_mean_knn_sq_distance(points, chunk=50), expected)

    def test_linear_operator_adjoint(self):
        model,images,_ = self.initialize()
        operator = IntensityOperator(model,images,model.geometry['rendering'])
        vector = torch.linspace(-.3,.5,48,dtype=torch.float64)
        pixels = torch.linspace(-.2,.4,images.numel(),dtype=torch.float64).reshape_as(images)
        torch.testing.assert_close((operator.forward(vector)*pixels).mean(),
                                   vector@operator.adjoint(pixels),rtol=3e-6,atol=1e-8)

    def test_constant_zero_images_and_tiny_point_count(self):
        self.args.num_pts = 1
        images = torch.zeros(1,9,13)
        geometry = make_geometry([1.,1.5,8.],images.shape,self.args)
        model = GaussianModel();attach_geometry(model,geometry)
        model.initialize_from_images(images,geometry,self.args,geometry['rendering'],'constant')
        torch.testing.assert_close(model.get_intensity,torch.zeros(1,1))
        self.assertTrue(torch.isfinite(model.get_scaling).all())

    def test_ply_geometry_roundtrip_hash_and_legacy_absence(self):
        model,_,_ = self.initialize()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.ply'
            save_model_ply(model,path)
            record = read_model_geometry(path)
            check_geometry(record['geometry'],model.geometry)
            self.assertEqual(record['geometry_reference_count'],48)
            other = GaussianModel();other.load_ply(str(path),device='cpu')
            torch.testing.assert_close(other.get_xyz,model.get_xyz)
            with path.open('ab') as stream:stream.write(b'changed')
            with self.assertRaisesRegex(ValueError,'不匹配'):read_model_geometry(path)
            self.assertIsNone(read_model_geometry(Path(directory)/'legacy.ply'))

    def test_fixed_reference_count_keeps_geometry_penalty(self):
        scales = torch.tensor([[.3, .3, 3.6], [.3, .3, .3]])
        before = anisotropic_scale_loss(scales, reference_count=2)
        after = anisotropic_scale_loss(torch.cat((scales, torch.ones(1, 3))), reference_count=2)
        torch.testing.assert_close(before, after)


@unittest.skipUnless(torch.cuda.is_available() and NATIVE_AVAILABLE,'需要 CUDA 与新渲染扩展')
class CudaInitializationTests(unittest.TestCase):
    initialize = InitializationTests.initialize
    def setUp(self):
        self.args = alpha_train_config()
        self.args.num_pts = 24
        self.args.init_fit_steps = 8

    def test_cuda_init_adjoint_then_backward(self):
        self.args.num_pts = 48
        state = torch.cuda.get_rng_state().clone()
        model,images,report = self.initialize('cuda')
        self.assertTrue(torch.equal(state,torch.cuda.get_rng_state()))
        self.assertLess(report['mse_after'],report['mse_before'])
        operator = IntensityOperator(model,images,model.geometry['rendering'])
        vector = torch.linspace(-.3,.5,48,device='cuda',dtype=torch.float64)
        pixels = torch.linspace(-.2,.4,images.numel(),device='cuda',dtype=torch.float64).reshape_as(images)
        torch.testing.assert_close((operator.forward(vector)*pixels).mean(),vector@operator.adjoint(pixels),rtol=5e-5,atol=1e-7)
        model.training_setup(self.args)
        from utils.Renderer import render
        prediction = render(model,[9,13],1,2,3,**model.geometry['rendering'])['render']
        (prediction-images[1]).square().mean().backward()
        for group in model.optimizer.param_groups:
            self.assertTrue(torch.isfinite(group['params'][0].grad).all())
        model.optimizer.step()

    def test_main_training_resume_and_repair(self):
        from GetInitialGaussian import training
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);data_root=root/'data';data_root.mkdir()
            y,x = np.mgrid[-1:1:12j,-1:1:16j]
            image = np.exp(-3*(x*x+y*y)).astype(np.float32)
            cmr = np.stack([np.stack([image,image*.8])*(1+.1*t) for t in range(3)])
            info = dict(schema={'array_axes':{'cmr':['T','Z','Y','X'],'labels':['phase','Z','Y','X']}},
                case={'case_id':'synthetic','original_split':'Testing'},
                annotation={'time_indices':[0,2],'phase_names':['ED','ES'],'has_labels':True,'phase_metadata_valid':True},
                source={'files':{'cmr_4d':{'zooms':[1.5,1.2,8.,1.]}}},
                processing={'output':{'cmr_shape':list(cmr.shape),'labels_shape':[2,2,12,16]}})
            np.savez_compressed(data_root/'synthetic.npz',cmr=cmr,labels=np.zeros((2,2,12,16),dtype=np.uint8),info=json.dumps(info))
            args=self.args
            args.dataset_paths['acdc']=str(data_root);args.dataset_mode='all'
            args.output_path=str(root/'training');args.iters=2;args.init_fit_steps=3
            args.interpolation=True;args.interpolate_iter=1;args.interpolation_probability=1.
            args.checkpoint_interval=2;args.save_iterations=[1];args.reconstruct_interval=0
            args.densify_from_iter=1;args.densify_until_iter=3;args.densification_interval=1
            args.structural_min_observations=1;args.structural_grad_multiplier=0.
            args.structure_log_interval=1;args.structural_prune_after_densify=False
            with mock.patch('GetInitialGaussian.update_loss_plot'):
                training(args)
                args.iters=3
                training(args)
                ply=Path(args.output_path)/'synthetic'/'point_cloud_final.ply'
                Path(str(ply)+'.geometry.json').unlink()
                # 模拟最终 PLY 记录缺失：完成后 checkpoint 已删除，按未完成病例重新训练。
                training(args)
            record=read_model_geometry(ply)
            self.assertEqual(record['iteration'],3)
            self.assertEqual(record['geometry_reference_count'],24)
            # 结构日志：续训截断后不重复；增殖窗口 [1,3) 内各有一次结构周期，第 3 步之后点数定型。
            import csv
            with open(ply.parent/'structure_log.csv',newline='') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual([int(row['iteration']) for row in rows],[1,2,3])
            self.assertEqual([int(row['structure_steps']) for row in rows],[1,1,0])
            self.assertEqual(rows[-1]['grad_threshold'],'')
            self.assertEqual(int(rows[-1]['points']),int(rows[-2]['points']))
            # 正常结束后只保留 final 权重，续训 checkpoint 已删除
            self.assertFalse((ply.parent/'training_state.pth').exists())
            self.assertEqual(record['training']['observed_slices'],[0,1])
            # 同一 PLY 经过真实评估入口，核对数据、几何记录与新版渲染接口。
            from evaluate import evaluate
            evaluation=load_config(ROOT/'config_test.yaml')
            evaluation.render_model='innergs_alpha_fixed_v1'
            evaluation.dataset_paths['acdc']=str(data_root)
            evaluation.dataset_mode='all'
            evaluation.stage1_ply_path=str(Path(args.output_path)/'{case}'/'point_cloud_final.ply')
            evaluation.output_path=str(root/'evaluation')
            evaluation.save_images=False
            rows=evaluate(evaluation)
            self.assertEqual(len(rows),2)
            self.assertTrue((root/'evaluation'/'metrics.csv').is_file())
            self.assertTrue((root/'evaluation'/'synthetic'/'reconstruction.npz').is_file())



if __name__=='__main__':unittest.main()
