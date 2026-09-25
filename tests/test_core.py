"""CPU-only correctness tests. No video weights, datasets or optimization loops."""
import json
import tempfile
import unittest
from pathlib import Path
import torch
from configs.config import TopoConfig
from topo_dit.models import TopoDiT_Model
from topo_dit.models.router.sinkhorn_ot import sinkhorn_knopp, marginal_errors
from topo_dit.models.perception.rope3d import ApplyPosEmb3D
from topo_dit.training.flow_matching import FlowMatcher
from topo_dit.training.trainer import TopoDiTTrainer
from topo_dit.checkpoint import save_checkpoint, load_checkpoint
from topo_dit.sampling import euler_flow_matching_sampler
from topo_dit.data.video_dataset import TopoVideoDataset


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.cfg=TopoConfig('configs/topodit_cpu.yaml')
        self.model=TopoDiT_Model(**self.cfg.config_dict['model'])
        self.z=torch.randn(2,4,2,3,3)
        self.c=torch.zeros(2,8)

    def test_uniform_rectangular_marginals(self):
        for n,k in [(2,2),(7,3),(2,5)]:
            a=sinkhorn_knopp(torch.zeros(1,n,k))
            torch.testing.assert_close(a,torch.full_like(a,1/k))
            self.assertLess(float(marginal_errors(a)['column_relative_error']),1e-5)

    def test_bounded_random_transport(self):
        a=sinkhorn_knopp(torch.rand(2,31,5)*2-1,iters=60)
        self.assertTrue(torch.isfinite(a).all())
        self.assertLess(float(marginal_errors(a)['column_relative_error']),1e-4)
        torch.testing.assert_close(a.sum(-1),torch.ones(2,31))

    def test_extreme_cost_finite(self):
        a=sinkhorn_knopp(torch.tensor([[[10000.,-10000.],[-10000.,10000.]]]))
        self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(a.sum(-1),torch.ones(1,2))

    def test_sinkhorn_gradient(self):
        cost=torch.randn(1,4,3,dtype=torch.float64,requires_grad=True)*0.1
        self.assertTrue(torch.autograd.gradcheck(lambda c:sinkhorn_knopp(c,iters=10), (cost,)))

    def test_invalid_transport(self):
        for kwargs in [{'epsilon':0},{'iters':0}]:
            with self.assertRaises(ValueError): sinkhorn_knopp(torch.zeros(1,2,2),**kwargs)
        with self.assertRaises(ValueError): sinkhorn_knopp(torch.full((1,2,2),float('nan')))

    def test_position_1024(self):
        x=ApplyPosEmb3D(1024)(torch.zeros(1,8,1024),2,2,2)
        self.assertEqual(x.shape,(1,8,1024))
        self.assertFalse(torch.equal(x[:,0],x[:,1]))
        self.assertFalse(torch.equal(x[:,0],x[:,2]))
        self.assertFalse(torch.equal(x[:,0],x[:,4]))

    def test_latent_contract(self):
        v,a,e=self.model(self.z,torch.tensor([0.2,0.8]),self.c)
        self.assertEqual(v.shape,self.z.shape)
        self.assertEqual(a.shape,(2,18,4))
        self.assertEqual(e.shape,(2,4,24))
        self.assertTrue(torch.isfinite(v).all())

    def test_time_changes_velocity(self):
        v0=self.model(self.z,torch.zeros(2),self.c)[0]
        v1=self.model(self.z,torch.ones(2),self.c)[0]
        self.assertGreater(float((v0-v1).abs().max().detach()),1e-5)

    def test_condition_changes_velocity(self):
        v0=self.model(self.z,torch.ones(2)*0.5,self.c)[0]
        v1=self.model(self.z,torch.ones(2)*0.5,torch.ones_like(self.c))[0]
        self.assertGreater(float((v0-v1).abs().max().detach()),1e-5)

    def test_no_identity_reconstruction(self):
        out,_,_=self.model.reconstruct(self.z)
        self.assertGreater(float((out-self.z).square().mean().detach()),0.01)
        (out-self.z).square().mean().backward()
        grad=self.model.router.proj_k.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()),0)

    def test_fm_gradients(self):
        trainer=TopoDiTTrainer(self.model,None,self.cfg.training)
        loss,_=trainer.compute_loss({'latent':self.z,'condition':self.c},2)
        loss.backward()
        for param in [self.model.latent_proj.weight,self.model.router.proj_q.weight,
                      self.model.dynamics.time_mlp[0].weight,self.model.renderer.proj_out.weight]:
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(float(param.grad.abs().sum()),0)

    def test_masked_loss_gradients(self):
        trainer=TopoDiTTrainer(self.model,None,self.cfg.training)
        loss,metrics=trainer.compute_loss({'latent':self.z,'condition':self.c},1)
        loss.backward()
        self.assertGreater(float(loss.detach()),0)
        self.assertGreater(float(self.model.router.proj_k.weight.grad.abs().sum()),0)
        self.assertIn('column_relative_error',metrics)

    def test_ada_zero_blocks(self):
        block=self.model.dynamics.blocks[0]
        x=torch.randn(2,4,24)
        torch.testing.assert_close(block(x,self.c),x)

    def test_flow_endpoints(self):
        noise=torch.randn_like(self.z)
        fm=FlowMatcher()
        for t,target in [(0,noise),(1,self.z)]:
            z,v,_=fm.construct_flow_target(self.z,noise,torch.full((2,),float(t)))
            torch.testing.assert_close(z,target)
            torch.testing.assert_close(v,self.z-noise)

    def test_sampler_repeatability_and_mode(self):
        self.model.train()
        a=euler_flow_matching_sampler(self.model,self.c,2,3,3,steps=3,seed=4)
        b=euler_flow_matching_sampler(self.model,self.c,2,3,3,steps=3,seed=4)
        torch.testing.assert_close(a,b)
        self.assertTrue(self.model.training)
        self.assertTrue(torch.isfinite(a).all())

    def test_sampler_integrates_velocity(self):
        class Constant(torch.nn.Module):
            latent_channels,cond_dim=4,8
            def forward(self,z,t,c,**kwargs): return torch.ones_like(z),None,None
        gen=torch.Generator().manual_seed(3)
        expected=torch.randn(2,4,2,3,3,generator=gen)+1
        actual=euler_flow_matching_sampler(Constant(),self.c,2,3,3,steps=4,seed=3)
        torch.testing.assert_close(actual,expected)

    def test_edit_feature_intervention(self):
        t=torch.ones(2)*0.5
        v=self.model(self.z,t,self.c)[0]
        edited=self.model(self.z,t,self.c,edit_index=0)[0]
        self.assertGreater(float((v-edited).abs().max().detach()),1e-5)
        with self.assertRaises(ValueError): self.model(self.z,t,self.c,edit_index=10)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.pt'
            save_checkpoint(path,self.model,self.cfg,0,2)
            restored,cfg,payload=load_checkpoint(path)
            self.assertEqual(cfg.model.num_heads,4)
            t=torch.ones(2)*0.5
            torch.testing.assert_close(self.model(self.z,t,self.c)[0],restored(self.z,t,self.c)[0])
            torch.save({'model_state_dict':{}},path)
            with self.assertRaises(ValueError): load_checkpoint(path)

    def test_cached_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d)
            torch.save(self.z[0],base/'z.pt')
            torch.save(torch.ones(8),base/'c.pt')
            record={'latent_path':'z.pt','condition_path':'c.pt','codec_id':'test','latent_scale':1.,'condition_id':'text-v1'}
            (base/'data.jsonl').write_text(json.dumps(record),encoding='utf-8')
            ds=TopoVideoDataset(base/'data.jsonl',4,8,'cached',1.,'test','text-v1')
            torch.testing.assert_close(ds[0]['condition'],torch.ones(8))
            with self.assertRaises(ValueError): TopoVideoDataset(base/'data.jsonl',4,8,'cached',1.,'test','other')
            (base/'z.pt').unlink()
            with self.assertRaises(FileNotFoundError): TopoVideoDataset(base/'data.jsonl',4,8,'cached',1.,'test','text-v1')

    def test_unconditional_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d); torch.save(self.z[0],base/'z.pt')
            (base/'data.jsonl').write_text(json.dumps({'latent_path':'z.pt','codec_id':'test','latent_scale':1.}),encoding='utf-8')
            ds=TopoVideoDataset(base/'data.jsonl',4,8,codec_id='test')
            torch.testing.assert_close(ds[0]['condition'],torch.zeros(8))

    def test_config_and_shape_errors(self):
        with self.assertRaises(ValueError): TopoConfig(config_dict={'model':{'dit_depth':2}})
        with self.assertRaises(AttributeError): _=self.cfg.model.missing
        with self.assertRaises(ValueError): self.model(self.z,torch.ones(2)*2,self.c)
        with self.assertRaises(ValueError): self.model(self.z,torch.ones(2),torch.zeros(2,9))

    def test_baseline_routes(self):
        for routing in ['softmax','uniform']:
            model=TopoDiT_Model(**{**self.cfg.config_dict['model'],'routing':routing})
            v,a,_=model(self.z,torch.ones(2)*0.5,self.c)
            self.assertTrue(torch.isfinite(v).all())
            torch.testing.assert_close(a.sum(-1),torch.ones(2,18))


    def test_codec_tensor_and_tuple_contract(self):
        from unittest.mock import patch
        from topo_dit.models.perception.causal_3d_vae import CosmosCausal3DVAE
        class Encoder(torch.nn.Module):
            def __init__(self, as_tuple):
                super().__init__(); self.as_tuple=as_tuple
            def forward(self,x):
                z=torch.nn.functional.avg_pool3d(x.mean(1,keepdim=True),kernel_size=(1,2,2),stride=(1,2,2))[:,:,::2]
                z=z.repeat(1,16,1,1,1)
                return (z,torch.zeros(1)) if self.as_tuple else z
        class Decoder(torch.nn.Module):
            def forward(self,z):
                t,h,w=z.shape[-3:]
                return torch.nn.functional.interpolate(z[:,:3],size=(1+(t-1)*2,h*2,w*2))
        with tempfile.TemporaryDirectory() as d:
            for name in ['encoder.jit','decoder.jit']: (Path(d)/name).touch()
            for as_tuple in [False,True]:
                with patch('torch.jit.load',side_effect=[Encoder(as_tuple),Decoder()]):
                    codec=CosmosCausal3DVAE(d,temporal_compression=2,spatial_compression=2,scale_factor=2)
                codec.train()
                self.assertFalse(codec.training)
                x=torch.ones(2,3,4,5,5)*0.25
                latent=codec.encode(x)
                self.assertEqual(latent.shape,(2,16,3,3,3))
                torch.testing.assert_close(latent,torch.ones_like(latent)*0.5)
                decoded=codec.decode(latent,crop_shape=(4,5,5))
                torch.testing.assert_close(decoded,x)

    def test_inference_cli_artifact(self):
        from unittest.mock import patch
        from topo_dit.inference_cli import main
        with tempfile.TemporaryDirectory() as d:
            ckpt=Path(d)/'untrained_test.pt'; output=Path(d)/'sample.pt'
            # Synthetic checkpoint solely for API validation, not generation evaluation.
            save_checkpoint(ckpt,self.model,self.cfg,0,2)
            args=['inference_edit.py','--ckpt',str(ckpt),'--latent-shape','2','3','3',
                  '--steps','2','--output',str(output)]
            with patch('sys.argv',args): main()
            result=torch.load(output,weights_only=True)
            self.assertEqual(result['latent'].shape,(1,4,2,3,3))
            self.assertEqual(result['codec_id'],'synthetic-test-only')
            save_checkpoint(ckpt,self.model,self.cfg,0,1)
            with patch('sys.argv',args),self.assertRaises(SystemExit): main()

    def test_entrypoint_help(self):
        import subprocess,sys
        for entry in ['train_phase1_autoenc.py','scripts/train_phase2_flow.py','scripts/inference_edit.py','scripts/prepare_latents.py']:
            result=subprocess.run([sys.executable,entry,'--help'],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('usage:',result.stdout)

    def test_missing_data_fails_without_retry(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'empty.jsonl'; p.write_text('',encoding='utf-8')
            with self.assertRaises(ValueError): TopoVideoDataset(p,4,8)


if __name__=='__main__':
    unittest.main()
