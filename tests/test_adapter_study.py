"""Synthetic CPU software fixtures ONLY; never paper/benchmark evidence."""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import torch
from PIL import Image
from jev.adapter_study import ARMS,fresh_arms,load
from visualjev.models.visual_jev_v3 import VisualJEVV3Config
from visualjev.models.visual_jev_v3_pipeline import AlignmentAdapterConfig,ThreeStageVisualJEV
from scripts.evaluation.run_jev_checkpoint_comparison import state_hash
from scripts.evaluation.jev_pair_protocol import sha256
from scripts.workflows.adapter_study_cache import FeaturePool
from scripts.training.train_adapter_study_matrix import train,score
from scripts.evaluation.evaluate_adapter_study_matrix import evaluate,add_row,new_totals
from scripts.evaluation.build_adapter_study_tables import build
from scripts.data.prepare_adapter_study_training import icon_membership


class FakeEncoder:
    def encode_text(self,row):
        return torch.tensor([[1.,0.,0.,0.,0.,0.],[0.,1.,0.,0.,0.,0.]])
    def encode_image(self,image):
        rgb=torch.tensor(image.getpixel((0,0)),dtype=torch.float32)/255
        return torch.cat([rgb,torch.ones(1)]).repeat(3,1)


class AdapterStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='synthetic_cpu_adapter_fixture_'); self.root=Path(self.temp.name)
        self.threads=torch.get_num_threads(); torch.set_num_threads(1)
        self.source=ThreeStageVisualJEV(AlignmentAdapterConfig(pre_merger_dim=4,teacher_dim=8,hidden_dim=5),
                    VisualJEVV3Config(vision_dim=8,text_dim=6,adapter_dim=4,num_heads=2,dropout=0.))
        self.source.eval().requires_grad_(False)
        self.config=dict(seeds=[7,8],epochs=2,gradient_accumulation=2,learning_rate=.001,invalid_weight=.1,
                         candidate_batch=2,head_batch=2,resume_every_updates=1,controls=True)
        self.rows=[]
        for i in range(4):
            path=self.root/f'image_{i}.png'; Image.new('RGB',(8,8),(i*30,10,5)).save(path)
            self.rows.append(dict(uid=f'SYNTHETIC:{i}',dataset='toy',images=[str(path)],question=f'toy question {i}',candidates=['x','y'],label=i%2,source_split='train',metadata={}))
        cp=self.root/'candidates.jsonl'; cp.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        self.manifest=dict(N=4,candidate_manifest_sha256=sha256(cp),source_membership_audit={'fixture_only':True},
                           split='test',is_true_test=True,revision='SYNTHETIC',fingerprint='SYNTHETIC')
        self.pool=FeaturePool(self.root/'features',{'fixture_only':True}); self.refs=[]
        for row in self.rows: self.refs.append(self.pool.get(row,FakeEncoder(),'toy')[1])

    def tearDown(self):
        torch.set_num_threads(self.threads); self.temp.cleanup()

    def test_arms_shared_scorer_and_mlp_initialization(self):
        models=fresh_arms(self.source,4,7)
        self.assertEqual(set(models),set(ARMS))
        self.assertEqual(state_hash(models['mlp_ce'].alignment),state_hash(models['mlp_invalid'].alignment))
        for model in models.values():
            self.assertEqual(state_hash(model.decision),state_hash(self.source.decision))
            self.assertFalse(any(p.requires_grad for p in model.decision.parameters()))

    def test_adapter_gradients_without_scorer_update(self):
        model=fresh_arms(self.source,4,7)['linear_ce']
        output=score(model,torch.randn(3,4),torch.randn(2,6),2)
        torch.nn.functional.cross_entropy(output[None],torch.tensor([0])).backward()
        self.assertTrue(any(p.grad is not None for p in model.alignment.parameters()))
        self.assertTrue(all(p.grad is None for p in model.decision.parameters()))

    def test_whole_train_checkpoint_roundtrip_and_completed_resume(self):
        output=self.root/'train'; results=train(self.root,self.config,self.manifest,self.rows,self.refs,self.pool,self.source,'toy',output,'cpu')
        self.assertEqual(len(results),2)
        for result in results:
            self.assertEqual(result['train_visits_per_arm'],8)
            self.assertEqual(result['updates_per_arm'],4)
            for arm,entry in result['checkpoints'].items():
                model,_=load(entry['path']); self.assertEqual(model.arm,arm)
                self.assertEqual(state_hash(model.decision),state_hash(self.source.decision))
        hashes=[r['checkpoints']['mlp_invalid']['sha256'] for r in results]
        again=train(self.root,self.config,self.manifest,self.rows,self.refs,self.pool,self.source,'toy',output,'cpu')
        self.assertEqual(hashes,[r['checkpoints']['mlp_invalid']['sha256'] for r in again])

    def test_partial_training_resume_identity_and_frozen_scorer(self):
        output=self.root/'train'; results=train(self.root,self.config,self.manifest,self.rows,self.refs,self.pool,self.source,'toy',output,'cpu')
        result_file=output/'toy/seed_7/result.json'; result_file.unlink()
        # This resume cursor is the final completed optimizer boundary: it must
        # regenerate checked artifacts without replaying/dropping train rows.
        again=train(self.root,self.config,self.manifest,self.rows,self.refs,self.pool,self.source,'toy',output,'cpu')
        self.assertEqual(again[0]['train_visits_per_arm'],8)
        self.assertEqual(state_hash(load(again[0]['checkpoints']['linear_ce']['path'])[0].decision),state_hash(self.source.decision))

    def test_mid_epoch_resume_matches_uninterrupted_tensors(self):
        baseline=train(self.root,self.config,self.manifest,self.rows,self.refs,self.pool,self.source,'toy',self.root/'baseline','cpu')
        import scripts.training.train_adapter_study_matrix as trainer
        real_save=torch.save; output=self.root/'interrupted'; fired=False
        def interrupt_after_saved_boundary(payload,path,*args,**kwargs):
            nonlocal fired
            real_save(payload,path,*args,**kwargs)
            if not fired and isinstance(payload,dict) and payload.get('updates')==1:
                # Mimic a crash immediately after an atomic optimizer checkpoint.
                Path(path).replace(Path(path).with_suffix('.pt')); fired=True
                raise RuntimeError('SYNTHETIC crash after first saved optimizer update')
        with patch.object(trainer.torch,'save',side_effect=interrupt_after_saved_boundary):
            with self.assertRaisesRegex(RuntimeError,'SYNTHETIC crash'):
                train(self.root,self.config,self.manifest,self.rows,self.refs,self.pool,self.source,'toy',output,'cpu')
        saved=torch.load(output/'toy/seed_7/training_resume.pt',map_location='cpu',weights_only=False)
        self.assertEqual((saved['epoch'],saved['offset'],saved['visits'],saved['updates']),(1,2,2,1))
        resumed=train(self.root,self.config,self.manifest,self.rows,self.refs,self.pool,self.source,'toy',output,'cpu')
        for before,after in zip(baseline,resumed):
            self.assertEqual(before['history'],after['history'])
            self.assertEqual(after['train_visits_per_arm'],8)
            for arm in ARMS:
                before_model,_=load(before['checkpoints'][arm]['path'])
                after_model,_=load(after['checkpoints'][arm]['path'])
                self.assertEqual(state_hash(before_model),state_hash(after_model))

    def test_full_evaluation_all_seeds_arms_and_tables(self):
        training=train(self.root,self.config,self.manifest,self.rows,self.refs,self.pool,self.source,'toy',self.root/'train','cpu')
        # Precomputed synthetic eval features avoid loading ANY pretrained model.
        result=evaluate(self.root,self.config,'toy',self.manifest,self.rows,'toy',training,self.pool,self.root/'evaluation','cpu')
        self.assertEqual(result['N'],4); self.assertEqual(set(result['metrics']['seeds']),{'7','8'})
        self.assertTrue(result['no_sampling'])
        summary=build(self.root/'evaluation',self.root/'reports',self.config,verify_predictions=True)
        self.assertEqual(summary['complete_backbone_dataset_cases'],1)
        for extension in ('csv','md','tex'):
            self.assertTrue((self.root/'reports'/('main_cross_backbone.'+extension)).is_file())

    def test_evaluation_resume_full_prefix_and_uncommitted_tail(self):
        training=train(self.root,self.config,self.manifest,self.rows,self.refs,self.pool,self.source,'toy',self.root/'train','cpu')
        directory=self.root/'evaluation/toy/toy'
        baseline=evaluate(self.root,self.config,'toy',self.manifest,self.rows,'toy',training,self.pool,self.root/'evaluation','cpu')
        predictions=directory/'predictions.jsonl'; lines=predictions.read_bytes().splitlines(keepends=True)
        (directory/'result.json').unlink()
        predictions.write_bytes(b''.join(lines[:2])+b'{"uid":"SYNTHETIC:2",')
        resumed=evaluate(self.root,self.config,'toy',self.manifest,self.rows,'toy',training,self.pool,self.root/'evaluation','cpu')
        records=[json.loads(line) for line in predictions.read_text().splitlines()]
        self.assertEqual([r['uid'] for r in records],[r['uid'] for r in self.rows])
        self.assertEqual(resumed['N'],4)
        for seed in ('7','8'):
            for arm in ARMS:
                self.assertEqual(baseline['metrics']['seeds'][seed][arm]['primary'],resumed['metrics']['seeds'][seed][arm]['primary'])
        self.assertEqual(len(list(directory.glob('uncommitted-tail-*.bin'))),1)

    def test_cache_checksum_change_rejected(self):
        path=self.pool.root/self.refs[0]['text']; path.write_bytes(b'corrupted software fixture')
        with self.assertRaisesRegex(ValueError,'checksum'):
            self.pool.get(self.rows[0],FakeEncoder(),'toy')

    def test_missing_seed_record_rejected(self):
        with self.assertRaisesRegex(ValueError,'seed coverage'):
            add_row(self.rows[0],{'runs':{}},new_totals(self.config),{'7':{}},False)

    def test_author_test_item_cannot_enter_train(self):
        problems={str(i):dict(split='train',ques_type=t) for i,t in enumerate(['choose_txt','choose_img','fill_in_blank'])}
        splits={t+'_'+s:([str(i)] if s=='train' else []) for i,t in enumerate(['choose_txt','choose_img','fill_in_blank']) for s in ['train','val','test']}
        counts={t:1 for t in ['choose_txt','choose_img','fill_in_blank']}
        self.assertEqual(len(icon_membership(problems,splits,counts)),3)
        problems['0']['split']='test'
        with self.assertRaisesRegex(ValueError,'held-out'):
            icon_membership(problems,splits,counts)


if __name__=='__main__': unittest.main()
