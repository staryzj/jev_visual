"""Synthetic CPU acceleration checks, not benchmark performance evidence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from scripts.workflows.adapter_study_acceleration import (
    LockedFeaturePool, FeaturePrefetch, compare_features, vision_batch, accelerated_training_cache,
)
from scripts.workflows.adapter_study_cache import FeaturePool, load_feature
from scripts.workflows.run_adapter_study_parallel import jobs_for_stage, training_order
from scripts.evaluation.run_jev_checkpoint_comparison import state_hash
from jev.adapter_study import load, ARMS
from tests import test_adapter_study as fixtures
from tests.test_adapter_study import FakeEncoder


class FakeBatchEncoder(FakeEncoder):
    def __init__(self, *args): self.closed = False
    def close(self): self.closed = True


class AccelerationTests(fixtures.AdapterStudyTests):
    def test_prefetch_preserves_training_tensors_and_loss_history(self):
        import scripts.training.train_adapter_study_matrix as trainer
        config = {**self.config, 'seeds': [7]}
        before = trainer.train(self.root,config,self.manifest,self.rows,self.refs,self.pool,self.source,
                               'toy',self.root/'baseline','cpu')[0]
        order = training_order(config,self.root/'prefetched','toy',7,len(self.rows))
        prefetch = FeaturePrefetch(self.pool,self.refs,order,depth=3)
        try:
            with patch.object(trainer,'load_feature',prefetch):
                after = trainer.train(self.root,config,self.manifest,self.rows,self.refs,self.pool,self.source,
                                      'toy',self.root/'prefetched/training','cpu')[0]
        finally: prefetch.close()
        self.assertEqual(before['history'],after['history'])
        self.assertEqual(before['updates_per_arm'],after['updates_per_arm'])
        for arm in ARMS:
            self.assertEqual(state_hash(load(before['checkpoints'][arm]['path'])[0]),
                             state_hash(load(after['checkpoints'][arm]['path'])[0]))

    def test_locked_pool_preserves_old_cache_bytes(self):
        before = {p: p.read_bytes() for p in self.pool.root.rglob('*.pt')}
        pool = LockedFeaturePool(self.pool.root,self.pool.identity)
        for row, ref in zip(self.rows,self.refs):
            value, reference = pool.get(row,FakeEncoder(),'toy')
            self.assertEqual(reference,ref)
        for path, data in before.items(): self.assertEqual(path.read_bytes(),data)

    def test_shared_feature_computed_once_across_concurrent_writers(self):
        from concurrent.futures import ThreadPoolExecutor
        created=[]
        def task(_):
            pool=LockedFeaturePool(self.root/'concurrent',{'fixture':True})
            def compute():
                created.append(True)
                return torch.ones(3,4)
            return pool.stored('text','shared-key',compute)[0]
        with ThreadPoolExecutor(max_workers=8) as executor:
            values=list(executor.map(task,range(8)))
        self.assertEqual(len(created),1)
        self.assertTrue(all(torch.equal(value,values[0]) for value in values))

    def test_canvas_header_matches_multi_image_and_intrinsic_blank(self):
        pool = LockedFeaturePool(self.root/'locked',{'fixture':True})
        row = {**self.rows[0], 'images':[self.rows[0]['images'][0],self.rows[1]['images'][0]]}
        a = pool.get(row,FakeEncoder(),'toy')
        b = FeaturePool(self.root/'serial',{'fixture':True}).get(row,FakeEncoder(),'toy')
        self.assertEqual(a[1],b[1])
        intrinsic = {**row,'images':[], 'metadata':{'intrinsic_no_image':True}}
        pool.get(intrinsic,FakeEncoder(),'toy')
        self.assertEqual(pool.canvas_sizes[()],(224,224))

    def test_complete_cache_resume_never_encodes_again(self):
        index = self.pool.root/'index_toy.json'
        with patch('scripts.workflows.adapter_study_cache.FrozenEncoders',side_effect=AssertionError('No encoder on cache hits')):
            refs = accelerated_training_cache(self.root,self.config,self.rows,'toy',self.pool,index,'cpu',2,2)
            again = accelerated_training_cache(self.root,self.config,self.rows,'toy',self.pool,index,'cpu',2,2)
        self.assertEqual(refs,again)
        self.assertEqual([r['uid'] for r in refs],[r['uid'] for r in self.rows])

    def test_cache_fresh_batches_and_full_order(self):
        pool = LockedFeaturePool(self.root/'batch',{'fixture':True})
        with patch('scripts.workflows.adapter_study_cache.FrozenEncoders',FakeBatchEncoder), \
             patch('scripts.workflows.adapter_study_acceleration.vision_batch',side_effect=lambda e, ims:[e.encode_image(i) for i in ims]):
            refs = accelerated_training_cache(self.root,self.config,self.rows,'toy',pool,pool.root/'index_toy.json','cpu',3,2)
        self.assertEqual([r['uid'] for r in refs],[r['uid'] for r in self.rows])
        for ref in refs: self.assertTrue(torch.equal(load_feature(pool,ref)['text'],FakeEncoder().encode_text({})))
        self.assertEqual(json.loads((pool.root/'batch_audit_toy.jsonl').read_text().splitlines()[0])['status'],'checked')

    def test_batch_failure_uses_all_original_rows(self):
        pool = LockedFeaturePool(self.root/'fallback',{'fixture':True})
        with patch('scripts.workflows.adapter_study_cache.FrozenEncoders',FakeBatchEncoder), \
             patch('scripts.workflows.adapter_study_acceleration.vision_batch',side_effect=ValueError('SYNTHETIC unsupported batch')):
            refs = accelerated_training_cache(self.root,self.config,self.rows,'toy',pool,pool.root/'index_toy.json','cpu',3,2)
        self.assertEqual(len(refs),4)
        self.assertEqual(json.loads((pool.root/'batch_audit_toy.jsonl').read_text().splitlines()[0])['status'],'serial_fallback')


class BatchUnitTests(unittest.TestCase):
    def test_job_partition_complete_and_unique(self):
        config = dict(backbones=['siglip2','internvl','llava'], seeds=[1,2,3])
        datasets = [str(i) for i in range(12)]
        for stage, expected in [('cache',3),('train',9),('evaluate',36)]:
            jobs = jobs_for_stage(stage,config,datasets)
            self.assertEqual(len(jobs),expected); self.assertEqual(len(set(jobs)),expected)

    def test_numerical_guard_rejects_bad_features(self):
        a = torch.ones(3,4)
        self.assertTrue(compare_features(a,a)['bitwise_equal'])
        for b in (a*1.1, torch.full_like(a,float('nan')), torch.ones(2,4)):
            with self.assertRaises(ValueError): compare_features(a,b)

    def test_siglip_batch_splits_per_image(self):
        def processor(images,return_tensors):
            rgb = torch.tensor(images.getpixel((0,0)),dtype=torch.float32)
            return {'pixel_values':rgb.reshape(1,3,1,1)}
        def forward(pixel_values,return_dict):
            return SimpleNamespace(last_hidden_state=pixel_values.flatten(1)[:,None,:].repeat(1,3,1))
        visual = SimpleNamespace(name='siglip2',processor=processor,device=torch.device('cpu'),dtype=torch.float32,
                                 model=SimpleNamespace(vision_model=forward))
        encoder = SimpleNamespace(visual=visual,config={'token_count':3})
        images = [Image.new('RGB',(8,8),(5,7,9)),Image.new('RGB',(8,8),(11,13,15))]
        result = vision_batch(encoder,images)
        self.assertEqual(len(result),2)
        self.assertEqual(result[0].shape,(3,3))
        self.assertTrue(torch.equal(result[0][0].float(),torch.tensor([5.,7.,9.])))
        self.assertTrue(torch.equal(result[1][0].float(),torch.tensor([11.,13.,15.])))


if __name__=='__main__': unittest.main()
