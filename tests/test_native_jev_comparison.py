"""Synthetic CPU software checks, never benchmark accuracy evidence."""
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from jev.native_visual_baselines import conditional_scores,cold_conditional_scores,answer_tokens,text_windows
from scripts.evaluation.native_jev_comparison import evaluate_native,compare_case,build_tables,recover_prefix
from scripts.evaluation.jev_pair_protocol import sha256,write_json
from scripts.training.train_adapter_study_matrix import train
from scripts.evaluation.evaluate_adapter_study_matrix import evaluate
from tests import test_adapter_study as fixtures


class Cache:
    def __init__(self,total): self.total=total
    def batch_repeat_interleave(self,n): self.total=self.total.repeat_interleave(n,dim=0)


class CausalToy:
    def __call__(self,input_ids,attention_mask=None,past_key_values=None,pixel_values=None,logits_to_keep=0,use_cache=False,**kwargs):
        prefix=0 if past_key_values is None else past_key_values.total[:,None]
        image=0 if pixel_values is None else pixel_values.sum()
        states=input_ids.cumsum(-1)+prefix+image
        vocabulary=torch.arange(32,dtype=torch.float32)[None,None]
        logits=-(vocabulary-(states[:,:,None]%32)).square()/9
        if logits_to_keep: logits=logits[:,-logits_to_keep:]
        return SimpleNamespace(logits=logits,past_key_values=Cache(states[:,-1]) if use_cache else None)


class Tokenizer:
    eos_token_id=1
    def encode(self,text,add_special_tokens=False): return [ord(c)%20+2 for c in text]
    def __call__(self,text,add_special_tokens=True,**kwargs):
        ids=self.encode(text)
        return {'input_ids':ids+[1] if add_special_tokens else ids}


class ToyNative:
    windowed_texts=0; last_check=None
    protocol={'name':'SYNTHETIC CPU fixture, not a native benchmark'}
    def scores(self,image,question,candidates): return [1.,0.]


class ScoringTests(unittest.TestCase):
    def test_cached_matches_cold_all_variable_length_and_empty_candidates(self):
        model=CausalToy()
        encoded=dict(input_ids=torch.tensor([[3,4,5]]),attention_mask=torch.ones(1,3,dtype=torch.long),pixel_values=torch.tensor([2.]))
        tokens=answer_tokens(Tokenizer(),['','a','red','many words'])
        before=cold_conditional_scores(model,encoded,tokens)
        for batch in (1,2,8):
            after=conditional_scores(model,encoded,tokens,0,batch)
            self.assertTrue(torch.allclose(torch.tensor(before),torch.tensor(after),atol=1e-5))
        self.assertEqual(len(after),4)

    def test_siglip_windows_drop_no_body_tokens(self):
        tokenizer=Tokenizer(); text='The whole long question and candidate answer '*20
        windows,weights=text_windows(tokenizer,text,64)
        self.assertGreater(len(windows),1)
        self.assertTrue(all(len(w)<=64 and w[-1]==1 for w in windows))
        self.assertEqual([i for w in windows for i in w[:-1]],tokenizer.encode(text))
        self.assertEqual(sum(weights),len(tokenizer.encode(text)))


class FullComparisonTests(fixtures.AdapterStudyTests):
    def native_config(self):
        path=self.root/'eval/toy'; path.mkdir(parents=True,exist_ok=True)
        candidates=path/'candidates.jsonl'; candidates.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        m={**self.manifest,'status':'ready','complete_split':True,'public_labels':True,'candidate_manifest':str(candidates),
           'candidate_manifest_sha256':sha256(candidates),'dataset_name':'toy'}
        write_json(path/'manifest.json',m)
        return {**self.config,'evaluation_root':'eval'},m

    def test_full_native_resume_and_paired_jev_three_arms(self):
        config,m=self.native_config()
        native_root=self.root/'native'; result=evaluate_native(self.root,config,'toy','toy',{'fixture':'sha'},native_root,'cpu',baseline=ToyNative())
        self.assertEqual(result['N'],4); self.assertEqual(result['native_metric']['score'],.5)
        refs=self.refs; training=train(self.root,config,m,self.rows,refs,self.pool,self.source,'toy',self.root/'training','cpu')
        adapter_root=self.root/'adapted'
        evaluate(self.root,config,'toy',m,self.rows,'toy',training,self.pool,adapter_root,'cpu')
        comparison=compare_case(self.root,config,'toy','toy',native_root,adapter_root,self.root/'comparisons')
        self.assertEqual(comparison['N'],4)
        for arms in comparison['seeds'].values():
            for value in arms.values(): self.assertEqual(value['improved']+value['worsened']+value['unchanged'],4)
        table=build_tables(self.root/'comparisons',self.root/'reports')
        self.assertEqual(table['complete_cases'],1)
        for suffix in ('csv','md','tex'): self.assertTrue((self.root/'reports'/f'native_vs_jev.{suffix}').is_file())
        # Recover exactly two committed rows, preserve the partial final bytes.
        directory=native_root/'toy/toy'; (directory/'result.json').unlink()
        pp=directory/'predictions.jsonl'; lines=pp.read_bytes().splitlines(keepends=True)
        pp.write_bytes(b''.join(lines[:2])+b'{"uid":')
        again=evaluate_native(self.root,config,'toy','toy',{'fixture':'sha'},native_root,'cpu',baseline=ToyNative())
        self.assertEqual(again['N'],4)
        self.assertEqual([json.loads(l)['uid'] for l in pp.read_text().splitlines()],[r['uid'] for r in self.rows])
        self.assertEqual(len(list(directory.glob('uncommitted-native-tail-*.bin'))),1)

    def test_extra_native_prediction_rejected(self):
        config,m=self.native_config(); root=self.root/'native'
        evaluate_native(self.root,config,'toy','toy',{'fixture':'sha'},root,'cpu',baseline=ToyNative())
        directory=root/'toy/toy'; (directory/'result.json').unlink()
        with (directory/'predictions.jsonl').open('a') as h: h.write(json.dumps({'uid':'SYNTHETIC EXTRA'})+'\n')
        with self.assertRaisesRegex(ValueError,'ordered UID'):
            evaluate_native(self.root,config,'toy','toy',{'fixture':'sha'},root,'cpu',baseline=ToyNative())

    def test_native_result_checksum_change_rejected(self):
        config,m=self.native_config(); root=self.root/'native'
        evaluate_native(self.root,config,'toy','toy',{'fixture':'sha'},root,'cpu',baseline=ToyNative())
        with (root/'toy/toy/predictions.jsonl').open('a') as h: h.write('bad tail')
        with self.assertRaisesRegex(ValueError,'changed'):
            evaluate_native(self.root,config,'toy','toy',{'fixture':'sha'},root,'cpu',baseline=ToyNative())


if __name__=='__main__': unittest.main()
