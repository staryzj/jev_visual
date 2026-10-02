"""Scientific protocol guards; fixture examples are never benchmark results."""
from types import SimpleNamespace
from pathlib import Path
import pytest
from scripts.submission_benchmarks import make_row,REGISTRY
from scripts.evaluate_submission_benchmarks import result_metrics,vqa_score,valid_complete_result,different_image_index

def test_test_labels_hidden_are_validation_named():
    for name in ['aokvqa','textvqa']:
        assert REGISTRY[name]['split']=='validation'
        assert REGISTRY[name]['true_test'] is False
    assert REGISTRY['gqa']['split']=='testdev'

def test_textvqa_candidates_do_not_use_gold(tmp_path):
    args=SimpleNamespace(fetch_images=False,flickr30k_root=None)
    a,_=make_row('textvqa',{'question':'read it','ocr_tokens':['red','sign'],'answers':['unseen gold']},0,tmp_path,args)
    b,_=make_row('textvqa',{'question':'read it','ocr_tokens':['red','sign'],'answers':['different gold']},0,tmp_path,args)
    assert a[0]['candidates']==b[0]['candidates']
    assert 'unseen gold' not in a[0]['candidates']
    assert 'red sign' in a[0]['candidates']

def test_iconqa_native_task_not_empty_or_guessed(tmp_path):
    args=SimpleNamespace(fetch_images=False,flickr30k_root=None)
    rows,_=make_row('iconqa',dict(ques_type='choose_txt',choices='odd,even',answer='even',question='parity'),0,tmp_path,args)
    assert rows[0]['candidates']==['odd','even'] and rows[0]['label']==1
    rows,reason=make_row('iconqa',dict(ques_type='choose_img'),0,tmp_path,args)
    assert not rows and reason=='outside_native_select_txt_task'

def test_vqa_leave_one_out_consensus():
    assert vqa_score('two',['2']*10)==1
    assert vqa_score('cat',['cat']+['dog']*9)==pytest.approx(.3)
    assert vqa_score('cat',['cat']*2+['dog']*8)==pytest.approx(.6)
    assert vqa_score('cat',['cat']*3+['dog']*7)==pytest.approx(.9)
    assert vqa_score('cat',['cat']*4+['dog']*6)==1

def test_winoground_strict_ties_fail():
    rows=[dict(dataset='winoground')]
    tied=result_metrics(rows,[dict(scores_2x2=[[1,1],[1,2]])])
    assert tied['group_score']==0 and tied['text_score']==0
    good=result_metrics(rows,[dict(scores_2x2=[[2,0],[0,2]])])
    assert good['group_score']==good['text_score']==good['image_score']==1

def test_sugarcrepe_pp_requires_both_positives():
    rows=[dict(dataset='sugarcrepe_pp',metadata={'category':'replace_obj'})]
    assert result_metrics(rows,[dict(scores=[3,1,2])])['strict_itt_accuracy']==0
    assert result_metrics(rows,[dict(scores=[3,3,2])])['strict_itt_accuracy']==1

def test_partial_predictions_rejected():
    with pytest.raises(ValueError,match='incomplete'):
        result_metrics([{'dataset':'ai2d'}],[])

def test_invalid_score_values_or_incomplete_candidates_rejected():
    row=dict(dataset='ai2d',label=0,candidates=['a','b'])
    with pytest.raises(ValueError,match='non-finite'):
        result_metrics([row],[dict(scores=[float('nan'),0],prediction=0)])
    with pytest.raises(ValueError,match='coverage mismatch'):
        result_metrics([row],[dict(scores=[2],prediction=0)])

def test_oov_is_in_full_denominator():
    rows=[dict(dataset='gqa',label=0),dict(dataset='gqa',label=None)]
    m=result_metrics(rows,[dict(scores=[2,0],prediction=0),dict(scores=[2,0],prediction=0)])
    assert m['count']==2 and m['accuracy']==.5 and m['oov_count']==1
    m=result_metrics([dict(dataset='gqa',label=None)],[dict(scores=[2,0],prediction=0)])
    assert m['count']==m['oov_count']==1 and m['accuracy']==0

def test_wrong_protocol_never_reused_as_full():
    m=dict(N=2,candidate_manifest_sha256='manifest')
    r=dict(status='complete',protocol='sample',checkpoint_sha256='cp',candidate_manifest_sha256='manifest',metrics={'count':2})
    assert not valid_complete_result(r,m,'cp')
    r['protocol']='full_benchmark'; assert valid_complete_result(r,m,'cp')
    r['metrics']['count']=1; assert not valid_complete_result(r,m,'cp')

def test_wrong_image_has_different_content_not_filename():
    assert different_image_index(0,[('same',),('same',),('different',)])==2
    assert different_image_index(0,[('same',),('same',)]) is None

def test_tallyqa_native_simple_complex_strata():
    rows=[dict(dataset='tallyqa',label=0,metadata={'is_simple':True}),
          dict(dataset='tallyqa',label=1,metadata={'is_simple':False})]
    m=result_metrics(rows,[dict(scores=[2,0],prediction=0),dict(scores=[2,0],prediction=0)])
    assert m['count']==2 and m['accuracy']==.5
    assert m['simple']==dict(count=1,accuracy=1)
    assert m['complex']==dict(count=1,accuracy=0)
