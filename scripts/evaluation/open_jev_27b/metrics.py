"""Accuracy, calibration, paired tests, token counts and latency statistics."""
import math
import statistics


def stats(rows,full_required):
 supported=[x for x in rows if x.get('supported',True)];errors=sum(x.get('execution_error',False) for x in rows)
 valid=[x for x in supported if not x.get('execution_error',False)]
 result=dict(coverage_N=len(rows),required_N=full_required,supported_N=len(supported),unsupported_N=len(rows)-len(supported),execution_errors=errors,complete=len(rows)==full_required and errors==0,
  accuracy_pct=100*sum(x['correct'] for x in supported)/len(supported) if supported else None,invalid_outputs=sum(x.get('prediction') is None for x in valid),truncated_candidates=sum(x.get('truncated_candidate_count',0) for x in rows))
 # Only scalar-headed/matching routes provide candidate probabilities; do not
 # invent likelihoods for parsed native generated answers.
 for key in ['CE_raw','CE_calibrated','gold_probability_raw','gold_probability_calibrated']:
  values=[x[key] for x in valid if x.get(key) is not None];result[key]=sum(values)/len(values) if values else None
 probs=[x for x in valid if 'probabilities_raw' in x]
 if probs:
  result['high_confidence_wrong_raw']=sum(max(x['probabilities_raw'])>=.9 and not x['correct'] for x in probs);ece=0.
  for bin in range(15):
   xs=[x for x in probs if min(14,int(max(x['probabilities_raw'])*15))==bin]
   if xs:ece+=len(xs)/len(probs)*abs(sum(max(x['probabilities_raw']) for x in xs)/len(xs)-sum(x['correct'] for x in xs)/len(xs))
  result['ECE15_raw']=ece
 return result
def pair(a,b):
 import numpy as np
 aa={x['uid']:x for x in a if x.get('supported',True) and not x.get('execution_error',False)};bb={x['uid']:x for x in b if x.get('supported',True) and not x.get('execution_error',False)};uids=sorted(set(aa)&set(bb));groups={};gained=lost=0
 for uid in uids:
  x=aa[uid];y=bb[uid];assert x['label']==y['label'] and x['candidate_N']==y['candidate_N'] and x['cluster']==y['cluster'];d=int(x['correct'])-int(y['correct']);gained+=d==1;lost+=d==-1;v=groups.setdefault(x['cluster'],[0,0]);v[0]+=d;v[1]+=1
 if not uids:raise ValueError('No common supported UIDs')
 v=np.array(list(groups.values()),dtype=np.float64);rng=np.random.default_rng(20261027);boot=[]
 for start in range(0,2000,50):
  ix=rng.integers(0,len(v),size=(min(50,2000-start),len(v)));sample=v[ix];boot.extend((100*sample[:,:,0].sum(1)/sample[:,:,1].sum(1)).tolist())
 n=gained+lost;k=min(gained,lost)
 if n:
  logmass=math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)-n*math.log(2);scaled=term=1.
  for j in range(k,0,-1):term*=j/(n-j+1);scaled+=term
  logp=min(0.,logmass+math.log(scaled)+math.log(2));pvalue=math.exp(logp)
 else:logp=0.;pvalue=1.
 return dict(common_supported_UID_N=len(uids),delta_pp=100*(gained-lost)/len(uids),gained=gained,lost=lost,paired_cluster_bootstrap_CI95=np.quantile(boot,[.025,.975]).tolist(),McNemar_exact_two_sided_p=pvalue,McNemar_log_p=logp,multiple_comparisons='unadjusted p; all contrasts retained')


def distribution(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    def percentile(q):
        position = (len(values) - 1) * q
        low = int(position)
        fraction = position - low
        return values[low] * (1 - fraction) + values[min(low + 1, len(values) - 1)] * fraction
    return dict(n=len(values), mean=statistics.mean(values), p50=percentile(.5), p95=percentile(.95), p99=percentile(.99), total=sum(values))


def summarize_rows(rows, required):
    supported = [row for row in rows if row.get('supported', True)]
    valid = [row for row in supported if not row['execution_error']]
    times = [row['e2e_ms'] for row in valid]
    stats = distribution(times)
    return dict(
        coverage_N=len(rows), required_N=required,
        supported_N=len(supported), unsupported_N=len(rows) - len(supported),
        execution_errors=sum(row['execution_error'] for row in rows),
        accuracy_pct=100 * sum(row['correct'] for row in supported) / len(supported) if supported else None,
        invalid_outputs=sum(row.get('prediction') is None for row in valid),
        latency_ms=stats, throughput_qps=1000 * len(times) / sum(times) if times else None,
        generated_tokens=distribution([row.get('generated_tokens') for row in valid]),
        visible_generated_tokens=distribution([row.get('visible_generated_tokens') for row in valid]),
        input_positions=distribution([row.get('input_tokens_processed') for row in valid]),
        peak_memory_GiB=max((row.get('peak_memory_GiB', 0) for row in valid), default=None),
        generation_budget_exhausted=sum(row.get('budget_exhausted', False) for row in valid),
    )


