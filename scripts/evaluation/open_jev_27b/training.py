"""Shared two-stage optimization primitives; confirmation never invokes them."""
import torch
from torch.nn import functional as F
from .connector import alignment_loss
from .scorer import external_only

def trusted_decision_loss(logits,teacher_logits,label):
    ce=F.cross_entropy(logits[None],torch.tensor([label],device=logits.device))
    p=teacher_logits.detach().float().softmax(-1)
    trusted=int(teacher_logits.argmax())==label and float(p[label])>=.7
    kl=F.kl_div(logits.log_softmax(-1),p,reduction='sum') if trusted else logits.sum()*0
    return ce+kl,dict(CE=float(ce.detach()),KL=float(kl.detach()),teacher_trusted=trusted)

def serial_vjp(model,batches,feature,weights):
    total=torch.zeros_like(feature)
    for i in range(len(batches)):
        leaf=feature.detach().requires_grad_(True)
        score=model.logits(batches,leaf,order=[i])[0]
        total.add_(torch.autograd.grad(score,leaf,grad_outputs=weights[i])[0])
    assert torch.isfinite(total).all(); return total

def alignment_update(bridge,optimizer,records,grids,targets,loss_mode):
    optimizer.zero_grad(set_to_none=True)
    loss=alignment_loss(bridge(records,grids),targets,bridge.teacher_std,loss_mode)
    loss.backward(); norm=torch.nn.utils.clip_grad_norm_(bridge.parameters(),1.)
    assert torch.isfinite(norm) and norm>0; optimizer.step()
    return dict(loss=float(loss.detach()),gradient_norm=float(norm))

def decision_update(bridge,optimizer,official,batch,loss_mode):
    """Batch fields: source, grid, batches, teacher_feature and TRAIN label."""
    assert all(not p.requires_grad for p in official.mm.parameters())
    optimizer.zero_grad(set_to_none=True); reports=[]
    with external_only(official):
        for item in batch:
            feat=bridge([item['source']],[item['grid']])[0]
            with torch.no_grad():
                z=official.logits(item['batches'],feat.detach())
                teacher=official.logits(item['batches'],item['teacher_feature'])
            leaf=z.detach().requires_grad_(True)
            loss,report=trusted_decision_loss(leaf,teacher,item['label'])
            dz=torch.autograd.grad(loss/len(batch),leaf)[0]
            grad=serial_vjp(official,item['batches'],feat,dz)
            anchor=alignment_loss([feat],[item['teacher_feature']],bridge.teacher_std,loss_mode)/len(batch)
            anchor.backward(retain_graph=True); feat.backward(grad)
            reports.append(dict(report,feature_anchor=float(anchor.detach())*len(batch)))
    norm=torch.nn.utils.clip_grad_norm_(bridge.parameters(),1.)
    assert torch.isfinite(norm) and norm>0
    assert all(p.grad is None for p in official.mm.parameters())
    assert all(p.grad is None for p in official.reference.decision.head.parameters())
    optimizer.step(); official.unchanged()
    return dict(samples=reports,gradient_norm=float(norm))
