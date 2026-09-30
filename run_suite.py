"""Normalization-correct PDM controls. Run from remote PDM project root."""
import argparse, copy, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.vae_gaussian import GaussianVAE
from utils.dataset import ShapeNetCore
from tools.pointcloud_normalization import normalize_pointcloud, validate_checkpoint_scale_mode

ROOT=Path('experiments/pdm_z_control_20260925')
CKPT='logs_gen/GEN_2026_06_29__02_58_19_Clean_VAE_From_Scratch_KL001/ckpt_0.730159_300000.pt'
TARGET='targets/stageC8E_fixed_airplane_target.npy'
DATA='data/shapenet_v2pc15k.h5'
def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def dump(p,d):
    Path(p).write_text(json.dumps(d,indent=2))
def load():
    c=torch.load(CKPT,map_location='cpu',weights_only=False)
    validate_checkpoint_scale_mode(c['args'],'shape_unit')
    m=GaussianVAE(c['args']).cuda().eval(); m.load_state_dict(c['state_dict'])
    return m,c

def stats(x):
    x=x.reshape(-1,3).float()
    return dict(min=x.min(0).values.tolist(),max=x.max(0).values.tolist(),mean=x.mean(0).tolist(),std=float(x.std()),max_extent=float((x.max(0).values-x.min(0).values).max()),rms_radius=float((x-x.mean(0)).square().sum(1).mean().sqrt()))
def dataset(split):
    return ShapeNetCore(DATA,['chair'],split,'shape_unit')
def replace(x,tr):
    return torch.cat([x[:,:-len(tr)],tr[None].expand(len(x),-1,-1)],1)

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    m,c=load(); ds=dataset('train')
    raw=torch.from_numpy(np.load(TARGET)).float().reshape(1,-1,3)
    y,params=normalize_pointcloud(raw,mode='shape_unit',return_stats=True)
    np.save(ROOT/'target_shape_unit.npy',y[0].numpy())
    audit=dict(checkpoint=CKPT,checkpoint_args=vars(c['args']),original_target=TARGET,raw_target=stats(raw),corrected_target=stats(y),source_shape_unit_examples=[stats(ds[i]['pointcloud']) for i in range(4)],target_shift=params['shift'].tolist(),target_scale=params['scale'].tolist(),target_raw_sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest(),errors=['checkpoint trained shape_unit but latest optimizer/train/eval use shape_bbox','target loaded as raw H5 without normalization'],alpha_bar_T=float(m.diffusion.var_sched.alpha_bars[-1]))
    dump(ROOT/'normalization_audit.json',audit)
    print(json.dumps(audit,indent=2),flush=True)

def condition(mode,z,zt,zkey,zstar):
    if mode in ('geometric','control'): return zt
    if mode=='fixed_key': return zkey.expand(len(z),-1)
    if mode in ('fixed_target','target_only'): return zstar.expand(len(z),-1)
    raise ValueError(mode)

@torch.no_grad()
def measure(model,base,ds,y,tr,zkey,zstar,mode,out,n=64):
    model.eval(); base.eval(); saved={k:[] for k in ['A','B','C','D','Ztarget','source']}
    # Explicitly seeded before initial noise; eval randomness independent of training.
    for start in range(0,n,8):
        x=torch.stack([ds[i]['pointcloud'] for i in range(start,min(start+8,n))]).cuda()
        z,_=base.encoder(x); zt,_=base.encoder(replace(x,tr)); zb=condition(mode,z,zt,zkey,zstar)
        g=torch.Generator(device='cuda').manual_seed(42000+start)
        initial=torch.randn(x.shape,device='cuda',generator=g)
        for name,net,ctx in [('A',base,z),('B',base,zb),('C',model,z),('D',model,zb),('Ztarget',base,zstar.expand(len(x),-1))]:
            torch.manual_seed(52000+start)
            cloud=net.sample(ctx,x.shape[1],net.args.flexibility,initial_x_T=initial.clone())
            saved[name].append(cloud.cpu())
        saved['source'].append(x.cpu())
    saved={k:torch.cat(v) for k,v in saved.items()}
    metrics={}; arrays={}
    for name in ['A','B','C','D','Ztarget']:
        pc=saved[name].cuda(); target=y.expand(len(pc),-1,-1)
        vals=[]
        for j in range(0,len(pc),4):
            d=torch.cdist(pc[j:j+4],target[j:j+4]); p=d.min(2).values; q=d.min(1).values
            dsr=torch.cdist(pc[j:j+4],saved['source'][j:j+4].cuda())
            pn=normalize_pointcloud(pc[j:j+4],mode='shape_bbox'); yn=normalize_pointcloud(target[j:j+4],mode='shape_bbox')
            dn=torch.cdist(pn,yn)
            precision=(p<0.1).float().mean(1); recall=(q<0.1).float().mean(1)
            vals.append(torch.stack([p.mean(1)+q.mean(1),p.mean(1),q.mean(1),dsr.min(2).values.mean(1)+dsr.min(1).values.mean(1),dn.min(2).values.mean(1)+dn.min(1).values.mean(1),2*precision*recall/(precision+recall+1e-8),((pc[j:j+4]-pc[j:j+4].mean(1,keepdim=True)).square().sum(2).mean(1)).sqrt()],1).cpu())
        v=torch.cat(vals).numpy(); keys=['target_cd','prediction_to_target','target_to_prediction','source_cd','shape_bbox_cd','fscore_0.1','rms_radius']
        metrics[name]={k:float(v[:,i].mean()) for i,k in enumerate(keys)}
        arrays[name]=v
    gain=arrays['C'][:,0]-arrays['D'][:,0]
    rng=np.random.default_rng(42); boot=gain[rng.integers(0,len(gain),(10000,len(gain)))].mean(1)
    metrics['conditional_gain']={'mean':float(gain.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'fraction_positive':float((gain>0).mean())}
    metrics['clean_source_degradation_vs_base']=metrics['C']['source_cd']/metrics['A']['source_cd']-1
    metrics['target_leakage']=metrics['A']['target_cd']-metrics['C']['target_cd']
    metrics['protocol']={'mode':mode,'n':n,'normalization':'shape_unit','paired_initial_and_reverse_noise':True,'initialization':'standard Gaussian','cd':'sum bidirectional mean Euclidean distances','fscore_threshold':0.1,'fixed_key_definition':'mean triggered latent of first 64 train shapes; no test fitting','fixed_target_definition':'encoder posterior mean of normalized target','selected_condition':mode,'target_stats':stats(y.cpu())}
    dump(out/'metrics.json',metrics)
    np.savez_compressed(out/'samples.npz',**{k:v.numpy() for k,v in saved.items()},target=y.cpu().numpy(),zkey=zkey.cpu().numpy(),zstar=zstar.cpu().numpy())
    np.savez_compressed(out/'per_sample_metrics.npz',**arrays)
    print('EVAL',json.dumps(metrics),flush=True)

def train(a):
    seed(a.seed); out=ROOT/a.name; out.mkdir(parents=True,exist_ok=False)
    dump(out/'config.json',vars(a))
    base,c=load(); model=copy.deepcopy(base)
    for p in base.parameters(): p.requires_grad_(False)
    for p in model.encoder.parameters(): p.requires_grad_(False)
    y=torch.from_numpy(np.load(ROOT/'target_shape_unit.npy')).float()[None].cuda()
    payload=torch.load(ROOT/a.trigger_dir/'optimized_trigger.pt',map_location='cpu',weights_only=False)
    tr=payload['trigger'].cuda()
    train_ds=dataset('train'); test_ds=dataset('test')
    with torch.no_grad():
        zs=[]
        for i in range(0,64,16):
            x=torch.stack([train_ds[j]['pointcloud'] for j in range(i,i+16)]).cuda()
            zs.append(base.encoder(replace(x,tr))[0])
        zkey=torch.cat(zs).mean(0,keepdim=True); zstar=base.encoder(y)[0]
    torch.save({'zkey':zkey.cpu(),'zstar':zstar.cpu(),'trigger':tr.cpu()},out/'conditions.pt')
    loader=DataLoader(train_ds,batch_size=32,shuffle=True,num_workers=0,drop_last=True,generator=torch.Generator().manual_seed(a.seed))
    it=iter(loader); opt=torch.optim.Adam(model.diffusion.parameters(),lr=1e-4)
    seed(a.seed)
    started=time.time()
    with (out/'train.jsonl').open('w') as log:
        for step in range(1,a.steps+1):
            model.encoder.eval(); model.diffusion.train()
            try: x=next(it)['pointcloud'].cuda()
            except StopIteration: it=iter(loader); x=next(it)['pointcloud'].cuda()
            b=len(x); idx=torch.randperm(b,device='cuda')[:3]
            with torch.no_grad():
                z=base.encoder(x)[0]; zt=base.encoder(replace(x[idx],tr))[0]
                zb=condition(a.mode,z[idx],zt,zkey,zstar)
            sched=model.diffusion.var_sched
            t=sched.uniform_sample_t(b); ab=sched.alpha_bars[t,None,None]; eps=torch.randn_like(x)
            xt=ab.sqrt()*x+(1-ab).sqrt()*eps
            pc=model.diffusion.net(xt,beta=sched.betas[t],context=z)
            with torch.no_grad(): teacher=base.diffusion.net(xt,beta=sched.betas[t],context=z)
            lc=F.mse_loss(pc,eps); lr=F.mse_loss(pc,teacher)
            tb=sched.uniform_sample_t(len(idx)); abb=sched.alpha_bars[tb,None,None]; eb=torch.randn_like(x[idx]); yt=abb.sqrt()*y+(1-abb).sqrt()*eb
            pb=model.diffusion.net(yt,beta=sched.betas[tb],context=zb); lb=F.mse_loss(pb,eb)
            loss=lc+lr+a.bd_weight*lb
            if a.mode=='target_only': loss=lb
            opt.zero_grad(set_to_none=True); loss.backward(); gn=torch.nn.utils.clip_grad_norm_(model.diffusion.parameters(),10.); opt.step()
            if step==1 or step%100==0:
                row=dict(step=step,clean=float(lc),retain=float(lr),bd=float(lb),loss=float(loss),grad=float(gn),seconds=time.time()-started)
                log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row),flush=True)
            if step in {500,a.steps}:
                torch.save(dict(args=c['args'],state_dict=model.state_dict(),optimizer=opt.state_dict(),step=step,protocol=vars(a)),out/f'ckpt_{step:06d}.pt')
    measure(model,base,test_ds,y,tr,zkey,zstar,a.mode,out,n=a.samples)
    (out/'COMPLETE').write_text('finished training and evaluation\n')

if __name__=='__main__':
    torch.set_num_threads(4)
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); p.add_argument('--trigger_dir',default='trigger'); p.add_argument('--mode',default='geometric',choices=['control','geometric','fixed_key','fixed_target','target_only']); p.add_argument('--name',default='geometric_seed0'); p.add_argument('--seed',type=int,default=0); p.add_argument('--steps',type=int,default=2000); p.add_argument('--samples',type=int,default=64); p.add_argument('--bd_weight',type=float,default=.1)
    a=p.parse_args()
    if a.prepare: prepare()
    else: train(a)
