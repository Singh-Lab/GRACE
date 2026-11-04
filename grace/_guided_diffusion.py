#!/usr/bin/env python3
"""Guided discrete-diffusion baseline for constrained sequence optimization.
Trains a masked-denoiser prior on the task data, then generates sequences by MaskGIT-style
iterative refinement with ORACLE classifier-guidance (target ↑) + constraint feasibility guidance.
Same JSON I/O as optimize_generic.py (best_feasible_target / best_feasible_seq)."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
from optimize_generic import Oracle, BuiltinOracle, HammingOracle, make_oracle, onehot, decode

class Denoiser(nn.Module):
    def __init__(self, A, h=128):
        super().__init__()
        self.body=nn.Sequential(
            nn.Conv1d(A,h,7,padding='same'),nn.GELU(),
            nn.Conv1d(h,h,5,padding='same'),nn.GELU(),
            nn.Conv1d(h,h,5,padding='same'),nn.GELU(),
            nn.Conv1d(h,A,3,padding='same'))
    def forward(self,x): return self.body(x)

def load_seqs(csv, alph, L, maxn=20000):
    df=pd.read_csv(csv); pat=f'^[{alph}]+$'
    cols=[c for c in df.columns if df[c].dtype==object and df[c].astype(str).str.match(pat).mean()>0.5]
    if not cols: return []
    s=df[cols[0]].astype(str); s=s[s.str.len()>=L].str[:L]; s=s[s.str.match(pat)].tolist()
    if len(s)>maxn:
        import random; random.seed(0); s=random.sample(s,maxn)
    return s

def train_denoiser(seqs, A, L, alph, device, epochs=25):
    X=torch.stack([onehot(s,alph,device) for s in seqs]); m=Denoiser(A).to(device)
    opt=torch.optim.AdamW(m.parameters(),2e-3); N=len(X); bs=256
    for ep in range(epochs):
        perm=torch.randperm(N,device=device)
        for i in range(0,N,bs):
            xb=X[perm[i:i+bs]]
            r=torch.rand(xb.size(0),1,1,device=device)*0.9+0.1
            mask=(torch.rand(xb.size(0),1,L,device=device)<r)
            logits=m(xb*(~mask)); tgt=xb.argmax(1)
            l=F.cross_entropy(logits.transpose(1,2).reshape(-1,A),tgt.reshape(-1),reduction='none').reshape(xb.size(0),L)
            loss=(l*mask.squeeze(1)).sum()/mask.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()

def guided_generate(m, t_or, c_ors, c0, eps, sgn, A, L, alph, wt_oh, device, steps=8, guidance=8.0,
                    lam=25.0, beta=0.0, norm_guidance=False, mask_lo=0.05, mask_hi=0.30):
    import random
    # DIFFUSION INPAINTING: start from WT, mask a random fraction, regenerate guided
    # (matches the 'edit WT under constraint' framing of every other method)
    mf=random.uniform(mask_lo,mask_hi)
    x=wt_oh.clone().squeeze(0); filled=torch.ones(L,dtype=torch.bool,device=device)
    for p in torch.randperm(L,device=device)[:max(1,int(L*mf))]: x[:,p]=0; filled[p]=False
    base=int(filled.sum()); nmask=L-base
    for t in range(steps):
        keep=base+int(round(nmask*(t+1)/steps))
        with torch.enable_grad():
            P=F.softmax(m(x.unsqueeze(0)).squeeze(0),0).detach().requires_grad_(True)
            # The oracle must score the sequence actually being built. Positions
            # already committed are exact one-hot and can never change again, so
            # they are fed exactly; the denoiser's soft distribution stands only
            # where a letter is still open. Previously xg was P everywhere, so the
            # frozen wild-type positions were scored as the CNN's reconstruction of
            # themselves, and the constraint values driving the penalty belonged to
            # a sequence that was never a candidate. Gradients now flow only through
            # the open positions, which are the only ones guidance can act on.
            xg=torch.where(filled.unsqueeze(0), x, P).unsqueeze(0)
            obj=t_or(xg).sum()*sgn
            for i,o in enumerate(c_ors):
                obj=obj - lam*torch.clamp(torch.abs(o(xg).sum()-c0[i])-eps,min=0)
            g=torch.autograd.grad(obj,P)[0]
        # Categorical prior-preserving guidance.
        # The guided transition q maximizes  <q, guidance*g> - beta*KL(q || P),
        # whose closed form is  q  propto  P * exp(guidance*g/(1+beta)).
        #   beta = 0    -> DPS-style guidance
        #   beta -> inf -> q -> P, i.e. the pretrained sequence prior is preserved
        #
        # SCALE. `g` is the raw gradient of a learned oracle w.r.t. probabilities, so
        # its magnitude is an accident of that oracle's parameterisation. Added to
        # log P -- whose per-position spread is several nats -- an un-normalised g of
        # order 1e-3 gives the oracle no vote at all, and the sampler just replays the
        # prior. With norm_guidance we rescale g to unit RMS first, so `guidance` is a
        # genuine step size in logit units and is comparable across oracles and tasks.
        if norm_guidance:
            g = g / (g.std() + 1e-12)
        probs=F.softmax(torch.log(P.detach()+1e-9)+guidance*g/(1.0+beta),0)
        conf=probs.max(0).values.clone(); conf[filled]=-1e9
        nfill=keep-int(filled.sum())
        if nfill>0:
            for p in conf.topk(min(nfill,int((~filled).sum()))).indices:
                idx=torch.multinomial(probs[:,p],1); x[:,p]=0; x[idx,p]=1; filled[p]=True
    for p in range(L):
        if not filled[p]:
            probs=F.softmax(m(x.unsqueeze(0)).squeeze(0)[:,p],0)
            idx=torch.multinomial(probs,1); x[:,p]=0; x[idx,p]=1
    return decode(x,alph)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True); ap.add_argument('--target_oracle',required=True)
    ap.add_argument('--constraint_oracles',nargs='*',default=[]); ap.add_argument('--wt',required=True)
    ap.add_argument('--alphabet',required=True); ap.add_argument('--target_sign',type=float,default=1.0)
    ap.add_argument('--eps',type=float,default=0.1); ap.add_argument('--eps_list',default=None); ap.add_argument("--budget",type=int,default=140)
    ap.add_argument("--guidance",type=float,default=6.0); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--beta',type=float,default=0.0,
                    help="Prior-preserving KL weight. 0 = DPS-style guidance; "
                         "large beta keeps the guided transition close to the "
                         "unguided denoiser prior.")
    ap.add_argument('--norm_guidance',action='store_true',
                    help="Rescale the guidance gradient to unit RMS so --guidance is a "
                         "step size in logit units (default off = legacy behaviour).")
    ap.add_argument('--steps',type=int,default=8,help="Refinement steps per sample.")
    ap.add_argument('--max_seconds',type=float,default=0.0,
                    help="Stop drawing candidates once this many seconds have elapsed "
                         "and write what has been scored so far. 0 = no guard. The "
                         "budget is still reported, so a short run is visible as "
                         "n_candidates < budget rather than as a missing file.")
    ap.add_argument('--mask_lo',type=float,default=0.05)
    ap.add_argument('--mask_hi',type=float,default=0.30)
    ap.add_argument('--out',required=True); a=ap.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'; torch.manual_seed(a.seed); np.random.seed(a.seed)
    alph=a.alphabet; A=len(alph); L=len(a.wt); sgn=a.target_sign
    t_or=make_oracle(a.target_oracle,A,'t',device)
    c_ors=[HammingOracle(a.wt,alph,device) if p=='builtin:hamming'
           else (BuiltinOracle(p.split(':')[1],alph,device) if p.startswith('builtin:')
                 else make_oracle(p,A,f'c{i}',device))
           for i,p in enumerate(a.constraint_oracles)]
    eps_v=[float(v) for v in a.eps_list.split(',')] if a.eps_list else [a.eps]*len(c_ors)
    with torch.no_grad(): c0=[float(o(onehot(a.wt,alph,device).unsqueeze(0)).item()) for o in c_ors]
    seqs=load_seqs(a.data,alph,L)
    if len(seqs)<50: seqs=[a.wt]*200
    m=train_denoiser(seqs,A,L,alph,device)
    def feasible(cv): return all(abs(cv[i]-c0[i])<=eps_v[i] for i in range(len(c_ors)))
    best={'t':None,'edit':None,'seq':None}
    n_cand=0; n_feas=0; edits=[]
    # Per-candidate constraint values, for the violation / feasibility table.
    # Diffusion emits discrete sequences, so there is no separate relaxed space:
    # the same trace serves as both soft and decoded.
    hist=[]
    import time as _time; _t0=_time.time()
    with torch.no_grad(): wt_t=float(t_or(onehot(a.wt,alph,device).unsqueeze(0)).item())*sgn
    for b in range(a.budget):
        if a.max_seconds and _time.time()-_t0 > a.max_seconds:
            print(f"[guided_diffusion] wall guard at candidate {b}/{a.budget}",flush=True); break
        s=guided_generate(m,t_or,c_ors,c0,a.eps,sgn,A,L,alph,onehot(a.wt,alph,device).unsqueeze(0),device,steps=a.steps,guidance=a.guidance,
                          beta=a.beta,norm_guidance=a.norm_guidance,mask_lo=a.mask_lo,mask_hi=a.mask_hi)
        x=onehot(s,alph,device).unsqueeze(0)
        with torch.no_grad(): tv=float(t_or(x).item())*sgn; cv=[float(o(x).item()) for o in c_ors]
        n_cand+=1
        hist.append({'step':b,'target':tv,'constraints':cv,
                     'edit':sum(ch!=w for ch,w in zip(s,a.wt)),
                     'feasible':feasible(cv)})
        edits.append(sum(ch!=w for ch,w in zip(s,a.wt)))
        if feasible(cv):
            n_feas+=1
            if best['t'] is None or tv>best['t']:
                best={'t':tv,'edit':sum(ch!=w for ch,w in zip(s,a.wt)),'seq':s}
    out={'mode':'guided_diffusion','beta':a.beta,'guidance':a.guidance,
         'norm_guidance':bool(a.norm_guidance),'steps':a.steps,
         'mask_range':[a.mask_lo,a.mask_hi],'budget':a.budget,'wt_target':wt_t,
         'best_feasible_target':best['t'],'best_feasible_edit':best['edit'],
         'best_feasible_seq':best['seq'],'eps':a.eps,'n_prior_seqs':len(seqs),
         # candidate feasibility: fraction of GENERATED samples that satisfy all
         # constraints. This is the axis on which prior-preserving guidance is
         # expected to differ from DPS, independent of the best-of-budget score.
         'n_candidates':n_cand,'n_feasible':n_feas,
         'mean_edits_generated':(float(np.mean(edits)) if edits else None),
         'max_edits_generated':(int(max(edits)) if edits else None),
         'feasibility_rate':(n_feas/n_cand if n_cand else None),
         'history':hist,'soft_history':hist,'c0':c0,'eps_list':eps_v,
         'constraint_names':a.constraint_oracles}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); json.dump(out,open(a.out,'w'),indent=1)
    print(f"[guided_diffusion] beta={a.beta} best_feasible_target={best['t']} "
          f"feasible={n_feas}/{n_cand} ({100.0*n_feas/max(1,n_cand):.1f}%) "
          f"mean_edits={np.mean(edits) if edits else float('nan'):.1f} (prior_seqs={len(seqs)})")

if __name__=='__main__': main()
