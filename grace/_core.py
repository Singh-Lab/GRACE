"""Self-contained GRACE + AdaLead + DirectedEvolution optimizer for pluggable
one-hot CNN oracles (RNA/protein). Target: improve target property while holding
constraint properties within +/- eps of the wild-type value.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).parent))
try:                                     # package use
    from .projection import compute_grace_direction, flatten_gradients
except ImportError:                       # direct `python grace/_core.py` use
    from projection import compute_grace_direction, flatten_gradients

# ---- generic CNN rebuild (infer alphabet size + arch from checkpoint) ----
def build_cnn(state, A):
    # architecture must match trainers (CanyaCNN / RNACNN): 3 conv + head
    ch=[int(state['conv.0.weight'].shape[0])]  # first conv out
    class G(nn.Module):
        def __init__(s):
            super().__init__()
            layers=[]; keys=sorted({k.split('.')[1] for k in state if k.startswith('conv.') and k.endswith('.weight') and 'conv.'+k.split('.')[1]+'.weight' in state})
            # rebuild by reading conv shapes
            convs=[k for k in state if k.startswith('conv.') and k.endswith('.weight') and state[k].dim()==3]
            convs=sorted(convs,key=lambda k:int(k.split('.')[1]))
            mods=[]; inc=A
            for ck in convs:
                w=state[ck]; outc,_,ks=w.shape; idx=int(ck.split('.')[1])
                mods.append((idx,nn.Conv1d(inc,outc,ks,padding='same')))
                bn_i=idx+1
                if f'conv.{bn_i}.weight' in state: mods.append((bn_i,nn.BatchNorm1d(outc)))
                inc=outc
            # assemble conv sequential with GELU between (match trainer order conv,bn,gelu)
            seq=[]; maxidx=max(i for i,_ in mods)
            d={i:m for i,m in mods}
            i=0
            while i<=maxidx+2:
                if i in d: seq.append(d[i])
                elif any(f'conv.{i}' not in state for _ in [0]) and i-1 in d and isinstance(d.get(i-1),(nn.Conv1d,nn.BatchNorm1d)):
                    pass
                i+=1
            # simpler: conv,bn,gelu repeating -> reconstruct explicitly
            s.conv=None
    # The above is fragile; instead hardcode the two known arches by A.
    raise RuntimeError('use build_known')

def build_known(state, A, L):
    """Rebuild the exact arch from train_canya/train_toehold."""
    c0=state['conv.0.weight'].shape[0]
    if A==20:  # CanyaCNN: 128,192,256 k=7,5,3
        conv=nn.Sequential(nn.Conv1d(20,128,7,padding='same'),nn.BatchNorm1d(128),nn.GELU(),
            nn.Conv1d(128,192,5,padding='same'),nn.BatchNorm1d(192),nn.GELU(),
            nn.Conv1d(192,256,3,padding='same'),nn.BatchNorm1d(256),nn.GELU(),nn.AdaptiveAvgPool1d(1))
        head=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(0.2),nn.Linear(128,1))
    else:  # RNACNN: 128,128,128 k=7,5,3
        conv=nn.Sequential(nn.Conv1d(4,128,7,padding='same'),nn.BatchNorm1d(128),nn.GELU(),
            nn.Conv1d(128,128,5,padding='same'),nn.BatchNorm1d(128),nn.GELU(),
            nn.Conv1d(128,128,3,padding='same'),nn.BatchNorm1d(128),nn.GELU(),nn.AdaptiveAvgPool1d(1))
        head=nn.Sequential(nn.Linear(128,64),nn.GELU(),nn.Dropout(0.2),nn.Linear(64,1))
    class M(nn.Module):
        def __init__(s): super().__init__(); s.conv=conv; s.head=head
        def forward(s,x): return s.head(s.conv(x).squeeze(-1)).squeeze(-1)
    m=M(); m.load_state_dict(state); return m

class Oracle:
    def __init__(self, path, A, name, device):
        ck=torch.load(path,map_location=device,weights_only=False)
        self.L=ck['L']; self.mu=ck['mean']; self.sd=ck['std']; self.name=name
        self.m=build_known(ck['state'],A,ck['L']).to(device).eval()
        for p in self.m.parameters(): p.requires_grad_(False)
    def __call__(self,x):  # x: (B,A,L) -> (B,) denormalized
        return self.m(x)*self.sd+self.mu


class HammingOracle:
    """Differentiable mutation count from the wild type: L - sum(x * wt_onehot).
    Exact on one-hot inputs, and linear so it gives a clean gradient on relaxed ones.
    Used as a constraint with c0 = 0, so |H(x) - 0| <= eps is a substitution budget."""
    def __init__(self, wt, alph, device):
        A = len(alph)
        w = torch.zeros(A, len(wt), device=device)
        for j, c in enumerate(wt):
            if c in alph: w[alph.index(c), j] = 1.0
        self.w = w.unsqueeze(0)
        self.L = len(wt)
    def __call__(self, x):
        if x.dim() == 2: x = x.unsqueeze(0)
        return self.L - (x * self.w).sum(dim=(1, 2))


class BuiltinOracle:
    """Differentiable computable property (charge / hydrophobicity) from one-hot."""
    KD={'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
    CH={'D':-1.0,'E':-1.0,'K':1.0,'R':1.0,'H':0.5}
    def __init__(self,kind,alph,device):
        self.name=kind
        vec=[ (1.0 if (kind=='gc' and a in 'GC') else (self.KD.get(a,0.0) if kind=='hydro' else self.CH.get(a,0.0))) for a in alph]
        self.w=torch.tensor(vec,device=device).view(len(alph),1)  # (A,1)
    def __call__(self,x):  # x:(B,A,L) -> (B,)
        v=(x*self.w.unsqueeze(0)).sum(dim=(1,2))
        return v/x.shape[2] if self.name=='gc' else v

def onehot(seq,alph,device):
    A=len(alph); x=torch.zeros(A,len(seq),device=device)
    for j,c in enumerate(seq):
        if c in alph: x[alph.index(c),j]=1.0
    return x

def decode(logits,alph):
    idx=logits.argmax(0)
    return ''.join(alph[i] for i in idx.tolist())

_v2_info={}

def make_oracle(path, A, name, device):
    """Oracle factory. 'parm:<spec.pt>' loads a PARM ensemble (see parm_oracle.py);
    anything else is the standard CNN checkpoint."""
    if path.startswith('parm:'):
        from parm_oracle import PARMOracle
        return PARMOracle(path[len('parm:'):], device)
    if path.startswith('esm:'):
        from esm_oracle import ESMOracle
        return ESMOracle(path[len('esm:'):], device)
    if path.startswith('dna:'):
        from dna_oracle import DNACNNOracle
        return DNACNNOracle(path, device)
    if path.startswith('robust:'):
        # experiments/robust_predictor_grace -- multi-predictor target/constraint.
        # Refuses to load the held-out evaluator P3; see that module's docstring.
        import sys as _s
        _s.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                              'experiments' / 'robust_predictor_grace' / 'code'))
        from robust_oracle import MultiPredOracle
        return MultiPredOracle(path[len('robust:'):], device)
    return Oracle(path, A, name, device)



def projection_drift_probe(logits, t_or, all_c, c0_all, eps_all, tol_v, sgn, A, K, lr,
                           steps, device, active_tol_default=0.3):
    """Run a short PROJECTION-ONLY trajectory from the wild type and report the
    fraction of steps spent outside the constraint band. This is what the DriftGate
    decides on: if projection alone never leaves the band the reactive stage has
    nothing to restore. Runs on a CLONE, and the caller restores the RNG, so the
    real trajectory is bit-identical to an ungated run."""
    lg = logits.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([lg], lr=lr)
    outside = 0
    for step in range(steps):
        tau = max(0.3, 1.0 * (0.3) ** (step / max(1, steps)))
        xs = []
        for _ in range(K):
            g = -torch.log(-torch.log(torch.rand_like(lg) + 1e-9) + 1e-9)
            Pk = F.softmax((lg + 0.5 * g) / tau, dim=0)
            hard = F.one_hot(Pk.argmax(0), A).T.float()
            xs.append(hard + (Pk - Pk.detach()))
        x = torch.stack(xs)
        tval = (t_or(x).mean()) * sgn
        cvals = [o(x).mean() for o in all_c]
        if any(abs(float(cvals[i].item()) - c0_all[i]) > eps_all[i] for i in range(len(all_c))):
            outside += 1
        g_t = flatten_gradients(torch.autograd.grad(tval, [lg], retain_graph=True, allow_unused=True), [lg])
        g_cs = []
        for i, cv in enumerate(cvals):
            tl = tol_v[i] if i < len(tol_v) else active_tol_default
            if abs(float(cv.item()) - c0_all[i]) >= tl * eps_all[i]:
                g_cs.append(flatten_gradients(torch.autograd.grad(cv, [lg], retain_graph=True, allow_unused=True), [lg]))
        d, _ = compute_grace_direction(g_t, g_cs, 1e-6) if g_cs else (g_t, None)
        opt.zero_grad(); lg.grad = -d.view_as(lg); opt.step()
    return outside / max(1, steps)


def run(args):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    alph=args.alphabet; A=len(alph); L=len(args.wt)
    t_or=make_oracle(args.target_oracle,A,'target',device)
    c_ors=[]
    for i,p in enumerate(args.constraint_oracles):
        if p == 'builtin:hamming': c_ors.append(HammingOracle(args.wt,alph,device))
        elif p.startswith('builtin:'): c_ors.append(BuiltinOracle(p.split(':')[1],alph,device))
        else: c_ors.append(make_oracle(p,A,f'c{i}',device))
    # Per-constraint tolerances. Without --eps_list every constraint shares --eps,
    # which is the existing behaviour for every task already in the benchmark.
    eps_v = [float(v) for v in args.eps_list.split(',')] if args.eps_list else [args.eps]*len(c_ors)
    assert len(eps_v) == len(c_ors), f"--eps_list has {len(eps_v)} values for {len(c_ors)} constraints"
    # Per-constraint active-set tolerance. A constraint enters the active set once
    # |c - c0| >= tol * eps, so with one shared tol a wide budget-style constraint
    # (e.g. 10 substitutions) starts resisting at 0.3*10 = 3 edits, far too early.
    tol_v = [float(v) for v in args.tol_list.split(',')] if args.tol_list else [args.active_set_tol]*len(c_ors)
    assert len(tol_v) == len(c_ors), f"--tol_list has {len(tol_v)} values for {len(c_ors)} constraints"
    no_proj = {int(v) for v in args.no_project.split(',')} if args.no_project else set()
    assert all(0 <= i < len(c_ors) for i in no_proj), f"--no_project index out of range for {len(c_ors)} constraints"
    sgn=args.target_sign
    wt_x=onehot(args.wt,alph,device).unsqueeze(0)
    with torch.no_grad():
        t0=float(t_or(wt_x).item()); c0=[float(o(wt_x).item()) for o in c_ors]
    def feasible(cv): return all(abs(cv[i]-c0[i])<=eps_v[i] for i in range(len(c_ors)))
    best={'t':None,'edit':None,'seq':None}
    # Optional per-evaluation trace (--log_history). consider() is the single point
    # every mode scores a DISCRETE sequence through, so recording here gives one
    # trajectory definition shared by all of them. Off by default: nothing about the
    # optimisation changes, only whether the evaluations are also written out.
    hist=[]
    # Oracle-evaluation counter. Every method is compared at a matched number of
    # these, never at matched steps: one gradient step costs K*(1+n_constraints)
    # evaluations while one directed-evolution round costs 12*L, so equal steps
    # are nowhere near equal work.
    nev={'n':0}
    # Soft-space trace. consider() records the DECODED sequence, which is what
    # feasibility should be judged on, but constraint violation during optimization
    # lives in the relaxed space the optimizer actually moves through. Both are
    # needed: violation (soft) says how hard the method pushes past the boundary,
    # feasibility (decoded) says whether the sequence it would hand you is legal.
    soft_hist=[]
    # Per-step diagnostics for the motivating experiment:
    #   fo_norm  eta*||J_c d||, the first-order constraint interference of the
    #            update DIRECTION, measured before any reactive correction. It is
    #            a property of the direction, so it does not depend on the
    #            optimizer's rescaling.
    #   viol     sum_i max(0, |c_i - c_i^0|/eps_i - 1), the total violation
    #            normalised per constraint so that constraints on different
    #            scales contribute comparably.
    #   target   the predicted target score.
    drift_hist=[]
    def log_drift(step, cvals, tval, g_all, d):
        if not args.log_drift:
            return
        Jd = torch.stack([g @ d for g in g_all]) if g_all else torch.zeros(1)
        viol = sum(max(0.0, abs(float(cvals[i].item())-c0[i])/eps_v[i] - 1.0)
                   for i in range(len(cvals)))
        drift_hist.append({'step': int(step),
                           'fo_norm': float(args.lr*Jd.norm().item()),
                           'viol': float(viol),
                           'target': float(tval.item()),
                           'c_soft': [float(c.item()) for c in cvals]})
    def log_soft(step, cvals_t, tval_t):
        if args.log_history:
            soft_hist.append({'step': step, 'target': float(tval_t),
                              'constraints': [float(c) for c in cvals_t]})
    def consider(seq, step=None):
        x=onehot(seq,alph,device).unsqueeze(0)
        with torch.no_grad():
            tv=float(t_or(x).item())*sgn; cv=[float(o(x).item()) for o in c_ors]
        nev['n']+=1+len(c_ors)
        ed=sum(a!=b for a,b in zip(seq,args.wt))
        ok=feasible(cv)
        if args.log_history:
            # Search modes (botorch, adalead, ...) call consider() without a step;
            # fall back to the running evaluation count so every mode has a
            # monotone x-axis. `step` stays None there, so a reader can still tell
            # gradient steps from query counts.
            hist.append({'step': step if step is not None else len(hist),
                         'raw_step': step, 'target':tv, 'constraints':cv,
                         'edit':ed, 'feasible':ok, 'seq': seq})
        if ok and (best['t'] is None or tv>best['t']):
            best['t']=tv; best['edit']=ed; best['seq']=seq

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if args.mode=='grace_lagrangian':
        logits=(onehot(args.wt,alph,device)*4.0).clone().requires_grad_(True)
        lambdas=[torch.zeros((),device=device) for _ in c_ors]
        opt=(torch.optim.SGD([logits],lr=args.lr) if args.plain_sgd
             else torch.optim.Adam([logits],lr=args.lr))
        K=args.K
        for step in range(args.steps):
            tau=max(0.3,1.0*(0.3)**(step/max(1,args.steps)))
            xs=[]
            for _ in range(K):
                if args.det_relax:
                    xs.append(F.softmax(logits/tau,dim=0)); break
                g=-torch.log(-torch.log(torch.rand_like(logits)+1e-9)+1e-9)  # Gumbel
                Pk=F.softmax((logits+0.5*g)/tau,dim=0)
                hard=F.one_hot(Pk.argmax(0),A).T.float()
                xs.append(hard+(Pk-Pk.detach()))
            x=torch.stack(xs)  # (K,A,L)
            tval=(t_or(x).mean())*sgn
            cvals=[o(x).mean() for o in c_ors]
            nev['n']+=K*(1+len(c_ors))
            log_soft(step, cvals, tval)
            g_t=flatten_gradients(torch.autograd.grad(tval,[logits],retain_graph=True,allow_unused=True),[logits])
            # drift logging needs EVERY constraint gradient, not only the active
            # ones, so they are computed once here and reused below
            g_all=[flatten_gradients(torch.autograd.grad(cv,[logits],retain_graph=True,allow_unused=True),[logits])
                   for cv in cvals] if args.log_drift else None
            # active-set: project vs constraints near/over boundary
            g_cs=[]
            for i,cv in enumerate(cvals):
                if i in no_proj: continue
                if abs(float(cv.item())-c0[i])>=tol_v[i]*eps_v[i]:
                    g_cs.append(g_all[i] if g_all is not None else
                                flatten_gradients(torch.autograd.grad(cv,[logits],retain_graph=True,allow_unused=True),[logits]))
            d,_=compute_grace_direction(g_t,g_cs,1e-6) if g_cs else (g_t,None)
            # ALM penalty
            pen=torch.zeros((),device=device)
            for i,cv in enumerate(cvals):
                viol=torch.clamp(torch.abs(cv-c0[i])-eps_v[i],min=0)
                pen=pen+lambdas[i]*viol+(args.rho/2)*viol**2
            g_alm=flatten_gradients(torch.autograd.grad(pen,[logits],retain_graph=False,allow_unused=True),[logits]) if pen.requires_grad else torch.zeros_like(d)
            log_drift(step, cvals, tval, g_all, d)
            opt.zero_grad(); logits.grad=-(d - g_alm).view_as(logits); opt.step()
            with torch.no_grad():
                for i,cv in enumerate(cvals):
                    lambdas[i]=torch.clamp(lambdas[i]+args.dual_lr*(abs(cv.item()-c0[i])-args.eps),0,100)
            if step%3==0: consider(decode(logits.detach(),alph), step)
        consider(decode(logits.detach(),alph), args.steps)
    elif args.mode=='grace_v2':
        # SEPARATE from grace_lagrangian; baseline is untouched.
        # (1) plausibility constraint  (2) adaptive rho  (3) curvature-gated ALM
        from grace_v2 import AdaptiveRho, AlignmentGate, PlausibilityOracle, train_prior
        import csv as _csv
        global _v2_info
        extra=[]
        if args.prior_data:
            rows=list(_csv.DictReader(open(args.prior_data)))
            col=next((c for c in rows[0] if c.lower() in ('seq','sequence','protein','aa_seq','switch')),None)
            lib=[r[col] for r in rows if len(r[col])==L][:8000]
            prior=train_prior(lib,A,L,onehot,alph,device,epochs=args.prior_epochs)
            _po=PlausibilityOracle(prior)
            if args.plaus_eps<0:   # negative => calibrate from the library spread
                args.plaus_eps=_po.calibrate_eps(lib[:2000],onehot,alph,device,q=args.plaus_q)
            extra.append(_po)
        all_c=c_ors+extra
        with torch.no_grad():
            c0_all=c0+[float(o(wt_x).item()) for o in extra]
        eps_all=list(eps_v)+[args.plaus_eps]*len(extra)
        def feasible_all(cv): return all(abs(cv[i]-c0_all[i])<=eps_all[i] for i in range(len(all_c)))
        def consider_v2(seq, step=None):
            x=onehot(seq,alph,device).unsqueeze(0)
            with torch.no_grad():
                tv=float(t_or(x).item())*sgn; cv=[float(o(x).item()) for o in all_c]
            nev['n']+=1+len(all_c)
            ed=sum(a!=b for a,b in zip(seq,args.wt))
            ok=feasible_all(cv)
            # grace_v2 has its own accept function, so --log_history was silently a
            # no-op for this mode; record the same fields consider() does
            if args.log_history:
                hist.append({'step': step if step is not None else len(hist),
                             'raw_step': step, 'target':tv, 'constraints':cv,
                             'edit':ed, 'feasible':ok, 'seq': seq})
            if ok and (best['t'] is None or tv>best['t']):
                best['t']=tv; best['edit']=ed; best['seq']=seq
        logits=(onehot(args.wt,alph,device)*4.0).clone().requires_grad_(True)
        lambdas=[torch.zeros((),device=device) for _ in all_c]
        opt=torch.optim.Adam([logits],lr=args.lr); K=args.K
        rho_ctl=AdaptiveRho(rho0=args.rho, gamma=args.rho_gamma, rho_max=args.rho_max)
        if args.gate=='drift':
            from grace_v2 import DriftGate
            gate=DriftGate(thresh=args.drift_thresh)
        else:
            gate=AlignmentGate(thresh=args.align_thresh)
        rho=args.rho
        # ---- pre-flight alignment probe: decide the ALM mode once, before any
        # step is taken, so neither error mode (dual damage / accumulated drift)
        # is incurred while deciding.
        _cos=[]
        _rng_cpu=torch.get_rng_state()
        _rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        for _ in range(8):
            gsm=-torch.log(-torch.log(torch.rand_like(logits)+1e-9)+1e-9)
            Pp=F.softmax((logits+0.5*gsm)/1.0,dim=0); hp=F.one_hot(Pp.argmax(0),A).T.float()
            xp=(hp+(Pp-Pp.detach())).unsqueeze(0)
            tv_=(t_or(xp).mean())*sgn
            gt_=flatten_gradients(torch.autograd.grad(tv_,[logits],retain_graph=True,allow_unused=True),[logits])
            for o in all_c:
                cv_=o(xp).mean()
                gc_=flatten_gradients(torch.autograd.grad(cv_,[logits],retain_graph=True,allow_unused=True),[logits])
                den=(gt_.norm()*gc_.norm()).clamp(min=1e-12)
                _cos.append(float((gt_@gc_)/den))
        # the probe must not perturb the sample stream. Gumbel noise is drawn on the
        # logits' device, so the CUDA generator has to be restored too -- restoring
        # only the CPU state leaves the trajectory different from the baseline's.
        torch.set_rng_state(_rng_cpu)
        if _rng_cuda is not None: torch.cuda.set_rng_state_all(_rng_cuda)
        if args.gate=='drift':
            _rc=torch.get_rng_state()
            _rg=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            _df=projection_drift_probe(logits,t_or,all_c,c0_all,eps_all,tol_v,sgn,A,K,args.lr,
                                       args.probe_steps,device,args.active_set_tol)
            torch.set_rng_state(_rc)
            if _rg is not None: torch.cuda.set_rng_state_all(_rg)
            gate.decide(_df)
        else:
            gate.decide(_cos)
        for step in range(args.steps):
            tau=max(0.3,1.0*(0.3)**(step/max(1,args.steps)))
            xs=[]
            for _ in range(K):
                if args.det_relax:
                    xs.append(F.softmax(logits/tau,dim=0)); break
                g=-torch.log(-torch.log(torch.rand_like(logits)+1e-9)+1e-9)
                Pk=F.softmax((logits+0.5*g)/tau,dim=0); hard=F.one_hot(Pk.argmax(0),A).T.float()
                xs.append(hard+(Pk-Pk.detach()))
            x=torch.stack(xs)
            tval=(t_or(x).mean())*sgn
            cvals=[o(x).mean() for o in all_c]
            nev['n']+=K*(1+len(all_c))
            log_soft(step, cvals, tval)
            g_t=flatten_gradients(torch.autograd.grad(tval,[logits],retain_graph=True,allow_unused=True),[logits])
            g_cs=[]
            for i,cv in enumerate(cvals):
                if i in no_proj: continue
                if abs(float(cv.item())-c0_all[i])>=(tol_v[i] if i<len(tol_v) else args.active_set_tol)*eps_all[i]:
                    g_cs.append(flatten_gradients(torch.autograd.grad(cv,[logits],retain_graph=True,allow_unused=True),[logits]))
            d,_gi=compute_grace_direction(g_t,g_cs,1e-6) if g_cs else (g_t,None)
            # Alignment gate: cos(g_target, g_constraint) decides whether the
            # reactive stage is needed. compute_grace_direction already returns
            # these as conflict_scores, so this is free. An empty active set means
            # every constraint has slack, in which case there is nothing to restore.
            gate.observe(_gi["conflict_scores"] if _gi else [])
            viols=[torch.clamp(torch.abs(cvals[i]-c0_all[i])-eps_all[i],min=0) for i in range(len(all_c))]
            rho=rho_ctl.update(float(sum(float(v.item()) for v in viols))) if args.adaptive_rho else args.rho
            if gate.on:
                pen=torch.zeros((),device=device)
                for i,v in enumerate(viols): pen=pen+lambdas[i]*v+(rho/2)*v**2
                g_alm=flatten_gradients(torch.autograd.grad(pen,[logits],retain_graph=False,allow_unused=True),[logits]) if pen.requires_grad else torch.zeros_like(d)
            else:
                g_alm=torch.zeros_like(d)
            opt.zero_grad(); logits.grad=-(d-g_alm).view_as(logits); opt.step()
            with torch.no_grad():
                for i in range(len(all_c)):
                    lambdas[i]=torch.clamp(lambdas[i]+args.dual_lr*(abs(cvals[i].item()-c0_all[i])-eps_all[i]),0,100)
            if step%3==0: consider_v2(decode(logits.detach(),alph), step)
        consider_v2(decode(logits.detach(),alph), args.steps)
        _v2_info={**gate.summary(),'final_rho':rho_ctl.rho,'n_rho_increase':rho_ctl.n_increase,
                  'n_extra_constraints':len(extra)}
    elif args.mode in ('soft_penalty','unconstrained'):
        # scalarized: max target - lam*sum relu(|c-c0|-eps); NO projection (isolates projection value)
        # 'unconstrained' (FastSeqProp) is exactly this objective with lam=0, i.e. the
        # target gradient alone with no constraint handling of any kind.
        logits=(onehot(args.wt,alph,device)*4.0).clone().requires_grad_(True)
        opt=(torch.optim.SGD([logits],lr=args.lr) if args.plain_sgd
             else torch.optim.Adam([logits],lr=args.lr))
        lam=0.0 if args.mode=='unconstrained' else args.lam
        K_=args.K
        for step in range(args.steps):
            tau=max(0.3,1.0*(0.3)**(step/max(1,args.steps)))
            xs=[]
            for _ in range(args.K):
                if args.det_relax:
                    xs.append(F.softmax(logits/tau,dim=0)); break
                g=-torch.log(-torch.log(torch.rand_like(logits)+1e-9)+1e-9)
                Pk=F.softmax((logits+0.5*g)/tau,dim=0); hard=F.one_hot(Pk.argmax(0),A).T.float()
                xs.append(hard+(Pk-Pk.detach()))
            x=torch.stack(xs)
            tval=(t_or(x).mean())*sgn
            # evaluate the constraints once and reuse, so the soft trace records
            # exactly the values that enter the penalty
            cvals=[o(x).mean() for o in c_ors]
            nev['n']+=K_*(1+len(c_ors))
            log_soft(step, cvals, tval)
            g_all=[flatten_gradients(torch.autograd.grad(cv,[logits],retain_graph=True,allow_unused=True),[logits])
                   for cv in cvals] if args.log_drift else None
            obj=tval
            for i in range(len(c_ors)):
                obj=obj-lam*torch.clamp(torch.abs(cvals[i]-c0[i])-eps_v[i],min=0)
            if args.log_drift:
                g_t_only=flatten_gradients(torch.autograd.grad(
                    tval,[logits],retain_graph=True,allow_unused=True),[logits])
                log_drift(step, cvals, tval, g_all, g_t_only)
            opt.zero_grad(); (-obj).backward(); opt.step()
            if step%3==0: consider(decode(logits.detach(),alph), step)
        consider(decode(logits.detach(),alph), args.steps)
    elif args.mode in ('grace_only','lagrangian_only','basic_lagrangian','pcgrad'):
        # ablations / gradient-surgery baseline, same STE+Gumbel+K frontend as grace_lagrangian
        logits=(onehot(args.wt,alph,device)*4.0).clone().requires_grad_(True)
        lambdas=[torch.zeros((),device=device) for _ in c_ors]
        opt=(torch.optim.SGD([logits],lr=args.lr) if args.plain_sgd
             else torch.optim.Adam([logits],lr=args.lr)); K=args.K
        # PCGrad traverses tasks in a RANDOM order (Yu et al. 2020 sec 3.1). A dedicated,
        # seeded RNG keeps that reproducible without disturbing the shared sample stream.
        import random as _rnd; _pcg_rng=_rnd.Random(args.seed)
        for step in range(args.steps):
            tau=max(0.3,1.0*(0.3)**(step/max(1,args.steps)))
            xs=[]
            for _ in range(K):
                if args.det_relax:
                    xs.append(F.softmax(logits/tau,dim=0)); break
                g=-torch.log(-torch.log(torch.rand_like(logits)+1e-9)+1e-9)
                Pk=F.softmax((logits+0.5*g)/tau,dim=0); hard=F.one_hot(Pk.argmax(0),A).T.float()
                xs.append(hard+(Pk-Pk.detach()))
            x=torch.stack(xs)
            tval=(t_or(x).mean())*sgn
            cvals=[o(x).mean() for o in c_ors]
            nev['n']+=K*(1+len(c_ors))
            log_soft(step, cvals, tval)
            g_t=flatten_gradients(torch.autograd.grad(tval,[logits],retain_graph=True,allow_unused=True),[logits])
            if args.mode=='grace_only':
                # null-space projection only; NO Lagrangian dual / penalty
                g_all=[flatten_gradients(torch.autograd.grad(cv,[logits],retain_graph=True,allow_unused=True),[logits])
                       for cv in cvals] if args.log_drift else None
                g_cs=[(g_all[i] if g_all is not None else
                       flatten_gradients(torch.autograd.grad(cv,[logits],retain_graph=True,allow_unused=True),[logits]))
                      for i,cv in enumerate(cvals) if abs(float(cv.item())-c0[i])>=tol_v[i]*eps_v[i]]
                d,_=compute_grace_direction(g_t,g_cs,1e-6) if g_cs else (g_t,None)
                log_drift(step, cvals, tval, g_all, d)
                opt.zero_grad(); logits.grad=-d.view_as(logits); opt.step()
            elif args.mode in ('lagrangian_only','basic_lagrangian'):
                # lagrangian_only    : AUGMENTED Lagrangian (lambda*viol + rho/2*viol^2)
                # basic_lagrangian   : CLASSIC Lagrangian (lambda*viol only, no augmentation)
                # Neither projects; both isolate what the projection contributes.
                rho = 0.0 if args.mode=='basic_lagrangian' else args.rho
                pen=torch.zeros((),device=device)
                for i,cv in enumerate(cvals):
                    viol=torch.clamp(torch.abs(cv-c0[i])-eps_v[i],min=0)
                    pen=pen+lambdas[i]*viol+(rho/2)*viol**2
                g_alm=flatten_gradients(torch.autograd.grad(pen,[logits],retain_graph=False,allow_unused=True),[logits]) if pen.requires_grad else torch.zeros_like(g_t)
                opt.zero_grad(); logits.grad=-(g_t-g_alm).view_as(logits); opt.step()
                with torch.no_grad():
                    for i,cv in enumerate(cvals):
                        lambdas[i]=torch.clamp(lambdas[i]+args.dual_lr*(abs(cv.item()-c0[i])-eps_v[i]),0,100)
            else:  # pcgrad: gradient surgery across {maximize target, satisfy each constraint}
                tasks=[g_t.clone()]
                for i,cv in enumerate(cvals):
                    viol=torch.clamp(torch.abs(cv-c0[i])-eps_v[i],min=0)
                    gv=flatten_gradients(torch.autograd.grad(viol,[logits],retain_graph=True,allow_unused=True),[logits]) if viol.requires_grad else torch.zeros_like(g_t)
                    tasks.append(-gv)  # descend the violation
                order=list(range(len(tasks))); _pcg_rng.shuffle(order)
                proj=[]
                for a_i in order:
                    gi=tasks[a_i].clone()
                    for b_i in order:
                        if a_i==b_i: continue
                        gj=tasks[b_i]; dot=float((gi*gj).sum().item()); nj=float((gj*gj).sum().item())
                        if dot<0 and nj>1e-12: gi=gi-(dot/nj)*gj
                    proj.append(gi)
                d=torch.stack(proj).sum(0)
                opt.zero_grad(); logits.grad=-d.view_as(logits); opt.step()
            if step%3==0: consider(decode(logits.detach(),alph), step)
        consider(decode(logits.detach(),alph), args.steps)
    elif args.mode=='simulated_annealing':
        import random,math; random.seed(args.seed); cur=args.wt
        def scf(s):
            x=onehot(s,alph,device).unsqueeze(0)
            with torch.no_grad(): return float(t_or(x).item())*sgn
        cs=scf(cur)
        for step in range(args.budget):
            T=max(0.01,1.0*(1-step/args.budget))
            ss=list(cur); p=random.randrange(L); ss[p]=random.choice([a for a in alph if a!=ss[p]]); cand=''.join(ss)
            consider(cand); sc=scf(cand)
            if sc>cs or random.random()<math.exp((sc-cs)/T): cur=cand; cs=sc
    elif args.mode=='random':
        # pure random search: best-of-budget random sequences (WT + random mutations).
        # Tests whether the feasible high-value set is trivially findable (task saturation).
        import random; random.seed(args.seed)
        for _ in range(args.budget):
            nmut=random.randint(1,L); ss=list(args.wt)
            for p in random.sample(range(L),nmut): ss[p]=random.choice(alph)
            consider(''.join(ss))
    elif args.mode=='botorch':
        # Bayesian optimization: GP surrogate over flat one-hot, LogEI acq over a discrete
        # candidate pool (mutants of best + random). Constrained via penalized objective.
        import warnings; warnings.filterwarnings('ignore'); import random
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition.analytic import LogExpectedImprovement
        from botorch.models.transforms import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
        def flat(seq): return onehot(seq,alph,device).reshape(-1)
        def pen_obj(seq):
            x=onehot(seq,alph,device).unsqueeze(0)
            with torch.no_grad():
                tv=float(t_or(x).item())*sgn; cv=[float(o(x).item()) for o in c_ors]
            viol=sum(max(0.0,abs(cv[i]-c0[i])-args.eps) for i in range(len(c_ors)))
            return tv-10.0*viol
        def randmut(base,k=None):
            ss=list(base); k=k or random.randint(1,max(1,L//4))
            for p in random.sample(range(L),min(k,L)): ss[p]=random.choice(alph)
            return ''.join(ss)
        N0=20; total=max(args.budget,150); n_iter=min(total-N0,180)
        seqs=[args.wt]+[randmut(args.wt) for _ in range(N0-1)]
        for s in seqs: consider(s)
        X=torch.stack([flat(s) for s in seqs]).double()
        Y=torch.tensor([pen_obj(s) for s in seqs],device=device).double().unsqueeze(-1)
        acqf=None
        for it in range(n_iter):
            try:
                if it%4==0 or acqf is None:   # refit GP periodically (860-D fit is the bottleneck)
                    gp=SingleTaskGP(X,Y,outcome_transform=Standardize(1))
                    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood,gp))
                else:
                    gp.set_train_data(X, Y.squeeze(-1), strict=False)
                acqf=LogExpectedImprovement(gp,best_f=Y.max())
                topi=Y.squeeze(-1).argsort(descending=True)[:5].tolist()
                pool=set()
                for i in topi:
                    for _ in range(60): pool.add(randmut(seqs[i]))
                for _ in range(60): pool.add(randmut(args.wt,random.randint(1,L)))
                pool=list(pool)
                C=torch.stack([flat(s) for s in pool]).double().unsqueeze(1)
                with torch.no_grad(): a=acqf(C)
                pick=pool[int(a.argmax())]
            except Exception:
                pick=randmut(args.wt)
            consider(pick); seqs.append(pick)
            X=torch.cat([X,flat(pick).double().unsqueeze(0)])
            Y=torch.cat([Y,torch.tensor([[pen_obj(pick)]],device=device).double()])
    elif args.mode=='cbo':
        # CONSTRAINED Bayesian optimization. Unlike `botorch` above (single-output GP
        # on a hand-penalised scalar tv - 10*viol, i.e. plain LogEI), this fits a
        # MULTI-OUTPUT GP to [target, c_1..c_n] and uses qLogExpectedImprovement with
        # one OUTCOME CONSTRAINT per oracle, in the same two-sided band form as the
        # benchmark: |c_i - c0_i| - eps_i <= 0. Mirrors optimize_splice.py's
        # optimize_botorch_baseline, which is the genuine CBO implementation.
        import warnings; warnings.filterwarnings('ignore'); import random
        from botorch.acquisition.logei import qLogExpectedImprovement
        from botorch.acquisition.objective import GenericMCObjective
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.models.transforms import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

        def flat(seq): return onehot(seq,alph,device).reshape(-1)

        def outcomes(seq):
            """[target*sign, c_1..c_n] -- the GP models each separately."""
            x=onehot(seq,alph,device).unsqueeze(0)
            with torch.no_grad():
                tv=float(t_or(x).item())*sgn
                cv=[float(o(x).item()) for o in c_ors]
            return [tv]+cv

        def randmut(base,k=None):
            ss=list(base); k=k or random.randint(1,max(1,L//4))
            for pz in random.sample(range(L),min(k,L)): ss[pz]=random.choice(alph)
            return ''.join(ss)

        def feas_row(row):
            return all(abs(row[1+i]-c0[i])<=eps_v[i] for i in range(len(c_ors)))

        N0=20; total=max(args.budget,150); n_iter=min(total-N0,180)
        seqs=[args.wt]+[randmut(args.wt) for _ in range(N0-1)]
        for sq in seqs: consider(sq)
        rowsY=[outcomes(sq) for sq in seqs]
        X=torch.stack([flat(sq) for sq in seqs]).double()
        Y=torch.tensor(rowsY,device=device).double()
        acqf=None
        for it in range(n_iter):
            try:
                if it%4==0 or acqf is None:
                    gp=SingleTaskGP(X,Y,outcome_transform=Standardize(Y.shape[-1]))
                    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood,gp))
                else:
                    gp.set_train_data(X,Y,strict=False)
                feas=[r[0] for r in rowsY if feas_row(r)]
                best_f=max(feas) if feas else float(Y[:,0].max())
                acqf=qLogExpectedImprovement(
                    model=gp, best_f=best_f,
                    objective=GenericMCObjective(lambda samples, X=None: samples[...,0]),
                    constraints=[
                        (lambda samples, idx=1+i, cc=c0[i], ee=eps_v[i]:
                         (samples[...,idx]-cc).abs()-ee)
                        for i in range(len(c_ors))],
                )
                order=sorted(range(len(rowsY)),
                             key=lambda j:(feas_row(rowsY[j]), rowsY[j][0]), reverse=True)[:5]
                pool=set()
                for j in order:
                    for _ in range(60): pool.add(randmut(seqs[j]))
                for _ in range(60): pool.add(randmut(args.wt,random.randint(1,L)))
                pool=list(pool)
                C=torch.stack([flat(sq) for sq in pool]).double().unsqueeze(1)
                with torch.no_grad(): a=acqf(C)
                pick=pool[int(a.argmax())]
            except Exception:
                pick=randmut(args.wt)
            consider(pick); seqs.append(pick)
            r=outcomes(pick); rowsY.append(r)
            X=torch.cat([X,flat(pick).double().unsqueeze(0)])
            Y=torch.cat([Y,torch.tensor([r],device=device).double()])
    elif args.mode=='adalead':
        pop=[args.wt]; import random; random.seed(args.seed)
        for _ in range(args.budget//64):
            newpop=[]
            for s in pop[:16]:
                for _ in range(8):
                    ss=list(s)
                    for _ in range(random.randint(1,3)):
                        p=random.randrange(L); ss[p]=random.choice([a for a in alph if a!=ss[p]])
                    newpop.append(''.join(ss))
            for s in newpop: consider(s)
            # keep best-target feasible-ish
            def sc(s):
                x=onehot(s,alph,device).unsqueeze(0)
                with torch.no_grad(): return float(t_or(x).item())*sgn
            pop=sorted(set(newpop+pop),key=sc,reverse=True)[:32]
    elif args.mode=='directed_evolution':
        import random; random.seed(args.seed); cur=args.wt
        def sc(s):
            x=onehot(s,alph,device).unsqueeze(0)
            nev['n']+=1
            with torch.no_grad(): return float(t_or(x).item())*sgn
        _stop=False
        for _ in range(args.steps//10):
            best_local=cur; best_sc=sc(cur)
            # With an evaluation cap the last round is cut short. Scanning positions
            # in index order would then spend the whole budget on the N-terminal
            # part of the sequence, so --shuffle_scan visits them in random order.
            order=list(range(L))
            if args.shuffle_scan: random.shuffle(order)
            for p in order:
                for a in alph:
                    if a==cur[p]: continue
                    cand=cur[:p]+a+cur[p+1:]; consider(cand); s=sc(cand)
                    if s>best_sc: best_sc=s; best_local=cand
                    if args.max_evals and nev['n']>=args.max_evals: _stop=True; break
                if _stop: break
            if _stop or best_local==cur: break
            cur=best_local
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    _extra = globals().get('_v2_info') or locals().get('_v2_info') or {}
    json.dump({'mode':args.mode,'wt_target':t0*sgn,'c0':c0,'eps':args.eps,
               'n_evals':nev['n'],**_extra,
               'best_feasible_target':best['t'],'best_feasible_edit':best['edit'],
               'best_feasible_seq':best['seq'],
               **({'drift_history':drift_hist} if args.log_drift else {}),
               **({'history':hist,'soft_history':soft_hist,
                   'constraint_names':args.constraint_oracles,
                   'c0':c0,'eps_list':eps_v} if args.log_history else {})},
              open(args.out,'w'),indent=2)
    print(f"[{args.mode}] best_feasible_target={best['t']} edit={best['edit']} (WT_target={t0*sgn:.3f})")

def build_parser():
    """VENDORED CHANGE (the only one): the argument parser that lived in
    `__main__` is exposed so the task-level CLI in scripts/optimize.py can
    reuse these exact defaults instead of restating them. The algorithm and
    every default value are untouched -- see docs/VENDORED.md."""
    p=argparse.ArgumentParser()
    p.add_argument('--target_oracle',required=True); p.add_argument('--constraint_oracles',nargs='*',default=[])
    p.add_argument('--wt',required=True); p.add_argument('--alphabet',required=True)
    p.add_argument('--target_sign',type=float,default=1.0); p.add_argument('--eps',type=float,default=0.1); p.add_argument('--eps_list',default=None); p.add_argument('--tol_list',default=None)
    p.add_argument('--det_relax',action='store_true',
                   help='deterministic relaxation for the drift diagnostic: the '
                        'oracles see softmax(Theta/tau) with no Gumbel noise. The '
                        'Taylor split in Sec. 3.1 is a statement about the '
                        'deterministic functions, and differencing a K-sample '
                        'stochastic average measures the sampler instead.')
    p.add_argument('--plain_sgd',action='store_true',
                   help='plain gradient ascent instead of Adam. Only for the drift '
                        'diagnostic: it makes the realized step exactly eta*d, so '
                        'the Taylor terms separate cleanly. Adam preconditions the '
                        'direction and would mix them.')
    p.add_argument('--log_drift',action='store_true',
                   help='record the first-order / curvature split of the '
                        'constraint change at every step (motivating experiment)')
    p.add_argument('--shuffle_scan',action='store_true',
                   help='directed evolution: visit positions in random order within a '
                        'round, so a capped final round samples the whole sequence')
    p.add_argument('--max_evals',type=int,default=None,
                   help='stop once this many oracle evaluations have been spent. '
                        'The budget every method is matched on; n_evals in the '
                        'output records what was actually used.')
    p.add_argument('--no_project',default=None,
                   help='comma-separated constraint indices kept OUT of the projection '
                        'active set (still enforced by the ALM). For budget-style '
                        'constraints such as builtin:hamming, whose gradient is simply '
                        '"return to the wild type" and so deletes most of the target '
                        'direction when projected out.')
    p.add_argument('--mode',default='grace_lagrangian'); p.add_argument('--K',type=int,default=8); p.add_argument('--lam',type=float,default=10.0)
    p.add_argument('--steps',type=int,default=800)
    p.add_argument('--log_history',action='store_true',
                   help="Also write every discrete evaluation (step, target, "
                        "constraints, edit, feasible) to the output JSON.")
    p.add_argument('--lr',type=float,default=0.1); p.add_argument('--rho',type=float,default=10.0)
    p.add_argument('--dual_lr',type=float,default=0.05); p.add_argument('--budget',type=int,default=2000)
    p.add_argument('--active_set_tol',type=float,default=0.3)  # slack-aware: project only if |c-c0|>=tol*eps (0 = always project all)
    p.add_argument('--prior_data',default=None); p.add_argument('--prior_epochs',type=int,default=12)
    p.add_argument('--plaus_eps',type=float,default=0.15); p.add_argument('--rho_max',type=float,default=100.0)
    p.add_argument('--curv_thresh',type=float,default=1e-2); p.add_argument('--align_thresh',type=float,default=0.4); p.add_argument('--gate',default='align',choices=['align','drift']); p.add_argument('--drift_thresh',type=float,default=0.25); p.add_argument('--probe_steps',type=int,default=40); p.add_argument('--adaptive_rho',action='store_true'); p.add_argument('--plaus_q',type=float,default=0.5); p.add_argument('--rho_gamma',type=float,default=2.0)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--out',required=True)
    return p


if __name__ == '__main__':
    run(build_parser().parse_args())
