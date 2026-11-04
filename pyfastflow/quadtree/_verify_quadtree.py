"""Verify the quadtree against pure-Python and numpy references; raises on any mismatch.

Run with ``python -m pyfastflow.quadtree._verify_quadtree``.

Checked: leaf sets against a recursive build; neighbour slots against a scan of
the leaf-id raster; every device helper against numpy; link symmetry through
``reverse``; ``adapt`` against a fresh topology build with field conservation;
every transfer rule, including a lake at rest; ``regrid``; capacity overflow; masks; raster transfer; static placeholders;
``SFDFlowProgram(tree=...)`` drained area on all accumulation and depression
choices, across updates, with the tree surviving the Program.

Author: B.G (11/2026)
"""

import math

import cupy as cp
import numpy as np

from pyfastflow.core import Backend, KernelBuilder
from pyfastflow.flow import SFDFlowProgram
from pyfastflow.quadtree import DynamicQuadtree, Quadtree, transfer as tf

be = Backend.from_name("cupy")
pool = be.PoolCls()
def own_raster(t):
    ny,nx=t.ny,t.nx; return t.to_raster(cp.arange(t.capacity,dtype=cp.int32))
def ref_build(lm,L,per):
    """pure-python: split on map, then split until balanced"""
    ny,nx=lm.shape; S=1<<L; leaves=set()
    def rec(l,x,y):
        s=1<<l
        if x>=nx or y>=ny: return
        over=x+s>nx or y+s>ny
        if l>0 and (over or lm[y:min(y+s,ny),x:min(x+s,nx)].min()<l):
            for a in (0,s//2):
                for b in (0,s//2): rec(l-1,x+a,y+b)
        else: leaves.add((l,x,y))
    for y in range(0,ny,S):
        for x in range(0,nx,S): rec(L,x,y)
    while True:
        o=-np.ones((ny,nx),int); ls={}
        for v in leaves:
            l,x,y=v; o[y:y+(1<<l),x:x+(1<<l)]=l
        bad=set()
        for v in leaves:
            l,x,y=v;s=1<<l
            # all pixels in the ring
            rx=np.arange(x-1,x+s+1); ry=np.arange(y-1,y+s+1)
            pts=[(y-1,a) for a in rx]+[(y+s,a) for a in rx]+[(b,x-1) for b in ry]+[(b,x+s) for b in ry]
            for py,px in pts:
                if per[0]: px%=nx
                if per[1]: py%=ny
                if 0<=px<nx and 0<=py<ny and o[py,px]>l+1: bad.add((py,px,o[py,px]))
        if not bad: return leaves
        for py,px,lv in bad:
            s=1<<lv; x=px//s*s; y=py//s*s
            if (lv,x,y) in leaves:
                leaves.remove((lv,x,y))
                for a in (0,s//2):
                    for b in (0,s//2): leaves.add((lv-1,x+a,y+b))
def ref_table(t,per,d8):
    """independent slot table from the id raster"""
    n=t.n; O=own_raster(t); ny,nx=t.ny,t.nx; lv=cp.asnumpy(t.level);X=cp.asnumpy(t.x);Y=cp.asnumpy(t.y)
    first=(1,4,6,9) if d8 else (0,2,4,6); ns=12 if d8 else 8
    tb=-np.ones((n,ns),int)
    def at(py,px):
        if per[0]: px%=nx
        if per[1]: py%=ny
        if 0<=px<nx and 0<=py<ny: return O[py,px]
        return -1
    for i in range(n):
        l,x,y=int(lv[i]),int(X[i]),int(Y[i]); s=1<<l
        for f,pix in enumerate(([(y-1,x+a) for a in range(s)],[(y+a,x-1) for a in range(s)],[(y+a,x+s) for a in range(s)],[(y+s,x+a) for a in range(s)])):
            ow=[at(*p) for p in pix]
            if ow[0]<0: continue
            u=list(dict.fromkeys(ow)); low=f<2
            if len(u)==2:
                a,b=(0,1) if low else (1,0); tb[i,first[f]+a]=u[0]; tb[i,first[f]+b]=u[1]
            else:
                j=u[0]
                if lv[j]==l: tb[i,first[f]+(0 if low else 1)]=j
                else:
                    along=x if f in(0,3) else y
                    p=(along>>l)&1; tb[i,first[f]+(p if low else 1-p)]=j
        if d8:
            for slot,cx,cy,ax,ay in ((0,x-1,y-1,x-1,y),(3,x+s,y-1,x+s,y),(8,x-1,y+s,x-1,y+s-1),(11,x+s,y+s,x+s,y+s-1)):
                j=at(cy,cx)
                if j<0: continue
                bx=x if slot in(0,8) else x+s-1
                by=y-1 if slot in (0,3) else y+s
                if j!=at(ay,ax) and j!=at(by,bx): tb[i,slot]=j
    return tb
def balance_ok(t,per):
    O=own_raster(t);lv=cp.asnumpy(t.level);m=lv[O]
    for ax,p in ((0,per[1]),(1,per[0])):
        a=np.abs(m-np.roll(m,1,ax)) if p else np.abs(np.diff(m,axis=ax))
        if a.max(initial=0)>1: return False
    dg=[(1,1),(1,-1)]
    for dy,dx in dg:
        r=np.roll(np.roll(m,dy,0),dx,1); k=np.ones_like(m,bool)
        if not per[1]: k[0 if dy>0 else -1,:]=False
        if not per[0]: k[:,0 if dx>0 else -1]=False
        if np.abs(m-r)[k].max(initial=0)>1: return False
    return True
def csr_table(t):
    ptr=cp.asnumpy(t._array("PTR")); adj=cp.asnumpy(t._array("ADJ")); sl=cp.asnumpy(t._array("SLOTS")); ns=t._dom.n_slots
    tb=-np.ones((t.n,ns),int)
    for i in range(t.n):
        p=ptr[i]
        for k in range(ns):
            if int(sl[i])>>k&1: tb[i,k]=adj[p]; p+=1
        assert p==ptr[i+1]
    return tb
K='''extern "C" __global__ void probe(int* nb,float* di,float* wi,float* dxo,float* ar,float* cxo,float* cyo,int* lv,int* co,int* ac,int* nd,int* rv,float* db,int* jj,float* dd,int* ncnt,int* nsum){
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=$ctx.grid.n_leaves()$) return;
 int ns=$ctx.grid.N_NEIGHBOURS.get(0)$;
 for(int k=0;k<ns;k++){ int j=$ctx.grid.neighbour(i,k)$; nb[i*ns+k]=j; di[i*ns+k]=$ctx.grid.dist(i,k)$; wi[i*ns+k]=$ctx.grid.width(i,k)$; int rr=$ctx.grid.reverse(k)$; rv[k]=rr;
   int r0=$ctx.grid.neighbour_raw(i,k)$; db[i*ns+k]=$ctx.grid.dist_between_nodes(i,r0)$; int a; float b; $ctx.grid.neighbour_and_distance(i,k,&a,&b)$; jj[i*ns+k]=a; dd[i*ns+k]=b; }
 dxo[i]=$ctx.grid.dx(i)$; ar[i]=$ctx.grid.area(i)$; cxo[i]=$ctx.grid.cx(i)$; cyo[i]=$ctx.grid.cy(i)$; lv[i]=$ctx.grid.level(i)$; co[i]=$ctx.grid.can_out(i)$; ac[i]=$ctx.grid.is_active(i)$; nd[i]=$ctx.grid.nodata(i)$;
 int c=0,s=0; for(int p=$ctx.grid.nbr_begin(i)$;p<$ctx.grid.nbr_end(i)$;p++){ c++; s+=$ctx.grid.nbr_at(p)$; } ncnt[i]=c; nsum[i]=s;
}'''
def probe(t,per):
    n=t.capacity;ns=t._dom.n_slots
    k=KernelBuilder(K,domain=n).compose("grid",t.group).freeze().build()
    k.bind_leaf(t.parameters,prefix=("grid",))
    arrs={}
    for name,dt,sz in (("nb","i32",n*ns),("di","f32",n*ns),("wi","f32",n*ns),("dxo","f32",n),("ar","f32",n),("cxo","f32",n),("cyo","f32",n),("lv","i32",n),("co","i32",n),("ac","i32",n),("nd","i32",n),("rv","i32",ns),("db","f32",n*ns),("jj","i32",n*ns),("dd","f32",n*ns),("ncnt","i32",n),("nsum","i32",n)):
        h=pool.get_data(dt,(sz,)); arrs[name]=h; k.bind(name,h)
    c=k.compile(be); c(); out={a:h.to_numpy() for a,h in arrs.items()}; c.close() if hasattr(c,"close") else None; k.close()
    for h in arrs.values(): pool.release_data(h)
    return out
def check_helpers(t,per,tag):
    n=t.n;ns=t._dom.n_slots;o=probe(t,per); tb=csr_table(t); lv=cp.asnumpy(t.level);X=cp.asnumpy(t.x);Y=cp.asnumpy(t.y);d0=t.dx0;nx,ny=t.nx,t.ny
    assert (o["rv"]==ns-1-np.arange(ns)).all(),"reverse"
    nb=o["nb"].reshape(-1,ns)[:n]
    fl=cp.asnumpy(t._array("FLAGS"))[:n]
    exp=tb.copy()
    if t._has_nodata:
        nod=(fl&2)!=0; exp[(exp>=0)&(nod[exp]|nod[np.arange(n)[:,None]])]=-1
    assert (nb==exp).all(),tag+" neighbour"
    # links symmetric
    r,kk=np.nonzero(tb>=0); assert (tb[tb[r,kk],ns-1-kk]==r).all(),tag+" reverse link"
    s=(1<<lv).astype(float); cx=(X+s/2)*d0; cy=(Y+s/2)*d0
    assert np.allclose(o["dxo"][:n],s*d0) and np.allclose(o["ar"][:n],(s*d0)**2),tag+" dx"
    assert np.allclose(o["cxo"][:n],cx) and np.allclose(o["cyo"][:n],cy),tag+" c"
    assert (o["lv"][:n]==lv).all()
    j=tb[r,kk]; dx_=np.abs(cx[r]-cx[j]);dy_=np.abs(cy[r]-cy[j])
    if per[0]: dx_=np.minimum(dx_,nx*d0-dx_)
    if per[1]: dy_=np.minimum(dy_,ny*d0-dy_)
    D=np.hypot(dx_,dy_); di=o["di"].reshape(-1,ns)[r,kk]; assert np.allclose(di,D,rtol=1e-5),tag+" dist %g"%np.abs(di-D).max()
    db=o["db"].reshape(-1,ns)[r,kk]; assert np.allclose(db,D,rtol=1e-5),tag+" dist_between"
    jj=o["jj"].reshape(-1,ns)[:n]; assert (jj==nb).all(); dd=o["dd"].reshape(-1,ns)
    m=nb[r,kk]>=0; assert np.allclose(dd[r,kk][m],D[m],rtol=1e-5)
    isc=np.isin(kk,(0,3,8,11)) if ns==12 else np.zeros_like(kk,bool)
    w=np.minimum(s[r],s[j])*d0*np.where(isc,math.sqrt(2),1); assert np.allclose(o["wi"].reshape(-1,ns)[r,kk],w),tag+" width"
    # tail
    assert (o["ncnt"][:n]==(tb>=0).sum(1)).all() and (o["nsum"][:n]==np.where(tb>=0,tb,0).sum(1)).all(),tag+" nbr loop"
    assert (o["co"][:n]==(fl&1)).all() and (o["ac"][:n]==1-((fl>>1)&1)).all() and (o["nd"][:n]==((fl>>1)&1)).all()
rng=np.random.default_rng(0)
def lmap(ny,nx,L):
    m=np.full((ny,nx),L,np.int32); 
    for _ in range(4):
        y,x=rng.integers(ny),rng.integers(nx); m[y,x]=0
    m[rng.integers(ny)//2:,rng.integers(nx)//2]=rng.integers(0,2)
    return m
cases=[(64,64,3,"normal"),(57,43,3,"normal"),(40,96,4,"normal"),(64,64,3,"periodic_EW_NS"),(64,32,3,"periodic_EW"),(32,64,4,"periodic_NS"),(16,16,0,"normal"),(1,5,2,"normal")]
for ny,nx,L,bd in cases:
    per=(bd in("periodic_EW","periodic_EW_NS"),bd in("periodic_NS","periodic_EW_NS"))
    lm=lmap(ny,nx,L)
    for topo in ("D4","D8"):
        t=Quadtree(be,pool,lm,0.5,L,topology=topo,boundary=bd); assert t.parameters["N"].mode=="const"
        ref=ref_build(lm,L,per); got=set(zip(*(cp.asnumpy(a).tolist() for a in (t.level,t.x,t.y))))
        assert got==ref,("leaves",ny,nx,L,bd,len(got),len(ref),len(got-ref),len(ref-got))
        assert balance_ok(t,per),"bal"
        assert (csr_table(t)==ref_table(t,per,topo=="D8")).all(),("table",ny,nx,L,bd,topo)
        area=(4**cp.asnumpy(t.level)).sum(); assert area==nx*ny
        check_helpers(t,per,f"{ny}x{nx} L{L} {bd} {topo}")
        t.close()
    print("static ok",ny,nx,L,bd,"leaves",len(ref),"of",nx*ny,flush=True)
# masks
lm=lmap(64,64,3); nd=np.zeros((64,64),np.uint8); nd[:24,:24]=1; ol=np.zeros((64,64),np.uint8); ol[63,17]=1
t=Quadtree(be,pool,lm,1.0,3,topology="D8",nodata_mask=nd,outlet_mask=ol); fl=cp.asnumpy(t._array("FLAGS"))[:t.n]
lv=cp.asnumpy(t.level);X=cp.asnumpy(t.x);Y=cp.asnumpy(t.y)
for i in range(t.n):
    s=1<<lv[i]; a=nd[Y[i]:Y[i]+s,X[i]:X[i]+s].all(); b=ol[Y[i]:Y[i]+s,X[i]:X[i]+s].any(); assert bool(fl[i]&2)==bool(a) and bool(fl[i]&1)==bool(b)
check_helpers(t,(False,False),"mask"); t.close(); print("masks ok",flush=True)
# raster round trip
a=rng.random((57,43),dtype=np.float32); lm=lmap(57,43,3); t=Quadtree(be,pool,lm,1.0,3)
for mode,fn in (("mean",np.mean),("sum",np.sum),("min",np.min),("max",np.max)):
    f=cp.asnumpy(t.from_raster(a,mode=mode)); lv=cp.asnumpy(t.level);X=cp.asnumpy(t.x);Y=cp.asnumpy(t.y)
    for i in range(t.n):
        s=1<<lv[i]; assert abs(f[i]-fn(a[Y[i]:Y[i]+s,X[i]:X[i]+s]))<1e-4
rr=t.to_raster(t.from_raster(a)); assert abs(rr.mean()-a.mean())<1e-4
try: t.adapt(np.zeros(t.n)); raise SystemExit("static adapt ran")
except NotImplementedError: pass
try: t.regrid(lm); raise SystemExit("static regrid ran")
except NotImplementedError: pass
t.close(); print("raster/static placeholders ok",flush=True)
# dynamic
for ny,nx,L,bd in [(64,64,3,"normal"),(57,43,3,"normal"),(64,64,3,"periodic_EW_NS"),(128,128,5,"normal")]:
    per=(bd in("periodic_EW","periodic_EW_NS"),bd in("periodic_NS","periodic_EW_NS"))
    for topo in ("D4","D8"):
        lm=lmap(ny,nx,L); nd=np.zeros((ny,nx),np.uint8); nd[:9,:9]=1; ol=np.zeros((ny,nx),np.uint8); ol[-1,5]=1
        t=DynamicQuadtree(be,pool,lm,2.0,L,topology=topo,boundary=bd,capacity=nx*ny,nodata_mask=nd,outlet_mask=ol if topo=="D8" else None)
        assert t.parameters["N"].mode=="scalar"
        u=cp.zeros(t.capacity,cp.float32); v=cp.zeros(t.capacity,cp.float32)
        u[:t.n]=cp.asarray(rng.random(t.n,dtype=np.float32)); v[:t.n]=cp.asarray(rng.random(t.n,dtype=np.float32))
        for it in range(25):
            lv0=cp.asnumpy(t.level);X0=cp.asnumpy(t.x);Y0=cp.asnumpy(t.y);u0=cp.asnumpy(u)[:t.n];v0=cp.asnumpy(v)[:t.n];k0=dict(zip(zip(lv0,X0,Y0),u0))
            mass_u=(u0.astype(np.float64)*4.0**lv0).sum(); mass_v=v0.astype(np.float64).sum()
            p=rng.choice([.01,.1,.5]); f=np.zeros(t.n,np.int32); f[rng.random(t.n)<p]=1; f[rng.random(t.n)<(.2 if it%2 else .9)]=-1
            if it%5==0: f[rng.integers(t.n)]=1
            t.adapt(f,[(u,"mean"),(v,"sum")])
            lv=cp.asnumpy(t.level);X=cp.asnumpy(t.x);Y=cp.asnumpy(t.y)
            assert balance_ok(t,per),"bal"
            assert (4**lv).sum()==(4**lv0).sum(),"cover"
            # topology == fresh full build and independent reference
            pf,af,sf=t._full_topology(t.level,t.x,t.y,t._key)
            assert bool(cp.all(pf==t._array("PTR")[:t.n+1])) and bool(cp.all(af==t._array("ADJ")[:t.n_links])) and bool(cp.all(sf==t._array("SLOTS")[:t.n])),"csr vs fresh"
            assert bool(cp.all(t._cell_flags(t.level,t.x,t.y)==t._array("FLAGS")[:t.n])),"flags"
            if it%8==0: assert (csr_table(t)==ref_table(t,per,topo=="D8")).all(),"ref"; check_helpers(t,per,f"dyn {bd} {topo}")
            ku=cp.asnumpy(u)[:t.n];kv=cp.asnumpy(v)[:t.n]
            assert abs((ku.astype(np.float64)*4.0**lv).sum()-mass_u)<=1e-3*max(1,mass_u),"mass mean"
            assert abs(kv.astype(np.float64).sum()-mass_v)<=1e-3*max(1,mass_v),"mass sum"
            cur=set(zip(lv,X,Y)); old=set(k0)
            for key,val in zip(zip(lv,X,Y),ku):
                if key in old: assert val==k0[key],"kept value"
            assert bool(cp.all(u[t.n:]==0)),"tail"
            assert (cp.asnumpy(t._array("FLAGS"))[t.n:]==3).all() and (cp.asnumpy(t._array("SLOTS"))[t.n:]==0).all() and (cp.asnumpy(t._array("PTR"))[t.n:]==t.n_links).all()
            # a leaf flagged to refine must have split
            fi=np.nonzero(f==1)[0]
            for i in fi[:50]:
                if lv0[i]>0: assert (lv0[i],X0[i],Y0[i]) not in cur,"seed not split"
        # regrid
        lm2=lmap(ny,nx,L); mu=(cp.asnumpy(u)[:t.n].astype(np.float64)*4.0**cp.asnumpy(t.level)).sum(); mv=cp.asnumpy(v)[:t.n].astype(np.float64).sum()
        t.regrid(lm2,[(u,"mean"),(v,"sum")]); ref=ref_build(lm2,L,per); assert set(zip(*(cp.asnumpy(a).tolist() for a in (t.level,t.x,t.y))))==ref
        assert abs((cp.asnumpy(u)[:t.n].astype(np.float64)*4.0**cp.asnumpy(t.level)).sum()-mu)<=1e-3*max(1,mu) and abs(cp.asnumpy(v)[:t.n].astype(np.float64).sum()-mv)<=1e-3*max(1,mv)
        check_helpers(t,per,"regrid")
        # coarsen all
        for _ in range(L+2): t.adapt(-np.ones(t.n,np.int32),[(u,"mean")])
        print("dynamic ok",ny,nx,L,bd,topo,"after full coarsen n=",t.n,"levels",sorted(set(cp.asnumpy(t.level).tolist())),flush=True)
        t.close()
lm=np.full((64,64),3,np.int32); t=DynamicQuadtree(be,pool,lm,1.0,3,capacity=70,topology="D4"); u=cp.zeros(70,cp.float32);u[:t.n]=1.0; n0=t.n; pa=t._array("PTR").copy()
try: t.adapt(np.ones(t.n,np.int32),[(u,"mean")]); raise SystemExit("no overflow")
except ValueError as e: print("overflow:",e)
assert t.n==n0 and bool(cp.all(pa==t._array("PTR"))) and float(u[:n0].sum())==n0; t.close()
# raster kernels fail at compile
t=Quadtree(be,pool,np.full((16,16),1,np.int32),1.0,1)
try:
    k=KernelBuilder('extern "C" __global__ void r(float* o){ o[0]=$ctx.grid.DX.get(0)$; }',domain=1).compose("grid",t.group).freeze(); raise SystemExit("DX accepted")
except Exception as e: print("DX rejected:",type(e).__name__,str(e)[:80])
# transfer rules
def random_flags(t,p_ref,p_coar):
    f=np.zeros(t.n,np.int32); r=rng.random(t.n); f[r<p_ref]=1; f[r>1-p_coar]=-1; return f
def leaves(t): return cp.asnumpy(t.level),cp.asnumpy(t.x),cp.asnumpy(t.y)
for ny,nx,L,bd in [(64,64,3,"normal"),(57,43,3,"normal"),(64,64,3,"periodic_EW_NS")]:
    per=(bd=="periodic_EW_NS",bd=="periodic_EW_NS")
    lm=lmap(ny,nx,L); dem=(rng.random((ny,nx))*10+np.add.outer(np.arange(ny),np.arange(nx))*0.5).astype(np.float32); ras=rng.random((ny,nx)).astype(np.float32)
    t=DynamicQuadtree(be,pool,lm,1.0,L,topology="D4",boundary=bd,capacity=nx*ny)
    z=t.from_raster(dem); r=t.from_raster(ras,mode="max")
    a=t.from_raster(ras); b=t.from_raster(ras); c=t.from_raster(ras); hi=t.from_raster(ras); lo=t.from_raster(ras); h=t.from_raster(ras)
    q=cp.zeros(t.adj_capacity,cp.float32); tb=csr_table(t); rows,ks=np.nonzero(tb>=0); pp=np.arange(rows.size)
    # antisymmetric random discharge
    vals=rng.random(rows.size,dtype=np.float32)-0.5; idx={(i,j):p for p,(i,j) in enumerate(zip(rows,tb[rows,ks]))}
    for p,(i,j) in enumerate(zip(rows,tb[rows,ks])):
        if i>j: vals[p]=-vals[idx[(j,i)]]
    q[:rows.size]=cp.asarray(vals)
    rules=[(z,tf.FromRaster(dem)),(r,tf.FromRaster(ras,"max")),(a,"mean"),(b,"sum"),(c,"linear"),(hi,"max"),(lo,"min"),(h,tf.Depth(z)),(q,tf.FaceDischarge())]
    def div(t,qq):
        tb=csr_table(t); out=np.zeros(t.n); pt=cp.asnumpy(t._array("PTR")); Q=cp.asnumpy(qq)
        for i in range(t.n): out[i]=Q[pt[i]:pt[i+1]].astype(np.float64).sum()
        return out
    for it in range(14):
        lv0,X0,Y0=leaves(t); n0=t.n; A0=4.0**lv0
        old={k:cp.asnumpy(v)[:n0].copy() for k,v in dict(a=a,b=b,c=c,hi=hi,lo=lo,h=h,z=z,r=r).items()}; d0=div(t,q)
        k0={key:i for i,key in enumerate(zip(lv0,X0,Y0))}
        t.adapt(random_flags(t,.04,.6 if it%2 else .1),rules)
        lv,X,Y=leaves(t); A=4.0**lv; new={k:cp.asnumpy(v)[:t.n] for k,v in dict(a=a,b=b,c=c,hi=hi,lo=lo,h=h,z=z,r=r).items()}
        tol=lambda m:2e-4*max(1,abs(m))
        for k,w in (("a",1),("c",1),("h",1)):
            m0=(old[k].astype(np.float64)*A0).sum(); m1=(new[k].astype(np.float64)*A).sum(); assert abs(m0-m1)<=tol(m0),("conserve",k,m0,m1)
        assert abs(old["b"].astype(np.float64).sum()-new["b"].astype(np.float64).sum())<=2e-4*max(1,abs(old["b"].sum())),"sum"
        assert (new["h"]>=0).all(),"h negative"
        # z and r exactly match the raster; every leaf
        for i in range(t.n):
            s=1<<lv[i]; sl=(slice(Y[i],Y[i]+s),slice(X[i],X[i]+s))
            assert abs(new["z"][i]-dem[sl].mean())<2e-3,("z",i); assert abs(new["r"][i]-ras[sl].max())<1e-5,("r",i)
            key=(lv[i],X[i],Y[i]); 
            if key in k0:
                for f in "abchlr": 
                    if f=="l": continue
                    assert new[f][i]==old[f][k0[key]],("kept",f)
                assert new["hi"][i]==old["hi"][k0[key]] and new["lo"][i]==old["lo"][k0[key]]
        d1=div(t,q); assert abs(d1.sum())<1e-3,"global div"
        # each changed block: compare total divergence on a coarse common partition
        side_=1<<(L+1)
        def block(lvv,XX,YY,dd):
            b={}
            for l,x,y,v in zip(lvv,XX,YY,dd): b[(x//side_,y//side_)]=b.get((x//side_,y//side_),0.0)+v
            return b
        B0=block(lv0,X0,Y0,d0);B1=block(lv,X,Y,d1); assert max(abs(B0[k]-B1[k]) for k in B0)<1e-3,"block div"
        for i in range(t.n):
            key=(lv[i],X[i],Y[i])
            if key in k0: assert abs(d1[i]-d0[k0[key]])<1e-3,("kept div",i)
        # antisymmetry
        tbn=csr_table(t); pt=cp.asnumpy(t._array("PTR")); Qn=cp.asnumpy(q); rr,kk=np.nonzero(tbn>=0)
        pos={}
        for p,(i,j) in enumerate(zip(rr,tbn[rr,kk])): pos[(i,j)]=p
        for (i,j),p in pos.items(): assert abs(Qn[p]+Qn[pos[(j,i)]])<1e-4,("antisym",i,j)
    # lake volume with dry cells
    t.close()
    t=DynamicQuadtree(be,pool,lm,1.0,L,topology="D4",boundary=bd,capacity=nx*ny)
    zl=t.from_raster(dem); eta=float(np.median(dem)); hl=cp.maximum(eta-zl,0.0).astype(cp.float32)
    for it in range(4):
        n0=t.n; V0=float((hl[:n0].astype(cp.float64)*4.0**t.level).sum())
        t.adapt(random_flags(t,.3,0),[(zl,tf.FromRaster(dem)),(hl,tf.Depth(zl))])
        hn=cp.asnumpy(hl)[:t.n]; zn=cp.asnumpy(zl)[:t.n]; V1=(hn.astype(np.float64)*4.0**cp.asnumpy(t.level)).sum()
        assert abs(V0-V1)<=1e-3*V0,("lake vol",V0,V1)
    t.close()
    # a fully submerged lake stays at the same water surface
    t=DynamicQuadtree(be,pool,lm,1.0,L,topology="D4",boundary=bd,capacity=nx*ny)
    zl=t.from_raster(dem); eta=float(dem.max())+1.0; hl=(eta-zl).astype(cp.float32)
    for it in range(3):
        t.adapt(random_flags(t,.3,0),[(zl,tf.FromRaster(dem)),(hl,tf.Depth(zl))])
        assert np.abs(cp.asnumpy(zl)[:t.n].astype(np.float64)+cp.asnumpy(hl)[:t.n]-eta).max()<5e-3,"lake surface"
    t.close()
    # velocity with depth keeps the momentum, through splits and merges
    t=DynamicQuadtree(be,pool,lm,1.0,L,topology="D4",boundary=bd,capacity=nx*ny)
    zl=t.from_raster(dem); hl=t.from_raster(np.maximum(float(np.median(dem))-dem,0)+.01).astype(cp.float64)
    ul=cp.zeros_like(hl); ul[:t.n]=cp.asarray(np.random.default_rng(3).normal(size=t.n))
    moment=lambda: float((hl[:t.n]*ul[:t.n]*4.0**t.level.astype(cp.float64)).sum())
    for it in range(3):
        m0=moment(); t.adapt(random_flags(t,.2,.3),[(zl,tf.FromRaster(dem)),(ul,tf.Velocity(hl)),(hl,tf.Depth(zl))])
        assert abs(moment()-m0)<1e-6*(abs(m0)+1),"velocity momentum"
    t.close(); print("transfers ok",ny,nx,L,bd,flush=True)
# water_fill and Linear on known values
wf=cp.asnumpy(tf.water_fill(cp.asarray([1.0,0.0,0.1]),cp.asarray(np.array([[0.,0.,0.],[0.,0.,0.],[4.,1.,1.],[4.,1.,1.]]))))
assert np.allclose(wf[:,0],[2,2,0,0]) and np.allclose(wf[:,1],0) and np.allclose(wf[:,2],[.2,.2,0,0]),wf
t=DynamicQuadtree(be,pool,np.full((64,64),2,np.int32),1.0,2,topology="D4",capacity=4096)
ramp=np.add.outer(0.25*np.arange(64),0.5*np.arange(64)).astype(np.float32); c=t.from_raster(ramp)
ix=np.nonzero((cp.asnumpy(t.x)==16)&(cp.asnumpy(t.y)==16))[0]; f=np.zeros(t.n,np.int32); f[ix]=1
t.adapt(f,[(c,"linear")]); ok=cp.asnumpy(t.level)==1; 
exp=cp.asnumpy(t.from_raster(ramp)); assert np.allclose(cp.asnumpy(c)[:t.n][ok],exp[:t.n][ok],atol=1e-4),"linear exact on a ramp"
try: t.regrid(np.full((64,64),2,np.int32),[(c,"linear")]); raise SystemExit("regrid accepted linear")
except NotImplementedError: pass
t.close(); print("water_fill and linear ok",flush=True)
# SFD through the Program option
D=512; rng=np.random.default_rng(7)
fy=np.fft.fftfreq(D)[:,None]; fx=np.fft.rfftfreq(D)[None,:]; kk=np.hypot(fx,fy); kk[0,0]=1; amp=kk**-1.2; amp[0,0]=0
zh=np.fft.irfft2(amp*np.exp(2j*np.pi*rng.random((D,D//2+1))),s=(D,D)).astype(np.float32); zh=(zh-zh.min())/(zh.max()-zh.min())*500
with SFDFlowProgram(be,nx=D,ny=D) as dense:
    dense.z.from_numpy(zh); dense.route(); dense.resolve_minima(); dense.accumulate(); A=dense.drainage.to_numpy()
lm=np.clip(5-np.floor(np.log2(A)/2).astype(np.int32),0,5)
def run_sfd(f,t):
    f.z.array[...]=t.from_raster(zh); ar=cp.zeros(t.capacity,cp.float32); ar[:t.n]=(4.0**t.level).astype(cp.float32); f.source.from_numpy(cp.asnumpy(ar).reshape(1,-1))
    f.route(); f.resolve_minima(); f.accumulate()
    rec=f.rec.to_numpy().ravel(); q=f.drainage.to_numpy().ravel(); fl=cp.asnumpy(t._array("FLAGS")); ids=np.arange(t.capacity)
    assert not ((rec==ids)&((fl&1)==0)).any(),"sfd pits"
    assert (rec[t.n:]==ids[t.n:]).all(),"sfd tail"
    out=rec[:t.n]==ids[:t.n]; total=q[:t.n][out].astype(np.float64).sum()
    assert abs(total-D*D)<1e-3*D*D,("sfd area",total)
    r=rec[:t.n]; assert (cp.asnumpy(t.level)[r]>=0).all()
    return total
for topo in ("D4","D8"):
    for minima in ("cordonnier_carve","cordonnier_jump"):
        for accum in ("pointer_jump_push","rake_compress"):
            t=DynamicQuadtree(be,pool,lm,1.0,5,topology=topo)
            f=SFDFlowProgram(be,tree=t,topology=topo,accumulation=accum,local_minima=minima,source=np.ones((1,t.capacity),np.float32),**t.dims)
            r=[run_sfd(f,t)]
            fl=np.zeros(t.n,np.int32); fl[rng.choice(t.n,t.n//100,replace=False)]=1
            t.adapt(fl,[(f.z,"mean"),(f.drainage,"sum")]); r.append(run_sfd(f,t))
            t.adapt(-np.ones(t.n,np.int32)); r.append(run_sfd(f,t))
            f.close(); check_helpers(t,(False,False),"after program close")
            g=SFDFlowProgram(be,tree=t,topology=topo,accumulation=accum,local_minima=minima,source=np.ones((1,t.capacity),np.float32),**t.dims)
            r.append(run_sfd(g,t))
            try: t.close(); raise SystemExit("tree closed while bound")
            except Exception as e:
                if isinstance(e,SystemExit): raise
            g.close(); t.close()
            print("sfd ok",topo,minima,accum,"n=",t.n,"area",r,flush=True)
t=Quadtree(be,pool,lm,1.0,5,topology="D4")
for bad in ({"topology":"D8"},{"topology":"D4","nx":5},{"topology":"D4","dx":2.0}):
    a={**t.dims,**bad}
    try:
        p=SFDFlowProgram(be,tree=t,source=np.ones((1,a["nx"]),np.float32),**a); p.z; raise SystemExit(("mismatch accepted",bad))
    except Exception as e:
        if isinstance(e,SystemExit): raise
        print("rejected:",str(e)[:70])
t.close()
print("all checks passed")
