import argparse
import os
import os.path as osp
from shutil import copytree, rmtree
from tempfile import mkdtemp
import pickle

import camb
from numpy import *

from cosmoslik import *

camb.ignore_fatal_errors.value = True
camb.reionization.include_helium_fullreion.value = False

param = param_shortcut('start','scale')

class BadXe(Exception): pass

class CambReio(SlikPlugin):
    """
    CAMB with reionization eigenmodes
    """

    camb_params = ['As','ns','ombh2','omch2','cosmomc_theta','pivot_scalar','mnu','tau','ALens','nrun']
    z = linspace(0,50,1024)

    def __init__(self, modesfile, lmax=5000, DoLensing=True, mhprior=False, gpprior=False, clip=False):
        super().__init__(lmax=lmax, DoLensing=DoLensing, mhprior=mhprior, gpprior=gpprior, clip=clip)

        #load eigenmodes
        dat = loadtxt(modesfile)
        self.z_modes, self.modes = dat[0], dat[1:]
        self.Nz = len(self.z_modes)
        
        if mhprior:
            f = 1.08
            b = 0.15
            self.xe_fid = ((f-b)*(1-tanh((self.z-6)/0.5))/2 + b) * ((1-tanh((self.z-27)/0.5))/2)
            self.mplus, self.mminus = (sum(self.modes * (f-2*self.xe_fid) + x*f*abs(self.modes),axis=1)/self.Nz/2 for x in [1,-1])


    def __call__(self,background_only=False,**params):
        
        cp = self.cp = camb.set_params(lmax=self.lmax,H0=None,**{k:params[k] for k in self.camb_params if k in params})
        cp.k_eta_max_scalar = 2*self.lmax
        cp.DoLensing = self.DoLensing
        cp.NonLinear = 0
        self.H0 = cp.H0
        
        # get fiducial 
        if self.mhprior:
            self.xe = self.xe_fid
        else:
            camb.get_background(cp)
            self.xe = self.xe_fid = cp.Reion.get_xe(z=self.z)
            
        # add in modes
        if 'reiomodes' in params:
            m = array([params['reiomodes'].get('mode%i'%i,0) for i in range(len(self.modes))])
            
            if any(m!=0):
                dxe = dot(m,self.modes)
                self.xe = self.xe_fid + interp(self.z,self.z_modes,dxe)
                
                if self.mhprior:
                    if any(m < self.mminus) or any(m > self.mplus): raise BadXe()
                else:
                    if any(self.xe<-0.5) or any(self.xe>1.5): raise BadXe()
                
                if self.gpprior:
                    if interp(6,self.z[::-1],self.xe[::-1]) < (0.99 * 1.08): raise BadXe()
                
                if self.clip:
                    self.xe = clip(self.xe,0,1.17)
                
                cp.Reion.set_xe(z=self.z,xe=self.xe,smooth=1e-2)
            
        
        
        self.xe = self.xe[::10] #thin to keep chain size down
        
        if background_only:
            r = self.results = camb.get_background(cp)
            return r
        else:
            r = self.results = camb.get_results(cp)
            cl = dict(zip(['TT','EE','BB','TE'],(cp.TCMB*1e6)**2*r.get_total_cls(self.lmax).T))
            cl['TB'] = cl['EB'] = zeros(self.lmax)
            return cl


class CVLowP(SlikPlugin):
    
    def __init__(self,clEEobs,nlEEobs=0,fsky=1,lrange=(2,30)):
        super().__init__(**arguments())
        self.lslice = slice(*lrange)
        self.l = arange(*lrange)
        self.clEEobs_tot = clEEobs[self.lslice] + nlEEobs[self.lslice]

    def __call__(self,cl,nlEE=None):
        if nlEE is None: nlEE = self.nlEEobs
        clEEtot = cl["EE"][self.lslice] + nlEE[self.lslice]
        return self.fsky*sum((2*self.l+1)/2*(log(clEEtot)+self.clEEobs_tot/clEEtot))


class ClikCustomMDB(likelihoods.planck.clik):
    
    def __init__(self,*args,mdb=None,clik_file=None,**kwargs):
        
        if mdb:
            tmpdir = mkdtemp()
            new_clik_file = osp.join(tmpdir,clik_file)
            copytree(clik_file, new_clik_file)
            
            mdb_file = osp.join(new_clik_file,"clik","lkl_0","_mdb")
            with open(mdb_file) as f:
                mdb_dat = [l.split() for l in f.readlines()]
            
            for k,v in mdb.items():
                for l in mdb_dat:
                    if l[0]==k:
                        l[2]=str(v)
                    
            with open(mdb_file,"w") as f:
                f.write("\n".join(" ".join(l) for l in mdb_dat)+"\n")
        else:
            tmpdir = None
            new_clik_file = clik_file
        
        super().__init__(*args,clik_file=new_clik_file,**kwargs)
        
        if tmpdir: rmtree(tmpdir)
    
    


@SlikMain
class planck(SlikPlugin):

    def __init__(self, 
                 modesfile="reiotau_xepcs_v3.dat",
                 undo_mode_prior='',
                 model='lcdm_tau', 
                 only_lowp=False, 
                 nmodes=5, 
                 lowl='simlow',
                 tauprior='',
                 gpprior=False,
                 lowp_lmax=None,
                 doplot=False,
                 sampler='mh',
                 mhprior=False,
                 clip=False,
                 fidtau=0.055,
                 no_clik=False,
                 covmat=[]):
        super().__init__(**arguments())
        
    
        assert all([x in ['lcdm','mnu','tau','nrun','reiomodes','ALens','fixA','fixclamp'] for x in model.split('_')]), "Unrecognized model"
        assert lowl in ['simlow','bflike','cvlowp','simlowlike','commander'] or 'simlowlikeclik' in lowl, "Unrecognized lowl likelihood"


        if "lcdm" in model:
            self.cosmo = SlikDict(
                ns = param(0.962,0.006),
                ombh2 = param(0.02221,0.0002),
                omch2 = param(0.1203,0.002),
                theta = param(0.0104,0.00003),
                pivot_scalar = 0.05,
                mnu = param(0.06,0.1,min=0) if 'mnu' in model else 0.06,
                ALens = param(1,0.02) if 'ALens' in model else 1
            )
        else:
            self.cosmo = SlikDict(
                ns = 0.962,
                ombh2 = 0.02221,
                omch2 = 0.1203,
                theta = 0.0104,
                pivot_scalar = 0.05,
            )
            
        if 'nrun' in model:
            self.cosmo.nrun = param(0,0.01)
            
        if 'fixA' in model or 'fixclamp' in model:
            self.cosmo.logA = 3.108
        else:
            self.cosmo.logA = param(3.108,0.03,min=2.8,max=3.4)
            
        if 'tau' in model:
            self.cosmo.tau = param(0.055,0.01,min=0.02,max=0.1)
        else:
            self.cosmo.tau = fidtau
            
        if 'reiomodes' in model:
            self.cosmo.reiomodes = SlikDict()
            for i in range(nmodes):
                self.cosmo.reiomodes['mode%i'%i] = param(0,0.3 if mhprior else 3)

        self.camb = CambReio(modesfile,
                             lmax=200 if only_lowp else 5000,
                             DoLensing=(not only_lowp),
                             mhprior=mhprior,
                             clip=clip)
        if undo_mode_prior:
            with open(self.undo_mode_prior,"rb") as f:
                self.mode_prior = pickle.load(f).get(self.nmodes, lambda tau: 1)

        if not only_lowp:
            self.calPlanck = param(1,0.0025,gaussian_prior=(1,0.0025))
        else:
            self.calPlanck = 1

        if not no_clik:

            if not only_lowp:
                self.highl = likelihoods.planck.clik(
                    clik_file='plik_lite_v18_TT.clik'
                )
                
            if lowl=='commander':
                self.lowlT = likelihoods.planck.clik(
                    clik_file='commander_rc2_v1.1_l2_29_B.clik'
                )
            elif (lowl=='simlow') or ('simlowlikeclik' in lowl):
                if not only_lowp:
                    self.lowlT = likelihoods.planck.clik(
                        clik_file='commander_rc2_v1.1_l2_29_B.clik'
                    )
                self.lowlP = ClikCustomMDB(
                    clik_file=lowl.replace("clik","")+'_MA_EE_2_32_2016_03_31.clik',
                    mdb={"lmax": lowp_lmax} if lowp_lmax else None,
                    auto_reject_errors=True
                )
            elif lowl=='bflike':
                if lowp_lmax is not None: raise ValueError("bflike lmax not implemented")
                IQUspec = 'QU' if only_lowp else 'SMW'
                self.lowlP = likelihoods.planck.clik(
                    clik_file='/redtruck/benabed/release/clik_10.3/low_l/bflike/lowl_%s_70_dx11d_2014_10_03_v5c_Ap.clik/'%IQUspec,
                    auto_reject_errors=True
                )
            elif lowl in ['cvlowp','simlowlike']:
                p0 = {k:(v if isinstance(v,(float,int)) else v.start) for k,v in self.cosmo.items() if k!="reiomodes"}
                p0["cosmomc_theta"] = p0["theta"]
                p0["As"] = exp(p0["logA"])*1e-10
                clEEobs = self.camb(**p0)["EE"]
                if lowl=='simlowlike':
                    l = arange(100)
                    nlEEobs = (0.0143/l + 0.000279846) * l**2 / (2*pi)
                    fsky = 0.5
                else:
                    nlEEobs = 0
                    fsky = 1
                self.lowlP = CVLowP(clEEobs=clEEobs,nlEEobs=nlEEobs,fsky=fsky,lrange=(2,(int(lowp_lmax) if lowp_lmax else 30)+1))
            else:
                raise ValueError(lowl)
            
        self.priors = likelihoods.priors(self)

        self.priors.add_uniform_prior('cosmo.tau_out',0,0.1)

        # generate the file name for this chain based on all the options set
        run_id = [model]
        if 'reiomodes' in model: run_id.append('nmodes%i'%nmodes)
        if fidtau!=0.055: run_id.append('fidtau%.3f'%fidtau)
        if no_clik:
            run_id.append("noclik")
        else:
            if self.only_lowp: run_id.append('onlylowp')
            run_id.append(lowl)
            if lowp_lmax: run_id.append("lowplmax%s"%lowp_lmax)
        if mhprior: run_id.append("mhprior")
        if gpprior: run_id.append("gpprior")
        if tauprior:
            μ,σ = eval(tauprior)
            self.priors.add_gaussian_prior('cosmo.tau_out',μ,σ)
            run_id.append('taup%.3i%.3i'%(int(1e3*μ),int(1e3*σ)))
        if sampler=='emcee': run_id.append('emcee')
        if modesfile!="reiotau_xepcs.dat": run_id.append(osp.splitext(modesfile)[0])
        if undo_mode_prior:
            run_id.append("undo_"+undo_mode_prior.replace(modesfile.replace('.dat',''),'').replace('.dat','').strip('_'))
        if clip:
            run_id.append("clip")
        

        _sampler = {'mh':samplers.metropolis_hastings, 'emcee':samplers.emcee}[sampler]
        self.sampler = _sampler(
            self,
            num_samples = 1e8,
            output_file = 'chains/chain_'+'_'.join(run_id),
            cov_est = covmat,
            output_extra_params = [
                'lnls.highl','lnls.lowlT','lnls.lowlP','lnls.inv_mode_prior',
                # ('clTT','(100,)d'),('clTE','(100,)d'),('clEE','(100,)d'),('camb.xe','(103,)d'), 
                'cosmo.tau_out','cosmo.H0'
            ]
        )
        if sampler=='mh':
            self.sampler.update(dict(
                print_level = 1,
                proposal_update_start = 2000,
                proposal_scale = 1.5,
                mpi_comm_freq = 5,
            ))
        elif sampler=='emcee':
            self.sampler.update(dict(
                nwalkers = 100,
                output_freq = 1
            ))
                                

        


    def __call__(self):
        try:
            
            self.cosmo.cosmomc_theta = self.cosmo.theta
            if 'fixclamp' in self.model:
                tau = self.camb(background_only=True,**self.cosmo).get_tau()
                self.cosmo.logA = log(1.881*10) + 2*tau
                
            self.cosmo.As = exp(self.cosmo.logA)*1e-10

            if not self.no_clik:
                if not self.only_lowp:
                    self.highl.A_Planck = self.highl.calPlanck = self.calPlanck
                    if self.get('lowlT'): self.lowlT.A_planck = self.calPlanck
                if 'lowlP' in self: 
                    self.lowlP.A_planck = self.calPlanck
            
            self.lnls = SlikDict({k:nan for k in ['highl','lowlT','lowlP','inv_mode_prior']})

                
            self.cls = self.camb(**self.cosmo)
            
            self.cosmo.H0 = self.camb.H0
            
            if self.doplot:
                from matplotlib.pyplot import ion, plot, ylim, cla, gcf
                ion()
                cla()
                plot(self.camb.z,self.camb.xe_fid)
                plot(self.camb.z,self.camb.cp.Reion.get_xe(z=self.camb.z),marker=".")
                ylim(-1,2)
                gcf().canvas.draw() 
            
            if (any([any(isnan(x)) for x in list(self.cls.values())])
                or any([any(x>1e5) for x in list(self.cls.values())])):
                raise Exception("CAMB outputted crazy Cls")
        except Exception as e:
            if not isinstance(e,BadXe):
                print("Warning, rejecting step: "+str(e))
            return inf

        self.clTT = self.cls['TT'][:100]
        self.clTE = self.cls['TE'][:100]
        self.clEE = self.cls['EE'][:100]
        self.cosmo.tau_out = self.camb.results.get_tau()
        return lsumk(self.lnls,
            [('highl',lambda: 0 if self.get('highl') is None else self.highl(self.cls)),
             ('lowlT',lambda: 0 if self.get('lowlT') is None else self.lowlT(self.cls)),
             ('lowlP',lambda: 0 if self.get('lowlP') is None else self.lowlP(self.cls)),
             ('inv_mode_prior', lambda: log(max(3e-2,nan_to_num(self.mode_prior(self.cosmo.tau_out)))) if self.undo_mode_prior else 0)]
        )
