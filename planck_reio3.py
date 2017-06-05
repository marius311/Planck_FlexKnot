import argparse
import os
import os.path as osp
from shutil import copytree, rmtree
from tempfile import mkdtemp
from scipy.optimize import brentq
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

    def __init__(self, modesfile, lmax=5000, DoLensing=True, reiomodel="tanh",
                 mhprior=False, gpprior=False, clip=False, hardxe=None, mhfid=False, mhfidbase=0.15):
        super().__init__(lmax=lmax, DoLensing=DoLensing, mhprior=mhprior, gpprior=gpprior, 
                         clip=clip, hardxe=hardxe, mhfid=mhfid, mhfidbase=mhfidbase, reiomodel=reiomodel)

        #load eigenmodes
        dat = loadtxt(modesfile)
        z_modes, self.modes = dat[0], dat[1:]
        # interpolate onto our z values
        self.modes = array([interp(self.z,z_modes,m,left=0,right=0) for m in self.modes])
        self.Nz = len(self.z)
        α = (max(z_modes)-min(z_modes))/(max(self.z)-min(self.z))
        
        if mhfid:
            f, b = 1.08, self.mhfidbase
            self.xe_fid = ((f-b)*(1-tanh((self.z-6.25)/0.1))/2 + b) * ((1-tanh((self.z-30)/0.1))/2)
        
        if mhprior:
            assert mhfid, "--mhfid must be set if --mhprior is"
            self.mplus, self.mminus = (sum(self.modes * (f-2*self.xe_fid) + x*f*abs(self.modes),axis=1)/self.Nz/2/α for x in [1,-1])


    def __call__(self,background_only=False,**params):
        
        cp = self.cp = camb.set_params(lmax=self.lmax,H0=None,**{k:params[k] for k in self.camb_params if k in params})
        cp.k_eta_max_scalar = 2*self.lmax
        cp.DoLensing = self.DoLensing
        cp.NonLinear = 0
        self.H0 = cp.H0
        
        # get fiducial 
        if self.mhfid:
            self.xe = self.xe_fid
        else:
            if self.reiomodel=="tanh":
                camb.get_background(cp)
                self.xe = self.xe_fid = cp.Reion.get_xe(z=self.z)
            elif self.reiomodel=="exp":
                def dtau(zreio):
                    f = 1.08
                    zp = 6.1
                    xe = f*exp(-(log(0.5)/(zp-zreio))*(self.z-zp)/(1+0.02/(1e-6+self.z-zp)**2))
                    xe[self.z<zp] = f
                    self.xe = self.xe_fid = cp.Reion.set_xe(z=self.z,xe=xe,smooth=1e-3)
                    return camb.get_background(cp,no_thermo=True).get_tau() - params["tau"]
                dtau(brentq(dtau,6.2,10,rtol=1e-3))
            else:
                raise ValueError("Unrecognized reionization model '%s'"%self.reiomodel)
                   
            
        # add in modes
        if 'reiomodes' in params:
            m = array([params['reiomodes'].get('mode%i'%i,0) for i in range(len(self.modes))])
            
            if any(m!=0):
                self.xe = self.xe_fid + dot(m,self.modes)
                
                if self.mhprior:
                    if any(m < self.mhprior*self.mminus) or any(m > self.mhprior*self.mplus): raise BadXe()
                
                if self.hardxe:
                    if any(self.xe<self.hardxe[0]) or any(self.xe>self.hardxe[1]): raise BadXe()
                
                if self.gpprior:
                    if interp(6,self.z,self.xe) < (0.99 * 1.08): raise BadXe()
                
                if self.clip:
                    self.xe = clip(self.xe,0,1.17)
                
                self.xe = cp.Reion.set_xe(z=self.z,xe=self.xe,smooth=0.3)
            
        
        self.xe_thin = self.xe[::10]
        
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
                 reiomodel="tanh",
                 gpprior=False,
                 lowp_lmax=None,
                 lowp_lmin=None,
                 doplot=False,
                 noplotbadxe=False,
                 sampler='mh',
                 mhprior=0,
                 mhfid=False,
                 mhfidbase=0.15,
                 hardxe='(-0.5,1.5)',
                 clip=False,
                 fidtau=0.055,
                 no_clik=False,
                 extras='',
                 covmat=[]):
        
        hardxe=eval(hardxe)
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
                self.cosmo.reiomodes['mode%i'%i] = param(0,0.3 if "mh" in modesfile else 3)
                if 'binmodes' in modesfile:
                    self.cosmo.reiomodes['mode%i'%i].min = 0

        self.camb = CambReio(modesfile,
                             lmax=200 if only_lowp else 5000,
                             DoLensing=(not only_lowp),
                             mhprior=mhprior,
                             mhfid=mhfid,
                             reiomodel=reiomodel,
                             mhfidbase=mhfidbase,
                             gpprior=gpprior,
                             clip=clip,
                             hardxe=hardxe)
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
                mdb={}
                if lowp_lmax: mdb['lmax'] = lowp_lmax
                if lowp_lmin: mdb['lmin'] = lowp_lmin
                self.lowlP = ClikCustomMDB(
                    clik_file=lowl.replace("clik","")+'_MA_EE_2_32_2016_03_31.clik',
                    mdb=mdb,
                    auto_reject_errors=True
                )
            elif lowl=='bflike':
                if lowp_lmax or lowp_lmin: raise ValueError("bflike lmax not implemented")
                IQUspec = 'QU' if only_lowp else 'SMW'
                self.lowlP = likelihoods.planck.clik(
                    clik_file='/redtruck/benabed/release/clik_10.3/low_l/bflike/lowl_%s_70_dx11d_2014_10_03_v5c_Ap.clik/'%IQUspec,
                    auto_reject_errors=True
                )
            elif lowl in ['cvlowp','simlowlike']:
                assert not lowp_lmin, "not implemented"
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

        # self.priors.add_uniform_prior('cosmo.tau_out',0,0.1)

        # generate the file name for this chain based on all the options set
        run_id = [model]
        if 'reiomodes' in model: run_id.append('nmodes%i'%nmodes)
        if reiomodel!="tanh": run_id.append('reiomodel%s'%reiomodel)
        if fidtau!=0.055: run_id.append('fidtau%.3f'%fidtau)
        if no_clik:
            run_id.append("noclik")
        else:
            if self.only_lowp: run_id.append('onlylowp')
            run_id.append(lowl)
            if lowp_lmax: run_id.append("lowplmax%s"%lowp_lmax)
            if lowp_lmin: run_id.append("lowplmin%s"%lowp_lmin)
        if mhprior: run_id.append("mhprior"+(str(mhprior) if mhprior!=1 else ""))
        if mhfid: run_id.append("mhfid")
        if mhfidbase!=0.15: run_id.append("mhfidbase%.3i"%(int(1e2*mhfidbase)))
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
        if hardxe!=(-0.5,1.5):
            run_id.append("hardxe%.3i%.3i"%(int(1e2*abs(hardxe[0])),int(1e2*hardxe[1])))
        

        extra_format = {'clTT':('clTT','(100,)d'),
                        'clTE':('clTE','(100,)d'),
                        'clEE':('clEE','(100,)d'),
                        'xe':('camb.xe_thin','(103,)d')}
        _sampler = {'mh':samplers.metropolis_hastings, 'emcee':samplers.emcee}[sampler]
        self.sampler = _sampler(
            self,
            num_samples = 1e8,
            output_file = 'chains/chain_'+'_'.join(run_id),
            cov_est = covmat,
            output_extra_params = [
                'lnls.highl','lnls.lowlT','lnls.lowlP','lnls.inv_mode_prior',
                'cosmo.tau_out','cosmo.H0'
            ] + ([extra_format[e] for e in extras.split(',')] if extras else [])
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
                                

        


    def __call__(self, background_only=False):
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

                
            self.cls = self.camb(background_only=background_only,**self.cosmo)
            
            self.cosmo.H0 = self.camb.H0
            
            
            if (any([any(isnan(x)) for x in list(self.cls.values())])
                or any([any(x>1e5) for x in list(self.cls.values())])):
                raise Exception("CAMB outputted crazy Cls")
            
            badxe = False
        except Exception as e:
            badxe = isinstance(e,BadXe)
            if not badxe:
                print("Warning, rejecting step: "+str(e))
            return inf
        finally:
            if self.doplot and (not self.noplotbadxe or not badxe):
                from matplotlib.pyplot import ion, plot, ylim, cla, gcf
                ion()
                cla()
                plot(self.camb.z,self.camb.xe_fid)
                plot(self.camb.z,self.camb.xe,marker='.')
                plot(self.camb.z,self.camb.cp.Reion.get_xe(z=self.camb.z),marker=".")
                ylim(-1,2)
                gcf().canvas.draw() 
            
            
        self.cosmo.tau_out = self.camb.results.get_tau()

        if background_only: return 0
        
        self.clTT = self.cls['TT'][:100]
        self.clTE = self.cls['TE'][:100]
        self.clEE = self.cls['EE'][:100]
        return lsumk(self.lnls,
            [('highl',lambda: 0 if self.get('highl') is None else self.highl(self.cls)),
             ('lowlT',lambda: 0 if self.get('lowlT') is None else self.lowlT(self.cls)),
             ('lowlP',lambda: 0 if self.get('lowlP') is None else self.lowlP(self.cls)),
             ('inv_mode_prior', lambda: log(max(3e-2,nan_to_num(self.mode_prior(self.cosmo.tau_out)))) if self.undo_mode_prior else 0)]
        )
