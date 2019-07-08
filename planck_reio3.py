import argparse
import os
import os.path as osp
from shutil import copytree, rmtree
from tempfile import mkdtemp

import camb
import dill as pickle
from numpy import *
from numpy.linalg import norm
from numpy.random import uniform
from scipy.interpolate import pchip_interpolate
from scipy.optimize import brentq

from cosmoslik import *

camb.ignore_fatal_errors.value = True
camb.reionization.include_helium_fullreion.value = False

param = param_shortcut('start','scale')

class BadXe(Exception): pass

class CambReio(SlikPlugin):
    """
    CAMB with reionization eigenmodes
    """

    camb_params = ['As','ns','ombh2','omch2','cosmomc_theta','pivot_scalar','mnu','tau','ALens','nrun','r']
    z = linspace(0,50,1024)

    def __init__(self, params, clip=False, DoLensing=True, fHe=0.08, gpprior=False,
                 hardxe=None, include_helium_fullreion=True, lmax=5000, mhfid=False,
                 mhfidbase=0.15, mhprior=False, model=None, modesfile=None, nmodes=None,
                 reioknots=None, smooth=1e-5, zmin_cut=None):
        
        super().__init__(**arguments(exclude=['params']))

        # add the appropriate sampled parameters based on reionization model
        if 'reiotanh' in model:
            params.tau = param(0.055,0.01,min=0.02,max=0.2)
        
        elif 'reioknots' in model:
            params.reioknots = SlikDict()
            for zk in self.reioknots:
                params.reioknots['xe%.4i'%int(100*zk)] = param(0.2, 0.3, min=0, max=1+fHe)
        
        elif ('reioflexknots' in model) or ('reioflexknots2' in model):
            params.reioflexknots = SlikDict(nknots=nmodes)
            for i in range(nmodes):
                params.reioflexknots['z%i'%i]  = param(uniform(6,30), 4, min=(6 if self.gpprior else 1), max=30)
            for i in (range(nmodes) if 'reioflexknots2' in model else range(1,nmodes-1)):      
                params.reioflexknots['xe%i'%i] = param(0.5, 0.3, min=0, max=1+fHe)
        
        elif 'reiomodes' in model:
            
            # load eigenmodes
            dat = loadtxt(modesfile)
            z_modes, self.modes = dat[0], dat[1:]
            # interpolate onto our z values
            self.modes = array([interp(self.z,z_modes,m,left=0,right=0) for m in self.modes])
            self.Nz = len(self.z)
            
            if mhfid:
                xe_max, xe_base = 1+fHe, self.mhfidbase
                self.xe_fid = ((xe_max-xe_base)*(1-tanh((self.z-6.25)/0.1))/2 + xe_base) * ((1-tanh((self.z-30)/0.1))/2)
            
            if mhprior:
                assert mhfid, "--mhfid must be set if --mhprior is set"
                self.mplus, self.mminus = (sum(self.modes * (xe_max-2*self.xe_fid) + x*xe_max*abs(self.modes),axis=1)/2/norm(self.modes,axis=1)**2 for x in [1,-1])
                
                self.support = support = ~all(self.modes==0,axis=0)
                self.radius = sqrt(sum(support*(xe_max-self.xe_fid)**2) / sum(support))

            params.reiomodes = SlikDict()
            for i in range(nmodes):
                p = params.reiomodes['mode%i'%i] = param(0,0.3 if "mh" in modesfile else 3)
                if mhprior and 'cube' in mhprior:
                    p.min = self.mminus[i]
                    p.max = self.mplus[i]
                if 'binmodes' in modesfile:
                    p.min = 0
        
        else:
            params.tau = fidtau


    def __call__(self,background_only=False,**params):
        
        cp = self.cp = camb.set_params(lmax=self.lmax,H0=None,**{k:params[k] for k in self.camb_params if k in params})
        cp.k_eta_max_scalar = 2*self.lmax
        cp.DoLensing = self.DoLensing
        cp.NonLinear = 0
        cp.AccurateReionization = True
        cp.WantTensors = bool(params.get('r'))
        self.H0 = cp.H0
        fHe = self.fHe
        
        # compute Xe(z)
        if self.mhfid:
            self.xe = self.xe_fid
            
        elif "reiotanh" in self.model:
            camb.get_background(cp)
            self.xe = cp.Reion.get_xe(z=self.z)
            
        elif "reioexp" in self.model:
            def dtau(zreio):
                f = 1+fHe
                zp = 6.1
                xe = f*exp(-(log(0.5)/(zp-zreio))*(self.z-zp)/(1+0.02/(1e-6+self.z-zp)**2))
                xe[self.z<zp] = f
                xe = pchip2_interpolate(self.z,xe,self.z,nsmooth=30)
                self.xe = cp.Reion.set_xe(z=self.z,xe=xe,smooth=self.smooth)
                return camb.get_background(cp,no_thermo=True).get_tau() - params["tau"]
            dtau(brentq(dtau,6.2,10,rtol=1e-3))
            
        elif ("reioknots" in self.model) or ("reioflexknots" in self.model) or ("reioflexknots2" in self.model):
           
            if "reioflexknots" in params:
                N = params['reioflexknots'].nknots
                zi = sorted([params['reioflexknots']['z%i'%i] for i in range(N)])
                if "reioflexknots2" in self.model:
                    xei = [params['reioflexknots']['xe%i'%i] for i in range(N)]
                else:
                    xei = hstack([1+fHe,[params['reioflexknots']['xe%i'%i] for i in range(1,N-1)],0])
            else:
                zi,xei = transpose(sorted([(float(k[2:])/100,v) for k,v in params['reioknots'].items()]))      
               
            self.xe = pchip2_interpolate(self.z, pchip_interpolate(hstack([0,(5.9 if self.gpprior else 1),zi,30,50]), hstack([1+fHe,1+fHe,xei,0,0]), self.z), self.z, nsmooth=200)
            
        else:
            self.xe = zeros(self.Nz)
                   
        
        # if we have modes, add these in
        if 'reiomodes' in self.model:
            m = array([params['reiomodes'].get('mode%i'%i,0) for i in range(len(self.modes))])
            
            if any(m!=0):
                self.xe = self.xe + dot(m,self.modes)
                
                if self.mhprior and 'cube' in self.mhprior:
                    if any(m < self.mminus) or any(m > self.mplus): raise BadXe()
                if self.mhprior and 'sphere' in self.mhprior:
                    if norm(m) > self.radius: raise BadXe()
                
                if self.hardxe:
                    if any(self.xe<self.hardxe[0]) or any(self.xe>self.hardxe[1]): raise BadXe()
                
                if self.gpprior:
                    if interp(6,self.z,self.xe) < (0.99 * 1+fHe): raise BadXe()
                
                if self.clip:
                    self.xe = clip(self.xe,0,1+fHe)
                    
                self.xe = pchip2_interpolate(self.z,self.xe,self.z,nsmooth=30)
                
        # add in helium reionization
        if self.include_helium_fullreion:
            self.xe = self.xe + fHe*(tanh(-(self.z-3.5)/0.5)+1)/2
        
        # useful only for postprocessing to get τ(zmax, zmin_cut)
        if self.zmin_cut:
            self.xe[self.z<self.zmin_cut] = 0
        
        self.xe = cp.Reion.set_xe(z=self.z, xe=self.xe)
        self.xe_thin = self.xe[::10]
        
        if background_only:
            r = self.results = camb.get_background(cp)
            return r
        else:
            r = self.results = camb.get_results(cp)
            Dℓ = dict(zip(['TT','EE','BB','TE'],(cp.TCMB*1e6)**2*r.get_total_cls(self.lmax).T))
            toDℓ = hstack([[1],1/arange(1,self.lmax+1)/(arange(1,self.lmax+1)+1)])
            Dℓ.update(dict(list(zip(['pp','pT','pE'],self.results.get_lens_potential_cls(self.lmax).T * toDℓ))))
            Dℓ['TB'] = Dℓ['EB'] = zeros(self.lmax)
            return Dℓ



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
                 undo_tau_prior='',
                 model='lcdm_reiotanh', 
                 only_lowp=False, 
                 nmodes=5, 
                 lowl='simall_EE',
                 tau_prior='',
                 reioknots="6,7.5,10,20",
                 gpprior=False,
                 lowp_lmax=None,
                 lowp_lmin=None,
                 doplot=False,
                 noplotbadxe=False,
                 sampler='mh',
                 mhprior='cube_sphere',
                 mhfid=False,
                 mhfidbase=0.15,
                 hardxe='(-0.5,1.5)',
                 clip=False,
                 fidtau=0.055,
                 no_clik=False,
                 lensing=False,
                 highl='plik_lite_v18_TT.clik',
                 extras='',
                 covmat=[],
                 auto_reject_errors=True,
                 ):
        
        hardxe=eval(hardxe)
        if tau_prior: tau_prior = eval(tau_prior)
        if isinstance(reioknots,str): reioknots = eval(reioknots)
        super().__init__(**arguments())
        
        model = model.split('_')
        assert all([x in ['lcdm','mnu','tau','nrun','r','reiotanh','reiomodes','reioknots','reioflexknots','reioflexknots2','ALens','fixA','fixclamp'] for x in model]), "Unrecognized model"
        assert lowl in ['simlow','bflike','cvlowp','simlowlike','commander','lfi_EE'] or 'simlowlikeclik' in lowl or lowl.startswith('simall'), "Unrecognized lowl likelihood"



        # generate the file name for this chain based on all the options set
        self.run_id = run_id = ['_'.join(model)]
        if 'reiomodes' in model or 'reioflexknots' in model or 'reioflexknots2' in model: 
            run_id.append('nmodes%i'%nmodes)
        if 'reioknots' in model:
            run_id.append('knots_'+'_'.join(map(str,reioknots)))
        if fidtau!=0.055: run_id.append('fidtau%.3f'%fidtau)
        if no_clik:
            run_id.append("noclik")
        else:
            if self.only_lowp: run_id.append('onlylowp')
            run_id.append(lowl)
            if lowp_lmax: run_id.append("lowplmax%s"%lowp_lmax)
            if lowp_lmin: run_id.append("lowplmin%s"%lowp_lmin)
            if highl!='plik_lite_v18_TT.clik': run_id.append(highl)
            if lensing: run_id.append('lensing')
        if gpprior: run_id.append("gpprior")
        if tau_prior:
            run_id.append('taup%.3i%.3i'%(int(1e3*self.tau_prior[0]),int(1e3*self.tau_prior[1])))
        if sampler=='emcee': run_id.append('emcee')
        if 'reiomodes' in model: 
            run_id.append(osp.splitext(modesfile)[0])
            if mhprior=='cube_sphere': run_id.append("mhprior")
            elif mhprior: run_id.append("mhprior"+mhprior)
            if mhfid: run_id.append("mhfid")
            if mhfidbase!=0.15: run_id.append("mhfidbase%.3i"%(int(1e2*mhfidbase)))
        if undo_tau_prior:
            if isinstance(undo_tau_prior,str):
                run_id.append("undo_"+osp.basename(undo_tau_prior).replace('.dat',''))
            else:
                run_id.append("undo_tau_prior")
        if clip:
            run_id.append("clip")
        if hardxe!=(-0.5,1.5):
            run_id.append("hardxe%.3i%.3i"%(int(1e2*abs(hardxe[0])),int(1e2*hardxe[1])))



        if "lcdm" in model:
            self.cosmo = SlikDict(
                ns = param(0.962,0.006),
                ombh2 = param(0.02221,0.0002),
                omch2 = param(0.1203,0.002),
                theta = param(0.0104,0.00003),
                mnu = param(0.06,0.1,min=0) if 'mnu' in model else 0.06,
                ALens = param(1,0.02) if 'ALens' in model else 1
            )
        else:
            self.cosmo = SlikDict(
                ns = 0.962,
                ombh2 = 0.02221,
                omch2 = 0.1203,
                theta = 0.0104,
            )
            
        self.cosmo.pivot_scalar = self.cosmo.pivot_tensor = 0.05
        
        if 'nrun' in model:
            self.cosmo.nrun = param(0,0.01)
            
        if 'fixA' in model or 'fixclamp' in model:
            self.cosmo.logA = 3.108
        else:
            self.cosmo.logA = param(3.108,0.03,min=2.8,max=3.4)
            
        if 'r' in model:
            self.cosmo.r = param(0.1,0.1,min=0)
            
        if tau_prior:   
            self.priors.add_gaussian_prior('cosmo.tau_out',*self.tau_prior)
            
        self.camb = CambReio(self.cosmo,
                             modesfile=modesfile,
                             lmax=200 if only_lowp else 5000,
                             DoLensing=(not only_lowp),
                             mhprior=mhprior,
                             mhfid=mhfid,
                             model=model,
                             nmodes=nmodes,
                             reioknots=reioknots,
                             mhfidbase=mhfidbase,
                             gpprior=gpprior,
                             clip=clip,
                             hardxe=hardxe)
        
        if undo_tau_prior:
            if isinstance(undo_tau_prior,str):
                with open(self.undo_tau_prior,"rb") as f:
                    self.tau_prior = pickle.load(f)
            else:
                self.tau_prior = self.undo_tau_prior

        if not only_lowp:
            self.calPlanck = param(1,0.0025,gaussian_prior=(1,0.0025))
        else:
            self.calPlanck = 1

        if not no_clik:

            if not only_lowp:
                self.highl = likelihoods.planck.clik(clik_file=self.highl)
                
            if lensing:
                self.lensing = likelihoods.planck.clik(
                    clik_file='smicadx12_Dec5_ftl_mv2_ndclpp_p_teb_consext8.clik_lensing'
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
                    auto_reject_errors=auto_reject_errors
                )
            elif lowl.startswith('simall'):
                self.lowlP = likelihoods.planck.clik(
                    clik_file='simall_100x143_offlike5_%s_Aplanck_B.clik'%(lowl.split('_')[1])
                )
            elif lowl == 'lfi_EE':
                self.lowlP = likelihoods.planck.clik(
                    clik_file='lfi_likelihood/lowl_70_dx12_QU.clik'
                )
            elif lowl=='bflike':
                if lowp_lmax or lowp_lmin: raise ValueError("bflike lmax not implemented")
                IQUspec = 'QU' if only_lowp else 'SMW'
                self.lowlP = likelihoods.planck.clik(
                    clik_file='/redtruck/benabed/release/clik_10.3/low_l/bflike/lowl_%s_70_dx11d_2014_10_03_v5c_Ap.clik/'%IQUspec,
                    auto_reject_errors=auto_reject_errors
                )
            elif lowl in ['cvlowp','simlowlike']:
                assert not lowp_lmin, "not implemented"
                p0 = SlikDict({k:(v.start if isinstance(v,param) else v) for k,v in self.cosmo.items() if not isinstance(v,dict)})
                camb = CambReio(p0,lmax=200,DoLensing=False,model='reiotanh')
                p0["cosmomc_theta"] = p0["theta"]
                p0["As"] = exp(p0["logA"])*1e-10
                p0["tau"] = 0.055
                clEEobs = camb(**p0)["EE"]
                if lowl=='simlowlike':
                    l = arange(100)
                    nlEEobs = (0.0143/l + 0.000279846) * l**2 / (2*pi)
                    fsky = 0.5
                else:
                    nlEEobs = 0*clEEobs
                    fsky = 1
                self.lowlP = CVLowP(clEEobs=clEEobs,nlEEobs=nlEEobs,fsky=fsky,lrange=(2,(int(lowp_lmax) if lowp_lmax else 30)+1))
            else:
                raise ValueError(lowl)
            
        self.priors = likelihoods.priors(self)

        

        if sampler=='polychord':
            self.sampler = samplers.polychord(
                self,
                output_file = 'polychord/'+'_'.join(run_id),
                output_extra_params = ['cosmo.tau_out', 'cosmo.tau15_out', 'cosmo.H0'],
                read_resume = False,
                do_clustering = False,
                nlive = 300
            )
        else:
            extra_format = {'clTT':('clTT','(100,)d'),
                            'clTE':('clTE','(100,)d'),
                            'clEE':('clEE','(100,)d'),
                            'xe':('camb.xe_thin','(103,)d')}
            _sampler = {'mh':samplers.metropolis_hastings, 'emcee':samplers.emcee, 'priors':samplers.priors}[sampler]
            self.sampler = _sampler(
                self,
                num_samples = 1e8,
                output_file = 'chains/chain_'+'_'.join(run_id),
                cov_est = covmat,
                output_extra_params = [
                    'lnls.highl','lnls.lowlT','lnls.lowlP','lnls.inv_tau_prior','lnls.lensing',
                    'cosmo.tau_out','cosmo.tau15_out','cosmo.H0'
                ] + ([extra_format[e] for e in extras.split(',')] if extras else [])
            )
            if sampler in ['mh','priors']:
                self.sampler.update(dict(
                    print_level = 1,
                    proposal_update_start = 500,
                    proposal_scale = 1.5,
                    mpi_comm_freq = 10 if sampler=='mh' else 100,
                ))
            elif sampler=='emcee':
                self.sampler.update(dict(
                    nwalkers = 100,
                    output_freq = 1
                ))
                                

        


    def __call__(self, background_only=False):
        
        background_only = background_only or self.no_clik
        
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
                if self.get('lensing'):
                    self.lensing.A_planck = self.calPlanck
            
            self.lnls = SlikDict({k:nan for k in ['highl','lowlT','lowlP','inv_tau_prior','lensing']})

            #temp workaround for first step
            self.cosmo.tau_out = nan 
            
            self.cls = self.camb(background_only=background_only,**self.cosmo)
            
            self.cosmo.H0 = self.camb.H0
            
            
            if (not background_only and 
                (any([any(isnan(x)) for x in list(self.cls.values())])
                 or any([any(x>1e5) for x in list(self.cls.values())]))):
                raise BadXe("CAMB outputted crazy Cls")
            
            badxe = False
        except Exception as e:
            badxe = isinstance(e,BadXe)
            if not badxe:
                if self.auto_reject_errors:
                    print("Warning, rejecting step: "+str(e))
                else:
                    raise
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
        self.cosmo.tau15_out = self.camb.results.get_tau(15,50)

        if not background_only: 
            self.clTT = self.cls['TT'][:100]
            self.clTE = self.cls['TE'][:100]
            self.clEE = self.cls['EE'][:100]
        
        return lsumk(self.lnls,
            [('highl',   lambda: self.highl(self.cls)   if callable(self.get('highl'))   else 0),
             ('lowlT',   lambda: self.lowlT(self.cls)   if callable(self.get('lowlT'))   else 0),
             ('lowlP',   lambda: self.lowlP(self.cls)   if callable(self.get('lowlP'))   else 0),
             ('lensing', lambda: self.lensing(self.cls) if callable(self.get('lensing')) else 0),
             ('inv_tau_prior', lambda: log(max(1e-4,nan_to_num(self.tau_prior(self.cosmo.tau_out)))) if self.undo_tau_prior else 0)]
        )



"""Like pchip_interpolate but keeps the second derivative continuous too"""
def pchip2_interpolate(xi,yi,x,nsmooth=None):
    if nsmooth:
        xi2 = linspace(min(xi),max(xi),nsmooth)
        xi, yi = xi2, interp(xi2,xi,yi)
               
    y = hstack([[0],cumsum(pchip_interpolate((xi[1:]+xi[:-1])/2,diff(yi),(x[1:]+x[:-1])/2))])
    y /= y[0] - y[-1] / (yi[0] - yi[-1]) 
    return y + yi[0]
