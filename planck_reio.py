from cosmoslik import *
from numpy import *
import os, os.path as osp
from cosmoslik_plugins.likelihoods.clik import clik
from cosmoslik.subprocess_plugins import SubprocessClassDied
import argparse
import camb

param = param_shortcut('start','scale')

@subprocess_class(auto_restart=True)
class camb_reio(SlikPlugin):
    """
    CAMB with reionization eigenmodes
    """
    
    camb_params = ['As','ns','ombh2','omch2','cosmomc_theta','pivot_scalar','mnu']
    
    def __init__(self,lmax=5000):
        super(camb_reio,self).__init__(lmax=lmax)

        #load eigenmodes and smoothly transition the mode to Xe(z)=0 at high z
        #with a cosine window
        self.z,_=loadtxt("xefid.dat").T
        dl=20
        w=hstack([ones(len(self.z)-dl),(cos(pi*arange(dl)/(dl-1))+1)/2])
        self.modes=loadtxt("xepcs.dat")[1:]*w

        #compute fiducial Xe around which we perturb
        cp=camb.set_params(As=1e-9)
        camb.get_background(cp)
        self.fidxe=cp.Reion.get_xe(1/(1+self.z))


    def __call__(self,**params):

        cp=camb.set_params(lmax=self.lmax,H0=None,**{k:params[k] for k in self.camb_params})
        cp.k_eta_max_scalar=2*self.lmax
        cp.DoLensing=True
        cp.NonLinear=0
        if 'reiomodes' in params:
            self.xe=self.fidxe+sum([params['reiomodes']['mode%i'%i]*self.modes[i] 
                                    for i in range(95) if 'mode%i'%i in params['reiomodes']],axis=0)
            cp.Reion.set_xe(1/(1+self.z),self.xe)
        r=self.results=camb.get_results(cp)

        return dict(zip(['cl_%s'%x for x in ['TT','EE','BB','TE']],
                        (cp.TCMB*1e6)**2*r.get_cmb_power_spectra(spectra=['total'])['total'].T))



class planck(SlikPlugin):

    def __init__(self, model='lcdm_tau',nmodes=5):

        super(planck,self).__init__(**all_kw(locals()))

        self.cosmo = SlikDict(
            logA = param(3.108,0.03),
            ns = param(0.962,0.006),
            ombh2 = param(0.02221,0.0002),
            omch2 = param(0.1203,0.002),
            theta = param(0.0104,0.00003),
            pivot_scalar=0.05,
            mnu = 0.06,
        )
        self.cosmo.reiomodes = SlikDict()
        if 'tau' in model:
            self.cosmo.tau = param(0.085,0.01,min=0,gaussian_prior=(0.07,0.01))
        elif 'reiomodes' in model:
            for i in range(nmodes):
                self.cosmo.reiomodes['mode%i'%i] = param(0,0.03)

        self.get_cmb = camb_reio()
                
        self.calPlanck = param(1,0.0025,gaussian_prior=(1,0.0025))

        self.highl = clik(clik_file='plik_lite_v18_TT.clik')
        self.lowl = clik(clik_file='lowl_SMW_70_dx11d_2014_10_03_v5c_Ap.clik')

        self.priors = get_plugin('likelihoods.priors')(self)

        self.sampler = get_plugin('samplers.metropolis_hastings')(
            self,
            num_samples=1e7,
            print_level=3,
            proposal_update_start=500,
            mpi_comm_freq=10,
            output_file='chain',
            proposal_cov='planck_%s.covmat'%model,
            output_extra_params=[('get_cmb.xe','(95,)d')]
       )


    def __call__(self):
        self.cosmo.As = exp(self.cosmo.logA)*1e-10
        self.cosmo.cosmomc_theta = self.cosmo.theta
        self.highl.A_Planck = self.highl.calPlanck = self.calPlanck
        if self.lowl: self.lowl.A_planck = self.calPlanck

        try:
            self.cls = self.get_cmb(**self.cosmo)
        except SubprocessClassDied as e:
            print "Warning: "+str(e)
            return inf

        return lsum(
            lambda: self.priors(self),
            lambda: self.highl(self.cls),
            lambda: self.lowl(self.cls) if self.lowl else 0
        )

if __name__=='__main__':

    parser = argparse.ArgumentParser(prog='planck_reio')
    parser.add_argument('--model', default='lcdm_tau', help='lcdm_[tau|reiomodes]')
    args = parser.parse_args()

    assert args.model in ['lcdm_tau','lcdm_reiomodes']

    p=Slik(planck(model=args.model))
    for _ in p.sample(): pass

