from cosmoslik import *
from numpy import *
import os, os.path as osp
from cosmoslik_plugins.likelihoods.clik import clik
from cosmoslik.subprocess_plugins import SubprocessClassDied
import argparse
import camb
camb.ignore_fatal_errors.value = True

param = param_shortcut('start','scale')

# @subprocess_class
class CambReio(SlikPlugin):
    """
    CAMB with reionization eigenmodes
    """

    camb_params = ['As','ns','ombh2','omch2','cosmomc_theta','pivot_scalar','mnu','tau']

    def __init__(self,lmax=5000):
        super(CambReio,self).__init__(lmax=lmax)

        #load eigenmodes and smoothly transition the mode to Xe(z)=0 at high z
        #with a cosine window
        self.z,_ = loadtxt("xefid.dat").T
        dl = 20
        w = hstack([ones(len(self.z)-dl),(cos(pi*arange(dl)/(dl-1))+1)/2])
        self.modes = loadtxt("xepcs.dat")[1:]*w

        #compute fiducial Xe around which we perturb
        cp = camb.set_params(As=1e-9)
        camb.get_background(cp)
        self.fidxe = cp.Reion.get_xe(z=self.z)


    def __call__(self,**params):

        cp = camb.set_params(lmax=self.lmax,H0=None,**{k:params[k] for k in self.camb_params if k in params})
        cp.k_eta_max_scalar = 2*self.lmax
        cp.DoLensing = True
        cp.NonLinear = 0
        if 'reiomodes' in params:
            self.xe = self.fidxe+sum(params['reiomodes']['mode%i'%i]*self.modes[i]
                                     for i in range(95) if 'mode%i'%i in params['reiomodes'])
            cp.Reion.set_xe(z=self.z,xe=self.xe)
        else:
            self.xe = cp.Reion.get_xe(z=self.z)
        r = self.results = camb.get_results(cp)

        return dict(zip(['cl_%s'%x for x in ['TT','EE','BB','TE']],
                        (cp.TCMB*1e6)**2*r.get_cmb_power_spectra(spectra=['total'])['total'].T))


@SlikMain
class planck(SlikPlugin):

    def __init__(self, model='lcdm_tau',nmodes=5):

        super(planck,self).__init__(**all_kw(locals()))

        self.cosmo = SlikDict(
            logA = param(3.108,0.03),
            ns = param(0.962,0.006),
            ombh2 = param(0.02221,0.0002),
            omch2 = param(0.1203,0.002),
            theta = param(0.0104,0.00003),
            pivot_scalar = 0.05,
            mnu = param(0.06,0.1,min=0) if 'mnu' in model else 0.06
        )
        if 'tau' in model:
            self.cosmo.tau = param(0.085,0.01,min=0,gaussian_prior=(0.07,0.01))
        elif 'reiomodes' in model:
            self.cosmo.reiomodes = SlikDict()
            for i in range(nmodes):
                self.cosmo.reiomodes['mode%i'%i] = param(0,0.005)

        self.camb = CambReio()

        self.calPlanck = param(1,0.0025,gaussian_prior=(1,0.0025))

        self.highl = clik(clik_file='plik_lite_v18_TT.clik')
        self.lowlT = clik(clik_file='commander_rc2_v1.1_l2_29_B.clik')
        self.lowlP = clik(clik_file='simlow_MA_EE_2_32_2016_03_31.clik',auto_reject_errors=True)

        self.priors = get_plugin('likelihoods.priors')(self)

        run_id = [model]
        if 'reiomodes' in model: run_id.append('nmodes%i'%nmodes)

        self.sampler = get_plugin('samplers.metropolis_hastings')(
            self,
            num_samples = 1e7,
            print_level = 3,
            proposal_update_start = 500,
            mpi_comm_freq = 10,
            output_file = 'chains/chain_'+'_'.join(run_id),
            proposal_cov = args.covmat or 'planck_%s.covmat'%model,
            output_extra_params = [('camb.xe','(95,)d'), 'lnls.priors','lnls.highl','lnls.lowlT','lnls.lowlP',
                                   ('clTT','(100,)d'),('clTE','(100,)d'),('clEE','(100,)d')]
       )


    def __call__(self):
        self.cosmo.As = exp(self.cosmo.logA)*1e-10
        self.cosmo.cosmomc_theta = self.cosmo.theta
        self.highl.A_Planck = self.highl.calPlanck = self.calPlanck
        if self.lowlT: self.lowlT.A_planck = self.calPlanck

        self.lnls = SlikDict({k:nan for k in ['priors','highl','lowlT','lowlP']})

        try:
            self.cls = self.camb(**self.cosmo)
            if (any([any(isnan(x)) for x in self.cls.values()])
                or any([any(x>1e5) for x in self.cls.values()])):
                raise Exception("CAMB outputted crazy Cls")
        except Exception as e:
            print "Warning, rejecting step: "+str(e)
            return inf

        self.clTT = self.cls['cl_TT'][:100]
        self.clTE = self.cls['cl_TE'][:100]
        self.clEE = self.cls['cl_EE'][:100]

        return lsumk(self.lnls,
            [('priors',lambda: self.priors(self)),
             ('highl',lambda: self.highl(self.cls)),
             ('lowlT',lambda: self.lowlT(self.cls)),
             ('lowlP',lambda: self.lowlP(self.cls))]
        )

if __name__=='__main__':

    parser = argparse.ArgumentParser(prog='planck_reio')
    parser.add_argument('--model', default='lcdm_tau', help='lcdm_[tau|reiomodes]')
    parser.add_argument('--nmodes', default=5, type=int, help='number of reionization eigenmodes')
    parser.add_argument('--covmat',help='covmat file')
    args = parser.parse_args()

    assert all([x in ['lcdm','mnu','tau','reiomodes'] for x in args.model.split('_')]), "Unrecognized model"

    p=Slik(planck(model=args.model,nmodes=args.nmodes))
    for _ in p.sample(): pass
