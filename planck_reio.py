from cosmoslik import *
from numpy import *
import os, os.path as osp
from cosmoslik_plugins.likelihoods.clik import clik
from cosmoslik.subprocess_plugins import SubprocessClassDied
import argparse
import camb
camb.ignore_fatal_errors.value = True

# from matplotlib.pyplot import ion, plot, draw
# ion()

param = param_shortcut('start','scale')

class BadXe(Exception): pass

# @subprocess_class
class CambReio(SlikPlugin):
    """
    CAMB with reionization eigenmodes
    """

    camb_params = ['As','ns','ombh2','omch2','cosmomc_theta','pivot_scalar','mnu','tau']

    def __init__(self,lmax=5000, DoLensing=True):
        super(CambReio,self).__init__(lmax=lmax,DoLensing=DoLensing)

        #load eigenmodes
        dat = loadtxt("myxepcs.dat")
        self.z, self.modes = dat[0], dat[1:]

        #compute fiducial Xe around which we perturb
        cp = camb.set_params(As=1e-9)
        camb.get_background(cp)
        self.fidxe = cp.Reion.get_xe(z=self.z)


    def __call__(self,**params):

        cp = camb.set_params(lmax=self.lmax,H0=None,**{k:params[k] for k in self.camb_params if k in params})
        cp.k_eta_max_scalar = 2*self.lmax
        cp.DoLensing = self.DoLensing
        cp.NonLinear = 0
        if 'reiomodes' in params:
            self.xe = self.fidxe+sum(params['reiomodes']['mode%i'%i]*self.modes[i]
                                     for i in range(95) if 'mode%i'%i in params['reiomodes'])
            if any(self.xe<-1) or any(self.xe>2): raise BadXe()
            # plot(self.z,self.xe)
            # draw()
            cp.Reion.set_xe(z=self.z,xe=self.xe)
        else:
            self.xe = cp.Reion.get_xe(z=self.z)
        self.xe = self.xe[::10] #thin to keep chain size down
        r = self.results = camb.get_results(cp)

        return dict(zip(['cl_%s'%x for x in ['TT','EE','BB','TE']],
                        (cp.TCMB*1e6)**2*r.get_total_cls(self.lmax).T))


@SlikMain
class planck(SlikPlugin):

    def __init__(self, model='lcdm_tau', only_lowp=False, nmodes=5, lowl='simlow'):

        super(planck,self).__init__(**all_kw(locals()))

        if only_lowp:
            self.cosmo = SlikDict(
                logA = param(3.108,0.03,min=2.8,max=3.4),
                ns = 0.962,
                ombh2 = 0.02221,
                omch2 = 0.1203,
                theta = 0.0104,
                pivot_scalar = 0.05,
            )
        else:
            self.cosmo = SlikDict(
                logA = param(3.108,0.03),
                ns = param(0.962,0.006),
                ombh2 = param(0.02221,0.0002),
                omch2 = param(0.1203,0.002),
                theta = param(0.0104,0.00003),
                pivot_scalar = 0.05,
                mnu = param(0.06,0.1,min=0) if 'mnu' in model else 0.06,
                ALens = param(1,0.02) if 'ALens' in model else 1
            )
        if 'tau' in model:
            self.cosmo.tau = param(0.085,0.01,min=0.04)
        elif 'reiomodes' in model:
            self.cosmo.reiomodes = SlikDict()
            for i in range(nmodes):
                self.cosmo.reiomodes['mode%i'%i] = param(0,0.005)

        self.camb = CambReio(lmax=200 if only_lowp else 5000,
                             DoLensing=(not only_lowp))

        if not only_lowp:
            self.calPlanck = param(1,0.0025,gaussian_prior=(1,0.0025))
        else:
            self.calPlanck = 1

        if not only_lowp:
            self.highl = clik(clik_file='plik_lite_v18_TT.clik')
            self.lowlT = clik(clik_file='commander_rc2_v1.1_l2_29_B.clik')
            
        if lowl=='simlow':
            self.lowlP = clik(clik_file='simlow_MA_EE_2_32_2016_03_31.clik',auto_reject_errors=True)
        else:
            self.lowlP = clik(clik_file='/redtruck/benabed/release/clik_10.3/low_l/bflike/lowl_QU_70_dx11d_2014_10_03_v5c_Ap.clik/',
                              auto_reject_errors=True)

        self.priors = get_plugin('likelihoods.priors')(self)

        run_id = [model]
        if 'reiomodes' in model: run_id.append('nmodes%i'%nmodes)
        if self.only_lowp: run_id.append('onlylowp')
        run_id.append(lowl)

        self.sampler = get_plugin('samplers.metropolis_hastings')(
            self,
            num_samples = 1e7,
            print_level = 3,
            proposal_update_start = 5000,
            mpi_comm_freq = 10,
            output_file = 'chains/chain_'+'_'.join(run_id),
            proposal_cov = args.covmat or 'planck_%s.covmat'%model,
            output_extra_params = [('camb.xe','(103,)d'), 'lnls.priors','lnls.highl','lnls.lowlT','lnls.lowlP',
                                   ('clTT','(100,)d'),('clTE','(100,)d'),('clEE','(100,)d'),
                                   'cosmo.tau']
       )


    def __call__(self):
        self.cosmo.As = exp(self.cosmo.logA)*1e-10
        self.cosmo.cosmomc_theta = self.cosmo.theta
        if not self.only_lowp:
            self.highl.A_Planck = self.highl.calPlanck = self.calPlanck
            if self.lowlT: self.lowlT.A_planck = self.calPlanck
        self.lowlP.A_planck = self.calPlanck
        
        self.lnls = SlikDict({k:nan for k in ['priors','highl','lowlT','lowlP']})

        try:
            self.cls = self.camb(**self.cosmo)
            if (any([any(isnan(x)) for x in self.cls.values()])
                or any([any(x>1e5) for x in self.cls.values()])):
                raise Exception("CAMB outputted crazy Cls")
        except Exception as e:
            if not isinstance(e,BadXe):
                print "Warning, rejecting step: "+str(e)
            return inf

        self.clTT = self.cls['cl_TT'][:100]
        self.clTE = self.cls['cl_TE'][:100]
        self.clEE = self.cls['cl_EE'][:100]
        if 'tau' not in self.cosmo: self.cosmo.tau = self.camb.results.get_tau()

        return lsumk(self.lnls,
            [('priors',lambda: self.priors(self)),
             ('highl',lambda: 0 if self.only_lowp else self.highl(self.cls)),
             ('lowlT',lambda: 0 if self.only_lowp else self.lowlT(self.cls)),
             ('lowlP',lambda: self.lowlP(self.cls))]
        )

if __name__=='__main__':

    parser = argparse.ArgumentParser(prog='planck_reio')
    parser.add_argument('--model', default='lcdm_tau', help='lcdm_[tau|reiomodes]')
    parser.add_argument('--lowl', default='simlow', help='simlow|bflike')
    parser.add_argument('--nmodes', default=5, type=int, help='number of reionization eigenmodes')
    parser.add_argument('--covmat',help='covmat file')
    parser.add_argument('--only_lowp', action='store_true')
    args = parser.parse_args()

    assert all([x in ['lcdm','mnu','tau','reiomodes','ALens'] for x in args.model.split('_')]), "Unrecognized model"
    assert args.lowl in ['simlow','bflike'], "Unrecognized lowl likelihood"

    p=Slik(planck(model=args.model,nmodes=args.nmodes,only_lowp=args.only_lowp,lowl=args.lowl))
    for _ in p.sample(): pass
