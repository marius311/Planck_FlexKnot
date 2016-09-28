# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>


import numpy as np
from scipy.integrate import quad 
import math
import camb


def Fisher_BAO():
    #lss1 = np.loadtxt('boss_10000_0.05_0.8_0.1_1_bao_0.expt', usecols=(0,1,2,3))
    #lss2 = np.loadtxt('boss_2.1_4_1_22_0.73_10000_200_1_900_3600_3_0_985_1200qc.expt', usecols=(0,1,2,3))
    lss1 = np.loadtxt('desilz_14000_0.65_1.9_0.1_1_ELG1440_3_bao_0.expt',usecols=(0,1,2,3))
    lss2 = np.loadtxt('desi_2.1_4.2_1_23_1.09396_14000_200_1_1000_3501_4_0_985_1200qc.expt',usecols=(0,1,2,3))
    lss1[:,1] = 0.01*lss1[:,1]    #right scaling
    lss1[:,2] = 0.01*lss1[:,2]
    
    lss  = np.concatenate((lss1, lss2), axis=0)
    Num_z = lss[:,0].size
    redz  = lss[:,0]
    
    Covz  = np.zeros([Num_obs, Num_obs])
    Cov   = np.zeros([Num_z, Num_obs, Num_obs])
    fish  = np.zeros([Num_z, Num_obs, Num_obs])
    
    for i in range(Num_z):
        Covz[0][0] = (lss[i,1])**2                  #(sigma_(D/rs)/(D/rs))^2
        Covz[1][1] = (lss[i,2])**2                  #(sigma_H rs/H rs)^2
        Covz[0][1] =  lss[i,1]*lss[i,2]*lss[i,3]    #r*(sigma_(D/rs)/(D/rs))*(sigma_H rs/H rs)
        Covz[1][0] = Covz[0][1]   
        Cov[i,:,:] = Covz
        fish[i,:,:]= np.linalg.inv(Covz)
    return Num_z, redz, fish, Cov

# <codecell>
def rsstar_h(par):
    r = camb.get_background(camb.set_params(
        As=par[0]*1.e-9,
        ns=par[1],
        tau=par[2],
        ombh2=par[4],
        omch2=par[5],
        mnu=par[6],
        H0=None,
        cosmomc_theta=par[3]
    ))
    return r.get_derived_params()['rdrag'], r.Params.H0/100

def eta(x, omgbc, omgl, omgnu):
    h = hubble(x, omgbc, omgl, omgnu)
    return 3000./h                                     #1/H(z) in unit of Mpc

def hubble(x, omgbc, omgl, omgnu):
    omgr = (2.47+1.12)*1.e-5                           #photons + 2 massless neutrinos
    omgm = omgbc + omgnu                               #baryons + CDM + 1 massive neutrino
    return math.sqrt(omgm*(1+x)**3+omgr*(1+x)**4+omgl) #H(z) in unit of 100 km/s/Mpc
    
def observ(par, redz):                                 #return DA(z)/rs and H(z)*rs
    rs,h  = rsstar_h(par)
    omgbc = par[4] + par[5]
    omgnu = par[6]/94.
    omgl  = h**2-(omgbc+omgnu) 
    obs   = np.zeros([Num_z, 2])
    for i in range(Num_z):
        obs[i,0] = quad(eta, 0, redz[i], args=(omgbc, omgl, omgnu))[0]/rs
        obs[i,1] = hubble(redz[i], omgbc, omgl, omgnu)*rs
    return obs

def finite_diff(par, redz, delta):
    deriver = np.zeros([Num_z, Num_par, Num_obs])
    
    for i in range(Num_par):
        par_r    = [x for x in par]
        par_l    = [x for x in par]
        par_r[i] = par[i]*(1.+delta)
        par_l[i] = par[i]*(1.-delta)

        obs_r  = observ(par_r, redz)
        obs_l  = observ(par_l, redz)
        deriver[:,i, 0] = (obs_r[:,0]/obs_l[:,0] -1.)/(par[i]*delta*2)
        deriver[:,i, 1] = (obs_r[:,1]/obs_l[:,1] -1.)/(par[i]*delta*2)
    return deriver


Num_obs   = 2              #DA(z)/rs and H(z)*rs
Num_par   = 7              
Num_z, redz, fish_bao, Cov_bao = Fisher_BAO()

#       As      ns      tau     theta    omgb    omgc      mnu 
par  = [2.215, 0.9619, 0.0925, 0.0104, 0.022068, 0.12, 0.085]
delta = 0.05                       #stepsize of finite difference


deriver   = finite_diff(par, redz, delta)
fish_par  = np.zeros([Num_par, Num_par])
F_BAO     = np.zeros([7,7])
   
for i in range(Num_z):
    fish_z   = np.dot(np.dot(deriver[i,:,:], fish_bao[i,:,:]), np.transpose(deriver[i,:,:]))
    fish_par = fish_par + fish_z
    print("z_i = ", redz[i])
    
np.savetxt('F_DESI.dat', fish_par)
