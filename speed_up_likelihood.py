# This script shows how to evaluate the likelihood of a real dataset. This is based on https://github.com/nanograv/12p5yr_stochastic_analysis/blob/master/notebooks/pta_gwb_analysis.ipynb
import os, glob, json, pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sl

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const

import corner
import multiprocessing
import time
import _pickle as cPickle

psrlist = None # define a list of pulsar name strings that can be used to filter.
# set the data directory
datadir = './data'

print(datadir)

pickle_file = True
####################################################################################
# Here it shows how to import the data, but I created a pickle object to import it quickly
if pickle_file==False:
    # for the entire pta
    parfiles = sorted(glob.glob(datadir + '/par/*par'))
    timfiles = sorted(glob.glob(datadir + '/tim/*tim'))

    # filter
    if psrlist is not None:
        parfiles = [x for x in parfiles if x.split('/')[-1].split('.')[0] in psrlist]
        timfiles = [x for x in timfiles if x.split('/')[-1].split('.')[0] in psrlist]

    # Make sure you use the tempo2 parfile for J1713+0747!!
    # ...filtering out the tempo parfile... 
    parfiles = [x for x in parfiles if 'J1713+0747_NANOGrav_12yv3.gls.par' not in x]

    psrs = []
    ephemeris = 'DE438'
    for p, t in zip(parfiles, timfiles):
        psr = Pulsar(p, t, ephem=ephemeris)
        psrs.append(psr)

    filename = "psrs_obj.pkl"
    with open(filename, "wb") as output_file:
        cPickle.dump(psrs, output_file)

####################################################################################
# pickle object
else:
    with open(datadir + "/psrs_obj.pkl", "rb") as input_file:
        psrs = cPickle.load(input_file)

## Get parameter noise dictionary
noise_ng12 = datadir + '/channelized_12p5yr_v3_full_noisedict.json'

params = {}
with open(noise_ng12, 'r') as fp:
    params.update(json.load(fp))

####################################################################################
# here it is defined the model for the likelihood
# find the maximum time span to set GW frequency sampling
tmin = [p.toas.min() for p in psrs]
tmax = [p.toas.max() for p in psrs]
Tspan = np.max(tmax) - np.min(tmin)

# define selection by observing backend
selection = selections.Selection(selections.by_backend)

# white noise parameters
efac = parameter.Constant() 
equad = parameter.Constant() 
ecorr = parameter.Constant() # we'll set these later with the params dictionary

# red noise parameters
log10_A = parameter.Uniform(-20, -11)
gamma = parameter.Uniform(0, 7)

# dm-variation parameters
log10_A_dm = parameter.Uniform(-20, -11)
gamma_dm = parameter.Uniform(0, 7)

# GW parameters (initialize with names here to use parameters in common across pulsars)
log10_A_gw = parameter.Uniform(-18,-14)('log10_A_gw')
gamma_gw = parameter.Constant(4.33)('gamma_gw')

# white noise
ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)

# red noise (powerlaw with 30 frequencies)
pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)

# for spatial correlations you can do...
# spatial correlations are covered in the hypermodel context later
orf = utils.hd_orf()
cpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
gw = gp_signals.FourierBasisCommonGP(cpl, orf,
                                      components=30, Tspan=Tspan, name='gw')


# to add solar system ephemeris modeling...
bayesephem=False
if bayesephem:
    eph = deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

# timing model
tm = gp_signals.TimingModel(use_svd=True)

# full model
if bayesephem:
    s = ef + eq + ec + rn + tm + eph + gw
else:
    s = ef + eq + ec + rn + tm + gw

rn_s = ef + eq + ec + rn + gw
gwonly = ef + eq + ec + gw
# intialize PTA (this cell will take a minute or two to run)
models = []
models_gw = []
models_gwonly = []
for p in psrs:    
    models.append(s(p))
    models_gw.append(rn_s(p))
    models_gwonly.append(gwonly(p))

pta = signal_base.PTA(models)
pta_gw = signal_base.PTA(models_gw)
pta_gwonly = signal_base.PTA(models_gwonly)

# set white noise parameters with dictionary
pta.set_default_params(params)
pta_gw.set_default_params(params)
pta_gwonly.set_default_params(params)
##############################################################################################
# Here we calculate the likelihood
np.random.seed(42)

# draw "num" possible parameters
num = 10
x0 = [np.hstack([p.sample() for p in pta.params]) for i in range(num)]

# evaluate likelihood, I also print the values to check they are not infinite
pta.get_lnlikelihood(x0[0])
print("start timing")
tic = time.perf_counter()
old = np.asarray([pta.get_lnlikelihood(xx) for xx in x0])
toc = time.perf_counter()
print("the likelihood evaluation took: ",(toc-tic)/num, "seconds, number of pulsars", len(psrs))
print(old)
#######################################################################################
# Here we break down the timing into different pieces
import scipy.sparse as sps
from enterprise.signals.signal_base import simplememobyid
from sksparse.cholmod import cholesky, CholmodError


class LogLikelihoodLocal(object):
    def __init__(self, pta, cholesky_sparse=True):
        self.pta = pta
        self.cholesky_sparse = cholesky_sparse

    @simplememobyid
    def _block_TNT(self, TNTs):
        if self.cholesky_sparse:
            return sps.block_diag(TNTs, "csc")
        else:
            return sl.block_diag(*TNTs)

    @simplememobyid
    def _block_TNr(self, TNrs):
        return np.concatenate(TNrs)

    def __call__(self, xs, phiinv_method="cliques"):
        
        # map parameter vector if needed
        params = xs if isinstance(xs, dict) else self.pta.map_params(xs)

        loglike = 0

        # phiinvs will be a list or may be a big matrix if spatially
        # correlated signals
        tic = time.time()
        # TNrs = self.pta.get_FLr(params)
        # TNTs = self.pta.get_FLF(params)
        flf_flr_rLr = self.pta.get_FLF_FLr_dtLdt_rNr(params)
        TNTs = [ell[0] for ell in flf_flr_rLr]
        TNrs = [ell[1] for ell in flf_flr_rLr]
        loglike += -0.5 * np.sum([ell[2] for ell in flf_flr_rLr])
        del flf_flr_rLr
        toc = time.time()
        print("TNrs TNTs", toc - tic)
        # tic = time.time()
        # ------------------------------------------------------------------------------------------
        # # new get phi
        # # uncorrelated matrix
        # phis = [signalcollection.get_phi(params) for signalcollection in pta_gw._signalcollections]
        # # # correlation between pulsars
        # # Gamma = np.asarray([[utils.hd_orf(psrs[i].pos, psrs[j].pos) for i in range(len(psrs))] for j in range(len(psrs))])
        # # Gamma[range(len(Gamma)),range(len(Gamma))] = 0.0
        # Gamma = np.zeros((len(phis),len(phis)) )
        # Gamma[np.triu_indices(len(phis),k=1)] += np.asarray([utils.hd_orf(psrs[i].pos, psrs[j].pos) for i in range(len(psrs)) for j in range(len(psrs)) if j>i])
        # Gamma += Gamma.T
        # # # base background spectrum
        # rho_1psr = pta_gwonly.signals['J2317+1439_gw'].get_phi(params)
        # base_phi = np.zeros_like(phis[0])
        # base_phi[:len(rho_1psr)] = rho_1psr
        # # toc = time.time()
        # # print("phi", toc - tic)
        # # # create off terms
        # offPhi = sps.kron(Gamma, np.diag(base_phi),"csc")
        # # # get final
        # totPhi = offPhi + sps.block_diag([np.diag(pp) for pp in phis],"csc")

        # # get a dictionary of slices locating each pulsar in Phi matrix
        # slices = pta_gw._get_slices(phis)
        # pta_gw._resetcliques(totPhi.shape[0])
        # pta_gw._setpulsarcliques(slices, phis)

        # # iterate over all common signal classes
        # for csclass, csdict in pta_gw._commonsignals.items():
        #     # first figure out which indices are used in this common signal
        #     # and update the clique index
        #     pta_gw._setcliques(slices, csdict)
        
        # assert np.all(totPhi.toarray()==pta_gw.get_phi(params))
        # ------------------------------------------------------------------------------------------
        
        # tic = time.time()
        # this function makes sure that we can get directly a sparse matrix
        phiinvs = pta_gw.get_phiinv_byfreq_cliques(params, logdet=True, chol=False)#, phi_input=totPhi, chol=True)
        # toc = time.time()
        # print("phi inv", toc - tic)
        # get -0.5 * (rNr + logdet_N) piece of likelihood
        # the np.sum here is needed because each pulsar returns a 2-tuple
        # loglike += -0.5 * np.sum([ell for ell in pta.get_dtLdt(params)])
        # loglike += -0.5 * np.sum([ell for ell in self.pta.get_rNr_logdet(params)])

        # get extra prior/likelihoods
        loglike += sum(self.pta.get_logsignalprior(params))

        # red noise piece
        if self.pta._commonsignals:
            # tic = time.time()
            phiinv, logdet_phi = phiinvs

            TNT = self._block_TNT(TNTs)
            TNr = self._block_TNr(TNrs)
            # toc = time.time()
            # print("prepare to cholesky", toc-tic)

            if self.cholesky_sparse:
                # A = sps.block_diag([np.diag(1.0/pp) for pp in phis],"csc")
                # B = offPhi.copy()
                # S0 = - A @ B @ A
                # totalS = A + S0
                # for i in range(15):
                #     S1 = - A @ B @ S0
                #     totalS += S1.copy()
                #     S0 = S1.copy()
                try:
                    # tic = time.time()
                    
                    Sigma = TNT + sps.csc_matrix(phiinv) #totalS#
                    # breakpoint()
                    # plt.figure(); plt.imshow((Sigma.toarray()!=0.0)*1.0);plt.colorbar(); plt.savefig("matrix.pdf");
                    
                    cf = cholesky(Sigma, ordering_method='natural', mode='supernodal') #,'natural','amd','colamd' use_long=False)  # cf(Sigma)
                    expval = cf(TNr)
                    logdet_sigma = cf.logdet()
                    
                    # toc = time.time()
                    # print("do cholesky", toc-tic)
                    ######################################
                    
                    # # newTNT = TNT  + sps.block_diag([np.diag(pp) for pp in phis],"csc")
                    # FLFm1_FLr = np.linalg.solve(TNTs,TNrs)
                    # sum_FLr_FLFm1_FLr = np.sum([np.dot(fl, tnrs) for tnrs,fl in zip(TNrs,FLFm1_FLr)])
                    
                    # inv_FLF = np.linalg.inv(TNTs)
                    # # u_s_v= [np.linalg.svd(tnt)for tnt in TNTs]
                    # # inv_FLF = [np.dot(v.transpose(),np.dot(np.diag(s**-1),u.transpose())) for u,s,v in u_s_v]
                    # # det_sum  = np.sum([np.log(s) for u,s,v in u_s_v])

                    # newSigma = TNT @ totPhi @ TNT + TNT
                    # newcf = cholesky(newSigma)
                    # newFLr = self._block_TNr(TNr)
                    # expval2 = newcf(newFLr)
                    # logdet_sigma = newcf.logdet() + det_sum
                    # sum_FLr_FLFm1_Sigmam1_FLFm1_FLr = TNr.T @ sps.linalg.factorized(TNT + TNT @ sps.linalg.inv(phiinv) @ TNT)(TNr)# np.dot(expval2,newFLr)
                    # print(sum_FLr_FLFm1_Sigmam1_FLFm1_FLr)
                    
                    # # invnew = newcf.inv()
                    # invTNT = self._block_TNT(inv_FLF)#sps.linalg.pinv(TNT)
                    # # test1 = invTNT - sps.linalg.inv(TNT + TNT @ sps.linalg.inv(phiinv) @ TNT ) 
                    # # test2 = totPhi - totPhi @ (invTNT + phiinv) @ totPhi
                    # test1 - cf.inv() .toarray()
                    # print(TNr.T @ invTNT @ TNr - TNr.T @ sps.linalg.inv(TNT + TNT @ sps.linalg.inv(phiinv) @ TNT ) @ TNr)


                except CholmodError:  # pragma: no cover
                    return -np.inf
            else:
                try:
                    cf = sl.cho_factor(TNT + phiinv)  # cf(Sigma)
                    expval = sl.cho_solve(cf, TNr)
                    logdet_sigma = 2 * np.sum(np.log(np.diag(cf[0])))
                except sl.LinAlgError:  # pragma: no cover
                    return -np.inf

            # tic = time.time()
            loglike += -0.5 * (-np.dot(TNr, expval) + logdet_sigma + logdet_phi)
            # loglike += -0.5 * (-total + logdet_sigma + logdet_phi)
            # loglike += 0.5 * (np.dot(TNr, expval) - logdet_sigma - logdet_phi)
            # toc = time.time()
            # print("final computation", toc-tic)
        else:
            for TNr, TNT, pl in zip(TNrs, TNTs, phiinvs):
                if TNr is None:
                    continue

                phiinv, logdet_phi = pl
                Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

                try:
                    cf = sl.cho_factor(Sigma)
                    expval = sl.cho_solve(cf, TNr)
                except sl.LinAlgError:  # pragma: no cover
                    return -np.inf

                logdet_sigma = np.sum(2 * np.log(np.diag(cf[0])))

                loglike += 0.5 * (np.dot(TNr, expval) - logdet_sigma - logdet_phi)

        return loglike

# check each part
loglike = LogLikelihoodLocal(pta)
loglike(x0[0])
print("start timing")
tic = time.perf_counter()
new = np.asarray([loglike(xx) for xx in x0])
toc = time.perf_counter()
print("the likelihood evaluation took: ",(toc-tic)/num, "seconds, number of pulsars", len(psrs))

print("difference old likelihood",new-old, "proporional to ln(1e40)")
print(new)
#######################################################################################
# my output on my local laptop
"""
Warning: cannot find astropy, units support will not be available.
./data
start timing
the likelihood evaluation took:  1.2899806030094623 seconds, number of pulsars 45
[4889115.79706147 4888958.18703743 4889320.36953868 4889618.10010002
 4889434.77207024 4885037.71101035 4889419.02641993 4889265.41208175
 4889427.31364503 4886254.31924279]
start timing
the likelihood evaluation took:  0.39184193019755187 seconds, number of pulsars 45
difference old likelihood [246652.91515956 246652.91516962 246652.91516175 246652.91516039
 246652.91516251 246652.91515868 246652.91516194 246652.91515075
 246652.91515516 246652.91516152] proporional to ln(1e40)
"""