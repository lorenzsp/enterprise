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
gamma_gw = parameter.Uniform(0,7)('gamma_gw')

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

mono = gp_signals.FourierBasisCommonGP(cpl, utils.monopole_orf(),
                                      components=15, Tspan=Tspan, name='mono')


# to add solar system ephemeris modeling...
bayesephem=False
if bayesephem:
    eph = deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

# timing model
tm = gp_signals.TimingModel(use_svd=True)

# full model
if bayesephem:
    s = ef + eq + ec + rn + tm + eph + gw # + mono
    # additional
    rn_s = ef + eq + ec + rn + eph + gw # + mono
    gwonly = ef + eq + ec + eph + gw # + mono
else:
    s = ef + eq + ec + rn + tm + gw # + mono
    # additional
    rn_s = ef + eq + ec + rn + gw # + mono
    gwonly = ef + eq + ec + gw # + mono

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
num = 3
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

        tic = time.time()
        flf_flr_rLr = self.pta.get_FLF_FLr_dtLdt_rNr(params)
        TNTs = [ell[0] for ell in flf_flr_rLr]
        TNrs = [ell[1] for ell in flf_flr_rLr]
        loglike += -0.5 * np.sum([ell[2] for ell in flf_flr_rLr])
        del flf_flr_rLr
        toc = time.time()
        print("TNrs TNTs", toc - tic)
        tic = time.time()
        # this function makes sure that we can get directly a sparse matrix
        phiinvs = pta_gw.get_phiinv_byfreq_cliques(params, logdet=True, chol=True)#, phi_input=totPhi, chol=True)
        # phi_test = pta_gw.get_phi(params, cliques=True, chol=True)
        # Nf = len(TNTs[0])
        # Npsr = len(TNTs)
        # out = np.linalg.inv([phi_test[slice(i, Npsr*Nf, Nf), slice(i, Npsr*Nf, Nf)].toarray() for i in range(Nf)])
        # logdet_phi = np.sum( np.linalg.slogdet([phi_test[slice(i, Npsr*Nf, Nf), slice(i, Npsr*Nf, Nf)].toarray() for i in range(Nf)])[1] )
        toc = time.time()
        print("phi inv", toc - tic)
        # newinv = sps.bmat([[ sps.diags(out[:,A,B],format="csc") for A in range(Npsr)]for B in range(Npsr)],format="csc")
        
        # get extra prior/likelihoods
        loglike += sum(self.pta.get_logsignalprior(params))

        # red noise piece
        if self.pta._commonsignals:
            tic = time.time()
            phiinv, logdet_phi = phiinvs #  newinv, np.sum(out_ld[1])#

            TNT = self._block_TNT(TNTs)
            TNr = self._block_TNr(TNrs)

            # logdet_phi = 0.0
            # for i in range(Nf):
            #     # current_cf = cholesky(phi_test[slice(i, Npsr*Nf, Nf), slice(i, Npsr*Nf, Nf)])
            #     # logdet_phi += current_cf.logdet()
            #     TNT[slice(i, Npsr*Nf, Nf), slice(i, Npsr*Nf, Nf)] += sps.csc_matrix(out[i])

            toc = time.time()
            print("prepare to cholesky", toc-tic)

            if self.cholesky_sparse:
                try:
                    # breakpoint() 
                    tic = time.time()
                    # ------------------------------------------
                    # tmp_phi = pta_gw.get_phi(params, chol=True, cliques=True)
                    # U,S,V =  sps.linalg.svds(tmp_phi,k=100)
                    # tmp_Sigma = U.T @ TNT @ U+ np.diag(1/S)
                    # right = U.T @ TNr
                    # res = right @ np.linalg.solve(tmp_Sigma, right)
                    # expval = tmp_cf(b_vec)
                    # logdet_sigma = tmp_cf.logdet()
                    # ------------------------------------------
                    
                    # plt.figure(); plt.imshow((tmp_Sigma.toarray()!=0.0)*1.0);plt.colorbar(); plt.savefig("matrix.pdf");
                    Sigma = TNT + phiinv
                    cf = cholesky(Sigma, ordering_method='natural', mode='supernodal') #,'natural','amd','colamd' use_long=False)  # cf(Sigma)
                    expval = cf(TNr)
                    logdet_sigma = cf.logdet()
                    
                    toc = time.time()
                    print("do cholesky", toc-tic)

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