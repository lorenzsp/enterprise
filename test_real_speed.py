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

# intialize PTA (this cell will take a minute or two to run)
models = []
        
for p in psrs:    
    models.append(s(p))
    
pta = signal_base.PTA(models)

# set white noise parameters with dictionary
pta.set_default_params(params)

##############################################################################################
# Here we calculate the likelihood
np.random.seed(42)

# draw "num" possible parameters
num = 10
x0 = [np.hstack([p.sample() for p in pta.params]) for i in range(num)]

# evaluate likelihood, I also print the values to check they are not infinite
print("start timing")
tic = time.perf_counter()
print([pta.get_lnlikelihood(xx) for xx in x0])
toc = time.perf_counter()
print("the likelihood evaluation took: ",(toc-tic)/num, "seconds, number of pulsars", len(psrs))

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
        TNrs = self.pta.get_TNr(params)
        TNTs = self.pta.get_TNT(params)
        phiinvs = self.pta.get_phiinv(params, logdet=True, method=phiinv_method)
        
        
        # get -0.5 * (rNr + logdet_N) piece of likelihood
        # the np.sum here is needed because each pulsar returns a 2-tuple
        loglike += -0.5 * np.sum([ell for ell in self.pta.get_rNr_logdet(params)])

        # get extra prior/likelihoods
        loglike += sum(self.pta.get_logsignalprior(params))
        toc = time.time()
        print("first part", toc - tic)

        # red noise piece
        if self.pta._commonsignals:
            tic = time.time()
            phiinv, logdet_phi = phiinvs

            TNT = self._block_TNT(TNTs)
            TNr = self._block_TNr(TNrs)
            toc = time.time()
            print("prepare to cholesky", toc-tic)
            if self.cholesky_sparse:
                try:
                    tic = time.time()
                    cf = cholesky(TNT + sps.csc_matrix(phiinv))  # cf(Sigma)
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

            tic = time.time()
            loglike += 0.5 * (np.dot(TNr, expval) - logdet_sigma - logdet_phi)
            toc = time.time()
            print("final computation", toc-tic)
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
print([loglike(xx) for xx in x0])

#######################################################################################
# my output on my local laptop
"""
Warning: cannot find astropy, units support will not be available.
./data
start timing
[4889283.072539778, 4889034.559907954, 4889407.334923696, 4889648.942570506, 4889598.006753816, 4885220.336408644, 4889456.655002485, 4889341.636537235, 4889584.523676748, 4886285.761601724]
the likelihood evaluation took:  2.1442143232008677 seconds, number of pulsars 45
first part 0.1783900260925293
prepare to cholesky 0.07949709892272949
do cholesky 0.8084447383880615
final computation 2.4080276489257812e-05
first part 0.1932361125946045
prepare to cholesky 2.6226043701171875e-05
do cholesky 0.8444170951843262
final computation 2.09808349609375e-05
first part 0.18991684913635254
prepare to cholesky 3.0040740966796875e-05
do cholesky 0.8453202247619629
final computation 3.3855438232421875e-05
first part 0.1865549087524414
prepare to cholesky 4.57763671875e-05
do cholesky 0.8300662040710449
final computation 1.811981201171875e-05
first part 0.1863539218902588
prepare to cholesky 2.5987625122070312e-05
do cholesky 0.9051661491394043
final computation 2.193450927734375e-05
first part 0.19786882400512695
prepare to cholesky 2.574920654296875e-05
do cholesky 0.8559541702270508
final computation 1.7881393432617188e-05
first part 0.17209815979003906
prepare to cholesky 2.4080276489257812e-05
do cholesky 0.7788269519805908
final computation 2.002716064453125e-05
first part 0.17493605613708496
prepare to cholesky 2.7179718017578125e-05
do cholesky 0.8226759433746338
final computation 1.7881393432617188e-05
first part 0.1686108112335205
prepare to cholesky 2.574920654296875e-05
do cholesky 0.8022723197937012
final computation 1.9073486328125e-05
first part 0.1789391040802002
prepare to cholesky 2.5033950805664062e-05
do cholesky 0.8208808898925781
final computation 1.7881393432617188e-05
[4889283.072539778, 4889034.559907954, 4889407.334923696, 4889648.942570506, 4889598.006753816, 4885220.336408644, 4889456.655002485, 4889341.636537235, 4889584.523676748, 4886285.761601724]
"""