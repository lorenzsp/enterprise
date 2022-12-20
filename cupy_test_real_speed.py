# This script shows how to evaluate the likelihood of a real dataset. This is based on https://github.com/nanograv/12p5yr_stochastic_analysis/blob/master/notebooks/pta_gwb_analysis.ipynb
import os
print("process", os.getpid() )
dev = 7
os.system(f"CUDA_VISIBLE_DEVICES={dev}")
os.environ["CUDA_VISIBLE_DEVICES"] = f"{dev}"
os.system("echo $CUDA_VISIBLE_DEVICES")

import glob, json, pickle
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

pta.get_lnlikelihood(x0[0])
# evaluate likelihood, I also print the values to check they are not infinite
print("start timing")
tic = time.perf_counter()
print([pta.get_lnlikelihood(xx) for xx in x0])
toc = time.perf_counter()
print("the likelihood evaluation took: ",(toc-tic)/num, "seconds, number of pulsars", len(psrs))
# [4889283.07254719, 4889034.559904126, 4889407.334921482, 4889648.942572773, 4889598.006757148, 4885220.336421929, 4889456.655002594, 4889341.636542754, 4889584.523673296, 4886285.761601714]
#######################################################################################
# Here we break down the timing into different pieces
# phi_0 = rn(psrs[0]).get_phi(x0[0]) + gw(psrs[0]).get_phi(x0[0])
# B = tm(psrs[0]).get_phi(x0[0])
# Mmat = tm(psrs[0]).get_basis(x0[0])
# F = rn(psrs[0]).get_basis(x0[0])
import cupy as xp

import scipy.sparse as sps
from cupyx.scipy.linalg import block_diag, solve_triangular
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
        return xp.concatenate((xp.asarray(el) for el in TNrs))

    def __call__(self, xs, phiinv_method="cliques"):
        # map parameter vector if needed
        params = xs if isinstance(xs, dict) else self.pta.map_params(xs)

        loglike = 0

        # phiinvs will be a list or may be a big matrix if spatially
        # correlated signals
        tic = time.time()
        TNrs = self.pta.get_TNr(params)
        toc = time.time()
        print("TNr time", toc - tic)
        TNTs = self.pta.get_TNT(params)
        tic = time.time()
        toc = time.time()
        print("TNT time", toc - tic)
        tic = time.time()
        phiinvs = self.pta.get_phiinv(params, logdet=True, method=phiinv_method)
        toc = time.time()
        print("phinv time", toc - tic)
        
        # get -0.5 * (rNr + logdet_N) piece of likelihood
        # the np.sum here is needed because each pulsar returns a 2-tuple
        loglike += -0.5 * np.sum([ell for ell in self.pta.get_rNr_logdet(params)])

        # get extra prior/likelihoods
        loglike += sum(self.pta.get_logsignalprior(params))
        
        

        # red noise piece
        if self.pta._commonsignals:
            tic = time.time()
            phiinv, logdet_phi = phiinvs

            TNT = block_diag(*(xp.asarray(el) for el in TNTs))
            toc = time.time()
            print("prepare to cholesky TNT", toc-tic)
            tic = time.time()
            TNr = self._block_TNr(TNrs)
            toc = time.time()
            print("prepare to cholesky TNr", toc-tic)
            tic = time.time()
            Mat = TNT + xp.asarray(phiinv)
            toc = time.time()
            # print(TNT[10,10] ,TNr[10] )
            print("prepare to cholesky", toc-tic)
            if self.cholesky_sparse:
                try:
                    tic = time.time()
                    expval = xp.linalg.solve(Mat, TNr)
                    logdet_sigma = xp.linalg.slogdet(Mat)[1]
                    # del Mat
                    # del TNT
                    # del phiinv
                    # del phiinvs

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
            loglike += 0.5 * (float(xp.dot(TNr, expval))  - logdet_sigma - logdet_phi)
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
# [4889283.07254719, 4889034.559904126, 4889407.334921482, 4889648.942572773, 4889598.006757148, 4885220.336421929, 4889456.655002594, 4889341.636542754, 4889584.523673296, 4886285.761601714]
# [4889283.07254556, 4889034.55990401, 4889407.33493626, 4889648.94257218, 4889598.00675683, 4885220.33641609, 4889456.65500225, 4889341.6365405, 4889584.52366823, 4886285.76160171]
#######################################################################################
# my output on my local laptop
"""
process 2618116
7
WARNING: AstropyDeprecationWarning: The private astropy._erfa module has been made into its own package, pyerfa, which is a dependency of astropy and can be imported directly using "import erfa" [astropy._erfa]
./data
start timing
[4889338.246684989, 4890012.699740991, 4889458.793756989, 4877825.844757155, 4888633.338168786, 4889812.080338307, 4890095.903790723, 4882651.282209272, 4886128.722118754, 4889852.331927923]
the likelihood evaluation took:  1.7691439676098526 seconds, number of pulsars 45
TNr time 0.25646424293518066
TNT time 2.384185791015625e-07
phinv time 0.2847602367401123
prepare to cholesky 0.8602063655853271
do cholesky 0.8593297004699707
final computation 0.003132343292236328
TNr time 0.2660410404205322
TNT time 2.384185791015625e-07
phinv time 0.2724463939666748
prepare to cholesky 0.38646411895751953
do cholesky 0.21852922439575195
final computation 0.006399393081665039
TNr time 0.26895880699157715
TNT time 2.384185791015625e-07
phinv time 0.25919103622436523
prepare to cholesky 0.24916505813598633
do cholesky 0.21158647537231445
final computation 0.006550312042236328
TNr time 0.26970911026000977
TNT time 0.0
phinv time 0.278123140335083
prepare to cholesky 0.2556607723236084
do cholesky 0.2137746810913086
final computation 0.006236076354980469
TNr time 0.360095739364624
TNT time 0.0
phinv time 0.28118133544921875
prepare to cholesky 0.2458205223083496
do cholesky 0.20959687232971191
final computation 0.006417751312255859
TNr time 0.26848745346069336
TNT time 0.0
phinv time 0.2766842842102051
prepare to cholesky 0.25590062141418457
do cholesky 0.21483922004699707
final computation 0.006621360778808594
TNr time 0.2654285430908203
TNT time 2.384185791015625e-07
phinv time 0.2790966033935547
prepare to cholesky 0.24886322021484375
do cholesky 0.21185827255249023
final computation 0.006502628326416016
TNr time 0.2705235481262207
TNT time 0.0
phinv time 0.27634358406066895
prepare to cholesky 0.24841952323913574
do cholesky 0.21319890022277832
final computation 0.006229400634765625
TNr time 0.2664201259613037
TNT time 2.384185791015625e-07
phinv time 0.275454044342041
prepare to cholesky 0.24872350692749023
do cholesky 0.20990800857543945
final computation 0.00627589225769043
TNr time 1.8660674095153809
TNT time 2.384185791015625e-07
phinv time 0.3086674213409424
prepare to cholesky 0.2851841449737549
do cholesky 0.24275779724121094
final computation 0.006356716156005859
[array(4889338.24668152), array(4890012.69974084), array(4889458.79375593), array(4877825.84478885), array(4888633.33816775), array(4889812.08033219), array(4890095.90379081), array(4882651.28220918), array(4886128.72211163), array(4889852.33192779)]
"""