# This script shows how to evaluate the likelihood of a real dataset. This is based on https://github.com/nanograv/12p5yr_stochastic_analysis/blob/master/notebooks/pta_gwb_analysis.ipynb
import os
print("process", os.getpid() )
dev = 2
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
from enterprise.signals.signal_base import LogLikelihood, VectorizedLogLikelihood
loglike = LogLikelihood(pta,use_gpu=False,)
loglike_torch = VectorizedLogLikelihood(pta,use_gpu=True)
loglike_cupy = LogLikelihood(pta,use_gpu=True)

loglike(x0[0] )
tic = time.perf_counter()
print([loglike(x0[ii]) for ii in range(num)])
toc = time.perf_counter()
print("likelihood speed cpu", (toc - tic) /num )
# likelihood speed 0.8502637445926666

loglike_torch(x0[:2])
tic = time.perf_counter()
print(loglike_torch(x0[:num]))
toc = time.perf_counter()
print("likelihood speed torch vectorized", (toc - tic) /num )
#likelihood speed 0.6127635911107063


loglike_cupy(x0[0])
tic = time.perf_counter()
print([loglike_cupy(x0[ii]) for ii in range(num)])
toc = time.perf_counter()
print("likelihood speed", (toc - tic) /num )
# check each part
"""
output
start timing
[4889283.072545683, 4889034.5599048445, 4889407.334939813, 4889648.942571797, 4889598.006756077, 4885220.336414361, 4889456.655002068, 4889341.636543239, 4889584.523680755, 4886285.761601718]
the likelihood evaluation took:  2.519346130080521 seconds, number of pulsars 45
[4889283.072545683, 4889034.5599048445, 4889407.334939813, 4889648.942571797, 4889598.006756077, 4885220.336414361, 4889456.655002068, 4889341.636543239, 4889584.523680755, 4886285.761601718]
likelihood speed cpu 2.7487280969507992
[4889283.07254528 4889034.55990506 4889407.33494747 4889648.94257222
 4889598.00675541 4885220.33641351 4889456.65500313 4889341.63654126
 4889584.52367457 4886285.76160172]
likelihood speed torch vectorized 0.8733246718533337
[4889283.072544385, 4889034.559903989, 4889407.334933543, 4889648.942571439, 4889598.006757614, 4885220.336419006, 4889456.655003585, 4889341.636541557, 4889584.523670929, 4886285.761601711]
likelihood speed 0.3262205273844302"""