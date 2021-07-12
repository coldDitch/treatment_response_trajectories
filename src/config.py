
# faster inference but slower compilation time, should be false when developing stan models
PARALELLIZE = True

# diagnose MCMC stats, takes a bit of time but nessessary for checking for successful fit
DIAGNOSE = True 

# algorithm, mcmc #TODO other inference methods if needed (vi, mle)
ALGORITHM = 'mcmc'

# name of the stanfile
MODELNAME = 'TRmodel'

# seed for random generators
SEED = 1234

# plotting options
FIGSIZE = (16,9)
PLOTFIT = True 
PLOTBASE = True
PLOTRESP = True