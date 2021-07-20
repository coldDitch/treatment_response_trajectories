# Faster inference but slower compilation time
# should be false when developing stan models.
# note: PARALELLIZE affect the compilation.
# If paralellized model has been compiled it will be used by default
PARALELLIZE = True

# diagnose MCMC stats, takes a bit of time but nessessary for checking for successful fit
DIAGNOSE = True

# print summary of posterior
SUMMARY = True

# algorithm, mcmc #TODO other inference methods if needed (vi, mle)
ALGORITHM = 'mcmc'

# name of the generator stanfile
GENERATORNAME = 'generator'

# name of the model stanfile
MODELNAME = 'TRmodel_eiv_exposure'


# generator parameters
DAYS = 2
NOISESCALE = 1

# nutrient parameters
NUM_NUTRIENTS = 5

# eiv parameters
MEAL_REPORTING_NOISE = 1
MEAL_REPORTING_BIAS = 0

# seed for random generators
SEED = 1234

# plotting options
FIGSIZE = (5,5)
PLOTFIT = True 
PLOTBASE = True
PLOTRESP = True
