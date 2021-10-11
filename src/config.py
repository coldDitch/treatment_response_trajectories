"""
Config file for all the constants needed to run different experiments
#TODO move rest of the parameters here
"""

# Faster inference but slower compilation time
# should be false when developing stan models.
# note: PARALELLIZE affect the compilation.
# If paralellized model has been compiled it will be used by default
PARALELLIZE = True

# diagnose MCMC stats, takes a bit of time but nessessary for checking for successful fit
DIAGNOSE = True

LOGS = 'logs'

# print summary of posterior
SUMMARY = True

# algorithm, mcmc
#TODO other inference methods if needed (vi, mle)
ALGORITHM = 'mcmc'

# name of the generator stanfile
GENERATORNAME = 'generator_generalized'

DAYS = 1

# name of the model stanfile, "nutrient" has to be included in the name for utils to handle it
MODELNAME = 'TRmodel_eiv_nutrient_explaining'

# seed for random generators
SEED = 1234

# plotting options
FIGSIZE = (8,4.5)
PLOTFIT = True
PLOTBASE = True
PLOTRESP = True
PLOTMEALS = True
PLOTSAMPLES = True
