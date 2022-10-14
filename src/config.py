"""
Config file for all the constants needed to run different experiments
#TODO move rest of the parameters here
"""

# Faster inference but slower compilation time
# should be false when developing stan models.
# note: PARALELLIZE affect the compilation.
# If paralellized model has been compiled it will be used by default

PARALELLIZE = False

TRAIN_PERCENTAGE = 0.33
PATIENT_ID = list(range(14))

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

DAYS = 3

PRIOROVERRESPONSEH = False

NUTRIENTS = ['STARCH', 'SUGAR', 'FIBC', 'FAT', 'PROT']

# name of the model stanfile, "nutrient" has to be included in the name for utils to handle it
MODELNAME = 'TRmodel_hier_nutr'

# seed for random generators
SEED = 12345

#
SYNTHETIC = False

# plotting options
FIGSIZE = (8,4.5)
PLOTFIT = True
PLOTBASE = True
PLOTRESP = True
PLOTMEALS = True
PLOTSAMPLES = False
