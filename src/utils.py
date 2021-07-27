"""
    extra utility functions
"""

def fit_varnames(fit):
    """Find one dimensional variables (non array or log probability variales)

    Args:
        fit CmdStanMCMC: the model object fit

    Returns:
        [type]: [description]
    """
    summ = fit.summary()
    means = summ['Mean']
    return [key for key, val in means.iteritems() if not '[' in key and not '__' in key]


def summary(fit):
    """print summary and mean, Rhats for one dimensional variables

    Args:
        fit CmdStanMCMC: the model object fit
    """
    summ = fit.summary()
    print(summ)
    means = summ['Mean']
    print('means')
    for key, val in means.iteritems():
        if not '[' in key:
            print(key, val)
    print('R_hats')
    means = summ['R_hat']
    for key, val in means.iteritems():
        if not '[' in key:
            print(key, val)
