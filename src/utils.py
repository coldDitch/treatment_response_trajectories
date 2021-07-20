
def summary(fit):
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
