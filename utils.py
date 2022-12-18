def ttestBF(x, y):
    res = RBayesFactor.ttestBF(x=x, y=y)
    bf = res.slots['bayesFactor']['bf'][0]
    return bf

def proportionBF(x, y):
    res = RBayesFactor.proportionBF(x.sum(), x.sum()+y.sum(), p=y.mean())
    bf = res.slots['bayesFactor']['bf'][0]
    return bf
