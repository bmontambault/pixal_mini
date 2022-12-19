from .PredicateInduction import PredicateInduction

def pixal(data, dtypes, target, attributes=None, bins=25):
    pi = PredicateInduction(data, dtypes, target, attributes, bins)
    pi.search()
    return pi.accepted