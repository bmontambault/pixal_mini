from .utils import pixal
from .PredicateInduction import PredicateInduction

def pixal(data, dtypes, target, attributes=None, bins=25, side=None):
    pi = PredicateInduction(data, dtypes, target, attributes, bins, side=side)
    predicates = pi.search()
    return predicates
