import pandas as pd
from .utils import ttestBF
from .utils import proportionBF

class Predicate:
    
    def __init__(self, data, dtypes, attribute_values, attribute_mask=None, target=None, parent=None, side=None):
        self.data = data
        self.dtypes = dtypes
        self.attribute_values = attribute_values
        if attribute_mask is None:
            self.attribute_mask = self.get_attribute_mask()
        else:
            self.attribute_mask = attribute_mask
        self.mask = self.attribute_mask.all(axis=1)
        self.target = target
        self.bf_score = {}
        self.attributes = list(self.attribute_mask.columns)
        self.parent = parent
        self.side = side
        
    def get_attribute_mask(self):
        return pd.DataFrame({k: self.get_attribute_value_mask(k, v) for k,v in self.attribute_values.items()})
    
    def get_attribute_value_mask(self, attribute, value):
        if self.dtypes[attribute] in ('nominal', 'list'):
            return self.data[attribute].isin(value)
        else:
            return (self.data[attribute] >= value[0]) & (self.data[attribute] <= value[1])
        
    def bf(self, target=None, attribute=None, data=None, mask=None, apply_attribute=True):
        if data is None:
            data = self.data
        if mask is None:
            mask = self.mask
        if target is None:
            target = self.target
            
        if (target, attribute) not in self.bf_score:
            if attribute is not None and apply_attribute:
                other_mask = self.attribute_mask[[attr for attr in self.attributes if attr != attribute]].all(axis=1)
                self.bf_score[(target, attribute)] = self.bf(target, attribute, data.loc[other_mask], mask.loc[other_mask], False)
            else:
                if target == 'count':
                    self.bf_score[(target, attribute)]  = proportionBF(mask.astype(int), (~mask).astype(int), self.side)
                else:
                    x = data.loc[mask, target]
                    y = data.loc[~mask, target]
                    if self.dtypes[target] == 'binary':
                        self.bf_score[(target, attribute)] = proportionBF(x, y, self.side)
                    else:
                        self.bf_score[(target, attribute)] = ttestBF(x, y, self.side)
        return self.bf_score[(target, attribute)]
    
    def is_subsumed(self, predicate):
        attributes_overlap = set(self.attributes).issubset(predicate.attributes)
        if attributes_overlap:
            attributes_subsumed = True
            for attribute in self.attributes:
                attr_subsumed = self.is_subsumed_attribute(predicate, attribute)
                if not attr_subsumed:
                    attributes_subsumed = False
            if attributes_subsumed:
                return predicate.bf() >= self.bf()
            else:
                return False
        else:
            return False
        
    def is_subsumed_attribute(self, predicate, attribute):
        if self.dtypes[attribute] == 'nominal':
            a = self.attribute_values[attribute]
            b = predicate.attribute_values[attribute]
            return set(a).issubset(b)
        else:
            left_a, right_a = self.attribute_values[attribute]
            left_b, right_b = predicate.attribute_values[attribute]
            return (left_b<=left_a) & (right_b>=right_a)
    
    def __repr__(self):
        return ' '.join([f'{k}:{v}' for k,v in self.attribute_values.items()])
