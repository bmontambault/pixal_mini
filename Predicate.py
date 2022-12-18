import pandas as pd
from .utils import ttestBF
from .utils import proportionBF

class Predicate:
    
    def __init__(self, data, dtypes, attribute_values, attribute_mask=None, target=None, parent=None):
        self.data = data
        self.dtypes = dtypes
        self.attribute_values = attribute_values
        if attribute_mask is None:
            self.attribute_mask = self.get_attribute_mask()
        else:
            self.attribute_mask = attribute_mask
        self.mask = self.attribute_mask.all(axis=1)
        self.target = target
        self.bf_score = None
        self.attributes = list(self.attribute_mask.columns)
        self.parent = parent
        
    def get_attribute_mask(self):
        return pd.DataFrame({k: self.get_attribute_value_mask(k, v) for k,v in self.attribute_values.items()})
    
    def get_attribute_value_mask(self, attribute, value):
        if self.dtypes[attribute] in ('nominal', 'list'):
            return self.data[attribute].isin(value)
        else:
            return (self.data[attribute] >= value[0]) & (self.data[attribute] <= value[1])
        
    def bf(self, target=None):
        if self.bf_score is None:
            if target is None:
                target = self.target
            dtype = self.dtypes[target]
            x = self.data.loc[self.mask, target]
            y = self.data.loc[~self.mask, target]
            if dtype == 'numeric':
                self.bf_score = ttestBF(x, y)
            elif dtype == 'binary':
                self.bf_score = proportionBF(x, y)
        return self.bf_score
    
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
