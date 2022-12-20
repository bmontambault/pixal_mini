import pandas as pd
from IPython.display import clear_output
from .Predicate import Predicate

class PredicateInduction:
    
    def __init__(self, data, dtypes, target, attributes=None, bins=25, side=None):
        self.data = data
        self.dtypes = dtypes
        self.target = target
        self.bins = bins
        self.side = side
        
        self.data_ = pd.DataFrame()
        self.dtypes_ = {}
        self.col_map_left = {}
        self.col_map_right = {}
        for col in self.data.columns:
            if self.dtypes[col] == 'numeric' and col != self.target:
                d = pd.cut(self.data[col], bins=bins)
                d_map_left = pd.Series(d.unique()).apply(lambda x: x.left).sort_values()
                d_map_left.index = range(len(d_map_left))
                d_map_right = pd.Series(d.unique()).apply(lambda x: x.right).sort_values()
                d_map_right.index = range(len(d_map_right))
                new_col = col + '_binned'
                self.data_[new_col] = d.apply(lambda x: x.left).map(pd.Series(d_map_left.index, index=d_map_left))
            
                self.dtypes_[new_col] = 'ordinal'
                self.col_map_left[new_col] = d_map_left
                self.col_map_right[new_col] = d_map_right
            else:
                self.data_[col] = self.data[col]
                self.dtypes_[col] = self.dtypes[col]
                
        if attributes is None:
            self.attributes = [col for col in self.data_.columns if col != self.target]
        else:
            self.attributes = [col if self.dtypes[col] != 'numeric' else col+'_binned' for col in attributes]
        
        self.base_predicates = {attribute: self.get_base_predicates_attribute(attribute) for attribute in self.attributes}
        self.frontier = []
        self.accepted = []
        
    def get_base_predicates_attribute(self, attribute):
        dtype = self.dtypes_[attribute]
        values = self.data_[attribute].unique()
        if dtype == 'ordinal':
            return {value: Predicate(self.data_, self.dtypes_, {attribute: [value, value]}, target=self.target, side=self.side) for value in values}
        else:
            return {value: Predicate(self.data_, self.dtypes_, {attribute: [value]}, target=self.target, side=self.side) for value in values}
    
    def get_adjacent_attribute(self, predicate, attribute):
        if self.dtypes_[attribute] == 'nominal':
            return []
        else:
            left, right = predicate.attribute_values[attribute]
            left = left - 1
            right = right + 1
            left_adjacent = self.base_predicates[attribute].get(left)
            right_adjacent = self.base_predicates[attribute].get(right)
            return [adj for adj in [left_adjacent, right_adjacent] if adj is not None]
        
    def conjoin_predicates(self, predicates):
        attributes = list(set([a for b in [p.attributes for p in predicates] for a in b]))
        parent = max(predicates, key=lambda x: x.bf(self.target))
        attribute_values = {}
        for attribute in attributes:
            values_ = [p.attribute_values[attribute] for p in predicates if attribute in p.attributes]
            if self.dtypes_[attribute] == 'nominal':
                values = list(set([a for b in values_ for a in b]))
            else:
                all_left, all_right = zip(*values_)
                left = min(all_left)
                right = max(all_right)
                values = [left, right]
            attribute_values[attribute] = values
            
        new_predicate = Predicate(self.data_, self.dtypes_, attribute_values, target=self.target, parent=parent, side=self.side)
        new_bf = new_predicate.bf(self.target)
        return new_predicate, new_bf
    
    def insert_predicate_sorted(self, q, predicate, bf, return_index=False):
        if len(q) == 0:
            if return_index:
                return [predicate], None
            else:
                return [predicate]
        else:
            i = 0
            while i < len(q) and bf < q[i].bf():
                is_subsumed = predicate.is_subsumed(q[i])
                if is_subsumed:
                    if return_index:
                        return q, None
                    else:
                        return q
                else:
                    i += 1
            q.insert(i, predicate)

            subsumed_index = []
            for j in range(i+1, len(q)):
                is_subsumed = q[j].is_subsumed(predicate)
                if is_subsumed:
                    subsumed_index.append(j)
                    
            res = [q[k] for k in range(len(q)) if k not in subsumed_index]
            if return_index:
                return res, i
            else:
                return res
    
    def expand_predicate_attribute(self, predicate, attribute, verbose=False):
        q = [predicate]
        r = []
        while len(q)>0:
            p = q.pop(0)
            A = self.get_adjacent_attribute(p, attribute)
            any_bf_gt = False
            
            if verbose:
                print(p, p.bf())
                print(A)
            for a in A:
                p_, bf_ = self.conjoin_predicates([p, a])
                if verbose:
                    print(p_, bf_)
                if bf_ >= max(p.bf(),a.bf()):
                    q = self.insert_predicate_sorted(q, p_, bf_)
                    any_bf_gt = True
            if verbose:
                print()
            if not any_bf_gt:
                while p.parent is not None and p.parent.bf()>=p.bf():
                    p = p.parent
                r = self.insert_predicate_sorted(r, p, p.bf())
        return r
    
    def expand_predicate(self, predicate, verbose=False):
        children = [a for b in [self.expand_predicate_attribute(predicate, attribute, verbose) for attribute in predicate.attributes] for a in b]
        return children
            
    def expand(self, verbose=False, return_is_expanded=False):
        frontier = []
        is_expanded = []
        for p in self.frontier:
            children = self.expand_predicate(p, verbose)
            if len(children)>0:
                for child in children:
                    frontier, index = self.insert_predicate_sorted(frontier, child, child.bf(), True)
                    if index is not None:
                        is_expanded.insert(index, True)
            else:
                frontier, index = self.insert_predicate_sorted(frontier, p, p.bf(), True)
                if index is not None:
                    is_expanded.insert(index, False)
        self.frontier = frontier
        if return_is_expanded:
            return is_expanded
    
    def refine_predicate_attribute(self, predicate, attribute):
        children = []
        for _,p in self.base_predicates[attribute].items():
            p_, bf_ = self.conjoin_predicates([predicate, p])
            if bf_ > max(p.bf(),predicate.bf()):
                children = self.insert_predicate_sorted(children, p_, bf_)
        return children
    
    def refine_predicate(self, predicate):
        children = []
        for attribute in self.attributes:
            if attribute not in predicate.attributes:
                attribute_children = self.refine_predicate_attribute(predicate, attribute)
                for child in attribute_children:
                    children = self.insert_predicate_sorted(children, child, child.bf())
        if len(children) == 0:
            self.accepted = self.insert_predicate_sorted(self.accepted, predicate, predicate.bf())
        return children
            
    def refine(self):
        if len(self.frontier) == 0:
            for attr,p in self.base_predicates.items():
                for v in p.values():
                    self.frontier = self.insert_predicate_sorted(self.frontier, v, v.bf())
        else:
            frontier = []
            for p in self.frontier:
                children = self.refine_predicate(p)
                for child in children:
                    frontier = self.insert_predicate_sorted(frontier, child, child.bf())
            self.frontier = frontier
            
    def expand_nominal(self):
        accepted = []
        frontier = []
        for p in self.accepted:
            predicate_children = []
            if len(p.attributes)>1:
                for attr in p.attributes:
                    if self.dtypes_[attr] == 'nominal':
                        adj = [v for k,v in self.base_predicates[attr].items() if k not in p.attribute_values[attr]]
                        for a in adj:
                            p_, bf_ = self.conjoin_predicates([p, a])
                            if bf_ > max(p.bf(), a.bf()):
                                predicate_children.append(p_)
                if len(predicate_children) > 0:
                    for child in predicate_children:
                        frontier = self.insert_predicate_sorted(frontier, child, child.bf())
                else:
                    accepted.append(p)
            else:
                accepted.append(p)
        self.accepted = accepted
        self.frontier = frontier
    
    def search_(self, max_frontier_length=100):
        init = True
        while init or len(self.frontier)>0:
            if len(self.frontier) > max_frontier_length:
                self.frontier = self.frontier[:max_frontier_length]
            self.display_frontier_length()
            self.refine()
            self.display_frontier_length()
            self.expand()
            self.display_frontier_length()
            init = False
        self.expand_nominal()
        is_expanded = self.expand(return_is_expanded=True)
        
        self.display_frontier_length()
        accepted_index = []
        for i in range(len(self.frontier)):
            if is_expanded[i]:
                self.accepted = self.insert_predicate_sorted(self.accepted, self.frontier[i], self.frontier[i].bf())
                accepted_index.append(i)
        self.frontier = [self.frontier[j] for j in range(len(self.frontier)) if j not in accepted_index]
        self.display_frontier_length()
        
    def search(self, max_frontier_length=100):
        init = True
        while init or len(self.frontier)>0:
            self.search_(max_frontier_length)
            init = False
        self.accepted = [a for a in self.accepted if a.bf()>0]
        predicates = [self.map_predicate(a) for a in self.accepted]
        return predicates
    
    def map_predicate(self, predicate):
        attribute_values = {}
        attribute_mask = pd.DataFrame()
        for k,v in predicate.attribute_values.items():
            if '_binned' in k:
                left = self.col_map_left[k][v[0]]
                right = self.col_map_right[k][v[1]]
                attribute_values['_binned'.join(k.split('_binned')[:-1])] = [left, right]
                attribute_mask['_binned'.join(k.split('_binned')[:-1])] = predicate.attribute_mask[k]
            else:
                attribute_values[k] = v
                attribute_mask[k] = predicate.attribute_mask[k]
        predicate = Predicate(self.data, self.dtypes, attribute_values, attribute_mask)
        return predicate
            
    def display_frontier_length(self):
        clear_output()
        print('frontier length:', len(self.frontier))
