import numpy as np
import pandas as pd
import rpy2.rinterface_lib.callbacks
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, pandas2ri
import rpy2.rinterface_lib.callbacks

numpy2ri.activate()
pandas2ri.activate()
RBayesFactor=importr('BayesFactor', suppress_messages=True)

stdout = []
stderr = []
def add_to_stdout(line): stdout.append(line)
def add_to_stderr(line): stderr.append(line)
stdout_orig = rpy2.rinterface_lib.callbacks.consolewrite_print
stderr_orig = rpy2.rinterface_lib.callbacks.consolewrite_warnerror
rpy2.rinterface_lib.callbacks.consolewrite_print     = add_to_stdout
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = add_to_stderr

def ttestBF(x, y, side=None):
    if len(x)>1 and len(y)>1:
        if side == 'right':
            if y.mean()>x.mean():
                return -np.inf
        elif side == 'left':
            if x.mean()>y.mean():
                return -np.inf 
        res = RBayesFactor.ttestBF(x=x, y=y)
        bf = res.slots['bayesFactor']['bf'][0]
        return bf
    else:
        return -np.inf


def proportionBF(x, y, side=None):
    if side == 'right':
        if y.mean()>x.mean():
            return -np.inf
    elif side == 'left':
        if x.mean()>y.mean():
            return -np.inf
    p = min(max(.0001, y.mean()), .9999)
    res = RBayesFactor.proportionBF(x.sum(), len(x), p=p)
    bf = res.slots['bayesFactor']['bf'][0]
    return bf

def sample_predicate_continuous(data, feature, p):
    bins = int(np.ceil(1/p))
    feature_bin = np.random.choice(pd.cut(data[feature], bins=bins).unique())
    feature_left = feature_bin.left
    feature_right = feature_bin.right
    return feature_left, feature_right

def sample_predicate_numeric(data, feature, p):
    feature_left, feature_right = sample_predicate_continuous(data, feature, p)
    mask = ((data[feature] <= feature_right) & (data[feature] >= feature_left))
    return mask, [feature_left, feature_right]

def sample_predicate_date(data, feature, p):
    feature_left, feature_right = sample_predicate_continuous(data, feature, p)
    feature_left = str(feature_left).split(' ')[0]
    feature_right = str(feature_right).split(' ')[0]
    mask = ((data[feature] <= feature_right) & (data[feature] >= feature_left))
    return mask, [feature_left, feature_right]

def sample_predicate_ordinal(data, feature, p):
    feature_left, feature_right = sample_predicate_continuous(data, feature, p)
    feature_left = int(np.ceil(feature_left))
    feature_right = int(np.floor(feature_right))
    initial_mask = ((data[feature] <= feature_right) & (data[feature] >= feature_left))
    feature_left = data.loc[initial_mask, feature].min()
    feature_right = data.loc[initial_mask, feature].max()
    mask = ((data[feature] <= feature_right) & (data[feature] >= feature_left))
    return mask, [feature_left, feature_right]

def sample_predicate_nominal(data, feature, p):
    num_values = int(data[feature].nunique() * p)
    values = np.random.choice(data[feature].unique(), num_values, replace=False)
    mask = data[feature].isin(values)
    return mask, values

def sample_predicate_binary(data, feature, p):
    value = np.random.choice([0, 1])
    mask = data[feature] == value
    return mask, [value.item()]

def sample_predicate(data, dtypes, p, k=None, features=None, score_col=None, correlated_features={}):
    if features is None:
        features = []
        for i in range(k):
            feature = np.random.choice([col for col in data.columns if col != score_col and col not in features and (col not in correlated_features or correlated_features[col] not in features)])
            features.append(feature)
    else:
        k = len(features)
    predicate = {}
    mask = pd.DataFrame()
    feature_p = np.power(p, 1./k)
    
    for feature in features:
        if dtypes[feature] == 'numeric':
            sample_f = sample_predicate_numeric
        elif dtypes[feature] == 'ordinal':
            sample_f = sample_predicate_ordinal
        elif dtypes[feature] == 'date':
            sample_f = sample_predicate_date
        elif dtypes[feature] == 'nominal':
            sample_f = sample_predicate_nominal
        elif dtypes[feature] == 'binary':
            sample_f = sample_predicate_binary
            
        m, p = sample_f(data, feature, feature_p)
        mask[feature] = m
        predicate[feature] = p
    return mask, predicate

def sample_predicates(data, dtypes, num_predicates, min_k, max_k, p, score_col, correlated_features, max_groups):
    all_k = np.random.randint(min_k, max_k, num_predicates)
    predicates = []
    predicate_masks = []
    masks = []
    for k in all_k:
        valid = False
        max_tries = 100
        i = 0
        while not valid and i<max_tries:
            if len(masks) == max_groups:
                mask = masks[int(np.random.randint(0, max_groups-1))]
                m_, predicate = sample_predicate(data.loc[mask], dtypes, k, p, score_col, correlated_features)
                m = pd.DataFrame(columns=m_.columns, index=range(len(data))).fillna(False)
                for col in m.columns:
                    m.loc[mask, col] = m_[col]
            else:
                m, predicate = sample_predicate(data, dtypes, k, p, score_col, correlated_features)
            if m.all(axis=1).sum() > 0:
                valid = True
                predicates.append(predicate)
                predicate_masks.append(m)
                if len(masks) < max_groups:
                    masks.append(m.all(axis=1))
            else:
                i+=1
    return predicates, predicate_masks