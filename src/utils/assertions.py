#------------------------------------------------------------------------------#
# is_*() functions                                                           ####
#------------------------------------------------------------------------------#


def is_dataframe(obj):
    import pandas as pd
    return isinstance(obj, pd.DataFrame)


def is_series(obj):
    import pandas as pd
    return isinstance(obj, pd.Series)


def is_ndarray(obj):
    import numpy as np
    return isinstance(obj, np.ndarray)



#------------------------------------------------------------------------------#
# all_*() functions                                                         ####
#------------------------------------------------------------------------------#


def all_str(lst):
    if isinstance(lst, list):
        return all([isinstance(element, str) for element in lst])
    else:
        return isinstance(lst, str)
    

def all_bool(lst):
    if isinstance(lst, list):
        return all([isinstance(element, bool) for element in lst])
    else:
        return isinstance(lst, bool)


def all_int(lst):
    if isinstance(lst, list):
        return all([isinstance(element, int) for element in lst])
    else:
        return isinstance(lst, int)


def all_positive(lst):
    if isinstance(lst, list):
        return all([element > 0 for element in lst])
    else:
        return lst > 0


def all_dict(lst):
    if isinstance(lst, list):
        return all([isinstance(element, dict) for element in lst])
    else:
        return isinstance(lst, dict)


def all_real(lst):
    import numpy as np
    if isinstance(lst, list):
        return np.all(np.isreal(lst))
    else:
        return np.isreal(lst)


def all_dataframe(lst):
    if isinstance(lst, list):
        return all([is_dataframe(element) for element in lst])
    else:
        return is_dataframe(lst)


def all_ndarray(lst):
    if isinstance(lst, list):
        return all([is_ndarray(element) for element in lst])
    else:
        return is_ndarray(lst)


def all_dataframe_or_series(lst):
    import pandas as pd
    if isinstance(lst, list):
        return all([isinstance(element, (pd.DataFrame, pd.Series)) for element in lst])
    else:
        return isinstance(lst, (pd.DataFrame, pd.Series))

    
def all_dataframe_or_series_or_ndarray(lst):
    import pandas as pd
    import numpy as np
    if isinstance(lst, list):
        return all([isinstance(element, (pd.DataFrame, pd.Series, np.ndarray)) for element in lst])
    else:
        return isinstance(lst, (pd.DataFrame, pd.Series, np.ndarray))

    
def is_in(elem, lst):
    assert not hasattr(elem, "__len__")
    assert hasattr(lst, "__len__")
    return elem in lst

