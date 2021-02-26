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


def all_dataframe(lst):
    import pandas as pd
    if isinstance(lst, list):
        return all([isinstance(element, pd.DataFrame) for element in lst])
    else:
        return isinstance(lst, pd.DataFrame)


def all_ndarray(lst):
    import numpy as np
    if isinstance(lst, np.ndarray):
        return all([isinstance(element, np.ndarray) for element in lst])
    else:
        return isinstance(lst, np.ndarray)


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