def get_name(obj, namespace=globals()):
    name = [name for name in namespace if namespace[name] is obj][0]
    return name

def check_object(obj, name, _type:bool=True, _shape:bool=True, _head:bool=True, _head_size:int=10):
    
    import numpy as np
    
    out = ""
    
    out += "name: " + name
    
    if _type:
        out += "\ntype: " + str(type(obj))
    
    if _shape:
        if isinstance(obj, np.ndarray):
            out += "\nshape: " + str(obj.shape)
    
    if _head:
        out += "\nhead: " + str(obj[:_head_size])
        
    print(out + "\n")
    
def str_right(string:str, num_chars:int):
    """ Sub-Select the right-most number of characters from a string """
    assert isinstance(string, str)
    assert isinstance(num_chars, int)
    return string[-num_chars:]