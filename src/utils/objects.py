def get_name(obj, namespace=globals()):
    name = [name for name in namespace if namespace[name] is obj][0]
    return name

def check_object(obj, name, _type:bool=True, _shape:bool=True, _head:bool=True):
    """
    Check the details and attributes of an object

    Args:
        obj (any): The object to be checked
    """
    
    import numpy as np
    
    out = ""
    
    out += "name: " + name
    
    if _type:
        out += "\ntype: " + str(type(obj))
    
    if _shape:
        if isinstance(obj, np.ndarray):
            out += "\nshape: " + str(obj.shape)
    
    if _head:
        out += "\nhead: " + str(obj[:10])
        
    print(out + "\n")