
# Add by pfzhang
"""Simple layer DSL wrapper to ease creation of neural nets."""

from tvm import relay 

def bitserial_conv2d(data, weight=None, **kwargs):
    """Wrapper of bitserial_conv2d which automatically creates weights if not given
    
    Parameters
    ----------
    data: relay.Expr
        The input expression.

    weight: relay.Expr
        The weight to bitserial_conv2d
    
    kwargs: dict
        Additional arguments
    
    Returns
    ---------
    result: relay.Expr 
        The result 
    """

    name = kwargs.get('name')
    kwargs.pop('name')
    
    if not weight:
        weight = relay.var(name + '_weight')
    
    return relay.nn.bitserial_conv2d(data, weight, **kwargs)