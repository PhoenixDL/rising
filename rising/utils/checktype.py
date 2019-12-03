
def check_scalar(x):
    """
    Provide interface to check for scalars

    Parameters
    ----------
    x: typing.Any
        object to check for scalar

    Returns
    -------
    bool
        True if input is scalar
    """
    if isinstance(x, (int, float)):
        return True
    else:
        return False
