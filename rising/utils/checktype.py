
def check_scalar(x):
    """
    Provide interface to check for scalars

    Args:
        x: object to check for scalar

    Returns:
        True if input is scalar
    """
    return isinstance(x, (int, float))
