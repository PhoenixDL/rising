__DEBUG_MODE = False

# Functions to get and set the internal __DEBUG_MODE variable. This variable
# currently only defines whether to use multiprocessing or not. At the moment
# this is only used inside the DataManager, which either returns a
# MultiThreadedAugmenter or a SingleThreadedAugmenter depending on the current
# debug mode.
# All other functions using multiprocessing should be aware of this and
# implement a functionality without multiprocessing
# (even if this slows down things a lot!).


def get_debug_mode():
    """
    Getter function for the current debug mode
    Returns
    -------
    bool
        current debug mode
    """
    return __DEBUG_MODE


def switch_debug_mode():
    """
    Alternates the current debug mode
    """
    set_debug_mode(not get_debug_mode())


def set_debug_mode(mode: bool):
    """
    Sets a new debug mode
    Parameters
    ----------
    mode : bool
        the new debug mode
    """
    global __DEBUG_MODE
    __DEBUG_MODE = mode