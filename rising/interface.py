class AbstractMixin(object):
    """
    This class implements an interface which handles non processed arguments.
    Subclass all classes which mixin additional methods and attributes
    to existing classes with multiple inheritance from this class as backup
    for handling additional arguments.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            *args:
                positional arguments forwarded to next object if it is the
                    last class before object in mro
            **kwargs: keyword arguments saved to object if it is the last
                class before object in mro.
                Otherwise forwarded to next class.
        """
        mro = type(self).mro()
        mro_idx = mro.index(AbstractMixin)
        # +2 because index starts at 0 and only one more class should be called
        if mro_idx + 2 == len(mro):
            # only object init is missing
            super().__init__()
            for key, item in kwargs.items():
                setattr(self, key, item)
        else:
            # class is not last before object -> forward arguments
            super().__init__(*args, **kwargs)
