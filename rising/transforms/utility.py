from .abstract import AbstractTransform


class DoNothingTransform(AbstractTransform):
    def __init__(self, grad: bool = False, **kwargs):
        """
        Forward input

        Parameters
        ----------
        grad:
            enable gradient computation inside transformation
        kwargs:
            keyword arguments passed to superclass
        """
        super().__init__(grad=grad, **kwargs)

    def forward(self, **data) -> dict:
        """
        Forward input

        Parameters
        ----------
        data: dict
            input dict

        Returns
        -------
        dict
            input dict
        """
        return data
