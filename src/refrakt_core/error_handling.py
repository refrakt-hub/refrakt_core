class XAINotSupportedError(Exception):
    """
    Exception raised when XAI methods are attempted on contrastive learning models.

    This exception is used to signal that XAI components are not supported for
    contrastive family models (SimCLR, DINO, MSN) in refrakt v1.
    """

    pass
