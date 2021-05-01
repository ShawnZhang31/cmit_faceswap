class ImageError(Exception):
    """
    图像错误
    """
    def __init__(self, message=None, status=None):
        super().__init__(message, status)
        self.message = message
        self.status = status