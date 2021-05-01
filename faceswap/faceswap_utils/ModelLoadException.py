class ModelLoadError(Exception):
    """
    模型加载类异常
    """
    def __init__(self, message=None, status=None):
        super().__init__(message, status)
        self.message = message
        self.status = status