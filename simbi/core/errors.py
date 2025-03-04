class SimbiError(Exception):
    """Base exception for simbi errors"""
    pass

class ValidationError(SimbiError):
    """Raised when validation fails"""
    pass