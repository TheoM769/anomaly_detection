# __init__.py
import os
import pkgutil

__all__ = []

for _, modname, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
    __all__.append(modname)
    __import__(f"{__name__}.{modname}")
