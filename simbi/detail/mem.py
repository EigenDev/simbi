from threading import Thread
from typing import Any, Callable

def release_memory(func: Callable[...,Any]) -> Callable[...,Any]:
    def wrapper(*args: Any, **kwargs: Any) -> None:
        p = Thread(target=func, args=args, kwargs=kwargs)
        p.start()
        p.join()
    return wrapper
