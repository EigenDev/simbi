from threading import Thread
from typing import Any, Callable
import queue

def release_memory(func: Callable[...,Any]) -> Callable[...,Any]:
    def wrapper(*args: Any, **kwargs: Any) -> None:
        q: queue.Queue[None | Exception] = queue.Queue()

        def thread_func(*args: Any, **kwargs: Any) -> None:
            try:
                func(*args, **kwargs)
                q.put(None)  # No error
            except Exception as e:
                q.put(e)  # Put the exception in the queue

        p = Thread(target=thread_func, args=args, kwargs=kwargs)
        p.start()
        p.join()

        # Check if there was an exception
        exception = q.get()
        if exception is not None:
            raise exception

    return wrapper
