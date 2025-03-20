import argparse
from typing import Any


class ParseKVAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        setattr(namespace, self.dest, dict())
        for each in values:
            try:
                key, value = each.split("=")
                getattr(namespace, self.dest)[key] = value
            except ValueError as ex:
                message = "\nTraceback: {}".format(ex)
                message += "\nError on '{}' || It should be 'key=value'".format(each)
                raise argparse.ArgumentError(self, str(message))


def get_subparser(parser: argparse.ArgumentParser, idx: int) -> Any:
    subparser = [
        subparser
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
        for _, subparser in action.choices.items()
    ]
    return subparser[idx]


def max_thread_count(param: Any) -> int:
    import multiprocessing

    num_threads_available = multiprocessing.cpu_count()

    try:
        val = int(param)
    except ValueError:
        raise argparse.ArgumentTypeError("\nMust be a integer\n")

    if val > num_threads_available:
        raise argparse.ArgumentTypeError(
            f"\nTrying to set thread count greater than available compute core(s) equal to {num_threads_available}\n"
        )

    return val


__all__ = ["ParseKVAction", "get_subparser", "max_thread_count"]
