import logging
from ._detail import bcolors 
from typing import Any 

# adapted from this answer: https://stackoverflow.com/a/35804945/13874039
def addLoggingLevel(levelName: str, levelNum: int, methodName: str | None = None) -> None:
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present 

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
       raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
       raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
       raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self: "Any", message: str, *args: "Any", **kwargs: "Any") -> None:
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
            
    def logToRoot(message: str, *args: "Any", **kwargs: "Any") -> None:
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)
    

addLoggingLevel('PRINT', logging.DEBUG - 5)
class SimbiFormatter(logging.Formatter):
    non_fmt: str = "%(message)s"
    inf_fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    gen_fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG:    bcolors.UNDERLINE  + gen_fmt + bcolors.ENDC,
        logging.INFO:     bcolors.LIGHT_CYAN + inf_fmt + bcolors.ENDC,
        logging.WARNING:  bcolors.WARNING    + gen_fmt + bcolors.ENDC,
        logging.ERROR:    bcolors.FAIL       + gen_fmt + bcolors.ENDC,
        logging.CRITICAL: bcolors.BOLD       + gen_fmt + bcolors.ENDC,
        logging.PRINT:                         non_fmt, #type: ignore
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt   = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    


logger = logging.getLogger("SIMBI")
logger.setLevel(logging.PRINT) # type: ignore

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.PRINT) # type: ignore
console_handler.setFormatter(SimbiFormatter())
logger.addHandler(console_handler)