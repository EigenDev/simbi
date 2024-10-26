import logging
import sys
from ._detail import bcolors 
from typing import Any
import typing
TRACE_LEVEL_NUM = 5
VERBOSE_LEVEL_NUM = 15

logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
logging.addLevelName(VERBOSE_LEVEL_NUM, "VERBOSE")

class SimbiFormatter(logging.Formatter):
    non_fmt: str = "%(message)s"
    inf_fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    gen_fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        TRACE_LEVEL_NUM:                      bcolors.LIGHT_CYAN + inf_fmt + bcolors.ENDC,
        VERBOSE_LEVEL_NUM:                    bcolors.LIGHT_CYAN + inf_fmt + bcolors.ENDC,
        logging.INFO:                         non_fmt,
        logging.DEBUG:    bcolors.LIGHT_CYAN + inf_fmt + bcolors.ENDC,
        logging.WARNING:  bcolors.WARNING    + gen_fmt + bcolors.ENDC,
        logging.ERROR:    bcolors.FAIL       + gen_fmt + bcolors.ENDC,
        logging.CRITICAL: bcolors.BOLD       + gen_fmt + bcolors.ENDC,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt   = self.FORMATS.get(record.levelno, self.non_fmt)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class CustomLogger(logging.Logger):
    def trace(self, message: str, *args: Any, **kws: Any) -> None:
        if self.isEnabledFor(TRACE_LEVEL_NUM):
            self._log(TRACE_LEVEL_NUM, message, args, **kws)

    def verbose(self, message: str, *args: Any, **kws: Any) -> None:
        if self.isEnabledFor(VERBOSE_LEVEL_NUM):
            self._log(VERBOSE_LEVEL_NUM, message, args, **kws)

logging.setLoggerClass(CustomLogger)
logger: CustomLogger = typing.cast(CustomLogger, logging.getLogger("SIMBI"))
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(SimbiFormatter())
logger.addHandler(console_handler)