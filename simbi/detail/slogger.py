import logging
import sys
from ._detail import bcolors 
class SimbiFormatter(logging.Formatter):
    non_fmt: str = "%(message)s"
    inf_fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    gen_fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG:                         non_fmt,
        logging.INFO:     bcolors.LIGHT_CYAN + inf_fmt + bcolors.ENDC,
        logging.WARNING:  bcolors.WARNING    + gen_fmt + bcolors.ENDC,
        logging.ERROR:    bcolors.FAIL       + gen_fmt + bcolors.ENDC,
        logging.CRITICAL: bcolors.BOLD       + gen_fmt + bcolors.ENDC,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt   = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    


logger = logging.getLogger("SIMBI")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(SimbiFormatter())
logger.addHandler(console_handler)