import logging
from ._detail import bcolors 

class SimbiFormatter(logging.Formatter):
    format_txt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG:    bcolors.UNDERLINE  + format_txt + bcolors.ENDC,
        logging.INFO:     bcolors.LIGHT_CYAN + format_txt + bcolors.ENDC,
        logging.WARNING:  bcolors.WARNING    + format_txt + bcolors.ENDC,
        logging.ERROR:    bcolors.FAIL       + format_txt + bcolors.ENDC,
        logging.CRITICAL: bcolors.BOLD       + format_txt + bcolors.ENDC,
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