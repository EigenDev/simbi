from dataclasses import dataclass
from ...functional.helpers import order_of_mag
from typing import Any
from ...io.logging import logger, SimbiFormatter


@dataclass(frozen=True)
class ProblemIO:
    @staticmethod
    def print_params(problem_class: Any) -> None:
        if problem_class.log_parameter_setup:
            import logging
            from datetime import datetime
            from pathlib import Path

            timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
            Path(problem_class.log_output_dir).mkdir(parents=True, exist_ok=True)
            logfile = Path(problem_class.log_output_dir) / f"simbilog_{timestr}.log"
            logger.debug(f"Writing log file: {logfile}")
            file_handler = logging.FileHandler(logfile)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(SimbiFormatter())
            logger.addHandler(file_handler)

        logger.info("\nProblem Parameters:")
        logger.info("=" * 80)
        if problem_class.dynamic_args:
            for member in problem_class.dynamic_args:
                val = member.value
                if isinstance(val, float):
                    if order_of_mag(abs(val)) > 3:
                        logger.info(f"{member.name:.<30} {val:<15.2e} {member.help}")
                        continue
                    val = round(val, 3)
                val = str(val)
                logger.info(f"{member.name:.<30} {val:<15} {member.help}")
