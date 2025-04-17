import os
import logging


# Tailor logging for multi-processing

class Logger(logging.Logger):
    def info(self, msg, main_process_only=True, *args, **kwargs):
        if not main_process_only or int(os.environ["RANK"]) == 0:
            return super().info(msg, *args, **kwargs)
        

def set_logging():
    rank = int(os.environ["RANK"])
    format_str = f'[%(levelname)s - %(asctime)s - rank{rank}] %(name)s:%(lineno)s: %(message)s'
    logging.basicConfig(format=format_str, level=logging.INFO)
    logging.setLoggerClass(Logger)


def get_logger(name, log_level=logging.INFO):
    return logging.getLogger(name)


set_logging()
