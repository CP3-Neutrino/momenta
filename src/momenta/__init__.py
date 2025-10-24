import logging


log = logging.getLogger("momenta")
log.setLevel(logging.DEBUG)
logger_ch = logging.StreamHandler()
logger_ch.setLevel(logging.DEBUG)
logger_ch.setFormatter(logging.Formatter("[%(asctime)s:%(name)s:%(levelname)s] %(message)s"))
log.addHandler(logger_ch)
