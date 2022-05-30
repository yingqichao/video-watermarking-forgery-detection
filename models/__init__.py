import logging
logger = logging.getLogger('base')


def create_model(opt):
    from .IRNcrop_model import IRNcropModel as M
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
