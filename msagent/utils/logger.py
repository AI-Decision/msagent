import logging
import os
from time import sleep
from typing import Dict
import sys
from functools import wraps
import threading
from abc import abstractclassmethod, abstractmethod


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/log/'

level2fn = {
            'debug':logging.DEBUG, \
            'info':logging.INFO, \
            'warning':logging.WARNING,\
            'error':logging.ERROR, \
            'critical':logging.CRITICAL
            }


class ContextFilter(logging.Filter):

    def filter(self, record):
        if record.levelno < logging.DEBUG:
            return False
        return True


class BASELogger(object):

    logger = logging.getLogger(name='msagent')
    f_str = '%(asctime)s - %(name)s - %(filename)s - %(lineno)d - %(message)s'
    formatter = logging.Formatter(fmt=f_str)
    _instance_lock = threading.Lock()
    name2fn = {}

    def __init__(self, dir=BASE_DIR, level="debug", **kwargs):
        sleep(0.5)
        self.logger.setLevel(level2fn[level])
        self.dir_name = dir
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        if kwargs and isinstance(kwargs, Dict):
            fmt_str = kwargs.get('format', BASELogger.f_str)    
            self.formatter = logging.Formatter(fmt=fmt_str)
        fn_list = self.__register()
        for f in fn_list:
            self.name2fn[f] = getattr(self,f)
    
    def __register(self):
        return list(filter(lambda m: (not m.startswith('__') and not m.endswith("__") ) \
                           and callable(getattr(self, m)), dir(self)))

    @classmethod
    def extra_record(cls, *args, **kwargs):
        def wrapper(func):
            @wraps(wrapper)
            def inner_wrapper(object, msg, file_name=None, format=None, *args, **kwargs):
                if file_name:
                    file_path= BASE_DIR + file_name
                    fh = logging.FileHandler(file_path)
                    if isinstance(kwargs, Dict):
                        fl_name = kwargs.get('file_level', 'debug') 
                        fh.setLevel(level2fn[fl_name])
                    fmt_str = format or cls.f_str
                    fmat = logging.Formatter(fmt=fmt_str)
                    fh.setFormatter(fmt=fmat)
                    cls.logger.addHandler(fh)
                    func(object, msg, file_name, format, **kwargs)
                    cls.logger.removeHandler(fh)
                else:
                    func(object, msg, file_name, format, **kwargs)
            return inner_wrapper
 
        assert len(args) == 1 and callable(args[0]), \
                       'wrong size for decorator function'
        return wrapper(args[0])

    def select_mode(self, screen=False, file=True, **kwargs):
        assert screen is not False \
            or file is not False, 'the log mode shoule be clear'

        if screen:
            if isinstance(kwargs, Dict):
                screen_level = kwargs.get('screen_level', level2fn['debug'])
            screen_h = logging.StreamHandler(sys.stdout)
            screen_h.setLevel(screen_level)
            screen_h.setFormatter(fmt=self.formatter)
            self.logger.addHandler(screen_h)

        if file:
            if isinstance(kwargs, Dict):
                file_name = kwargs.get('file_name', 'full_record.log')
                file_level = kwargs.get('file_level', level2fn['debug'])
            file_h = logging.FileHandler(self.dir_name + file_name)
            file_h.setLevel(file_level)
            file_h.setFormatter(fmt=self.formatter)
            self.logger.addHandler(file_h)


class Logger(BASELogger):
    
    extra_record = BASELogger.extra_record

    def __init__(self, *args, **kwargs):
        super(Logger, self).__init__(*args, **kwargs)
    
    @classmethod
    def instance(cls, *args, **kwargs):
        with Logger._instance_lock:
            if not hasattr(Logger, "_instance"):
                Logger._instance = Logger(*args, **kwargs)
        return Logger._instance
    
    @extra_record
    def debug(self, msg, file_name=None, format=None, *args, **kwargs):
        BASELogger.logger.debug(msg)
    
    @extra_record
    def info(self, msg, file_name=None, format=None, **kwargs):
        BASELogger.logger.info(msg)
    
    @extra_record
    def warn(self, msg, file_name=None, format=None, **kwargs):
        self.logger.warn(msg)
    
    @extra_record
    def error(self, msg, file_name=None, format=None, **kwargs):
        self.logger.error(msg)
    
    @extra_record
    def critical(self, msg, file_name=None, format=None, **kwargs):
        self.logger.critical(msg)

logger = Logger.instance()
logger.select_mode()

