# -----------------------------------------------------------------------------
# MIT License
#
# Copyright (c) 2016 Stefan Heindorf, Martin Potthast, Benno Stein, Gregor Engels
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

import logging.handlers
import multiprocessing
import sys

import config


# Stream similar to stdout/stderr that writes to a logger instead of stdout/stderr
class _LoggingStream(object):
    def __init__(self, logger, stream_name, log_level=logging.INFO):
        self.logger = logger
        self.stream_name = stream_name
        self.log_level = log_level
        self.linebuf = ''
 
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            # sys.stderr.write(line.rstrip())
            self.logger.log(
                self.log_level,
                '[' + self.stream_name + "] " + line.rstrip())

    # e.g., the multiprocessing library calls sys.stderr.flush()
    def flush(self):
        pass


# Configure the logger
def _configure_listener(output_prefix):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
     
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
      
    # Sometimes causes exception on Windows.
    # see also: https://support.microsoft.com/en-us/kb/899149
    # see also: http://stackoverflow.com/questions/10411359/unpickling-large-objects-stored-on-network-drives
    fileHandler = logging.FileHandler(output_prefix + ".log")
    fileHandler.setLevel(logging.DEBUG)
    logger.addHandler(fileHandler)
     
    _setFormatters(_ContextChange("INIT", "INIT"))

    
def _setFormatters(context_change):
    logger = logging.getLogger()
    formatter = logging.Formatter(
        '[%%(asctime)s] [%%(levelname)8s] [%s] [%s] [%%(module)s] [%%(processName)s] %%(message)s' %
        (context_change.getNewTimeLabel(), context_change.getNewSystemName()),
        datefmt='%Y-%m-%d %H:%M:%S')
    
    for handler in logger.handlers:
        handler.setFormatter(formatter)


# This is the listener process top-level loop: wait for logging events
# (LogRecords)on the queue and handle them, quit when you get a None for a
# LogRecord.
def _listener_process(output_prefix, queue, lock):
    # the listener process has been successfully loaded now (in particular,
    # the main file has been executed now)
    lock.release()
    
    _configure_listener(output_prefix)
    while True:
        try:
            record = queue.get()
            
            # We send None as a sentinel to tell the listener to quit.
            if record is None:
                break
            elif isinstance(record, _ContextChange):
                _setFormatters(record)
            else:
                logger = logging.getLogger(record.name)
                logger.handle(record)
        # Has the program terminated?
        except (KeyboardInterrupt, SystemExit, EOFError):
            raise  # leave loop and close logger
        except:
            import traceback
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


# called by all processes upon initialization
def configure_logger(queue, outputPrefix):
    config.OUTPUT_PREFIX = outputPrefix
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    logger = logging.getLogger()
    
    # On linux, the logging handlers are inherited from the parent process.
    # Hence it is necessary to remove them first.
    for handler in logger.handlers:
        logger.removeHandler(handler)
    
    logger.addHandler(h)
    logger.setLevel(logging.DEBUG)
    
    # Redirect everything sent to stdout and stderr to the logger
    # (scikit-learn and pandas sometime sents warning to stdout or stderr)
    sys.stdout = _LoggingStream(logger, 'STDOUT', logging.INFO)
    sys.stderr = _LoggingStream(logger, 'STDERR', logging.WARN)


class QueueLogger:
    # static variables
    queue = None
    listener = None
    
    @staticmethod
    def getLoggingQueue():
        if (QueueLogger.queue is None):
            # QueueLogger.queue = multiprocessing.Queue(-1) # cannot be pickled
            QueueLogger.queue = multiprocessing.Manager().Queue(-1)  # can be pickled
             
        return QueueLogger.queue
    
    @staticmethod
    def setLoggingQueue(queue):
        QueueLogger.queue = queue
       
    # Starts a separate logger and waits until the process has been loaded
    @staticmethod
    def start(queue, outputPrefix):
        lock = multiprocessing.Lock()
        lock.acquire()
        
        QueueLogger.listener = multiprocessing.Process(
            target=_listener_process, args=(outputPrefix, queue, lock,))
        QueueLogger.listener.start()
        
        # lock is released by listener process after it has been loaded
        lock.acquire()
        
    # Stopping the process of the logger
    @staticmethod
    def stop():
        QueueLogger.queue.put_nowait(None)
        QueueLogger.listener.join()
    
    # changes the system name of the logger
    @staticmethod
    def setContext(time_label, system_name):
        contextChange = _ContextChange(time_label, system_name)
        QueueLogger.queue.put(contextChange)

      
class _ContextChange():
    def __init__(self, new_time_label, new_system_name):
        self.__new_time_label = new_time_label
        self.__new_system_name = new_system_name
    
    def getNewTimeLabel(self):
        return self.__new_time_label
    
    def getNewSystemName(self):
        return self.__new_system_name
