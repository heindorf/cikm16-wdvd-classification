#!/usr/bin/env python3

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

from argparse import ArgumentParser
import sys
import pickle

from src import learn
import src.queuelogger
import os

QUEUE_FILE_NAME = 'queue.p'


def _parse_args(argv=None):
    parser = ArgumentParser(description='WDVD-2016 Classification.')
    
    parser.add_argument('FEATURES',
                        metavar='<file>',
                        help='raw feature file')
    parser.add_argument('RESULTS',
                        metavar='<path prefix>',
                        help='path prefix for storing results')
    
    args = parser.parse_args()
    
    return args.FEATURES, args.RESULTS

# In the following, we make sure that all output to stdout and stderr is logged.
# In particular, we make sure the output of joblib.Parallel is captured.
if __name__ == "__main__":
    try:
        # Make sure there is no queue file from the last run of the program
        # The queue file is used to communicate the logging queue from the initial process to processes spawned later
        os.remove(QUEUE_FILE_NAME)
    except:
        pass
    
    queue = src.queuelogger.QueueLogger.getLoggingQueue()  # Starts a new manager process for the queue
    
    feature_file, output_prefix = _parse_args()
    learn.set_output_prefix(output_prefix)  # sets config.OUTPUT_PREFIX
    
    # Starts a new logging process (and waits until it has been loaded, i.e.,
    # pickle.load has been called and has thrown an exception, see below))
    src.queuelogger.QueueLogger.start(queue, output_prefix)
    src.queuelogger.configure_logger(queue, output_prefix)
    
    pickle.dump(queue, open(QUEUE_FILE_NAME, 'wb'))
    
    # calling the main program after all the logging has been setup
    sys.exit(learn.main(feature_file))
else:
    try:
        # sys.stderr.write("starting another process\n")

        # Will cause an exception when called for the manager process or
        # logging process because the queue has not been set yet.
        # This is on purpose to prevent calling configure_logger which
        # redirects sys.stdout and sys.stderr
        queue = pickle.load(open(QUEUE_FILE_NAME, 'rb'))
        
        src.queuelogger.QueueLogger.setLoggingQueue(queue)
        src.queuelogger.configure_logger(queue, 'FILE_NOT_SET')
        # sys.stderr.write("queue loaded successfully\n")
    except IOError:
        # sys.stderr.write('queue could not be loaded\n')
        pass
