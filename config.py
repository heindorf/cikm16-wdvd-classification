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

__version__ = "0.0.10"

OUTPUT_PREFIX = None  # is set during initialization

USE_TEST_SET = True
STATISTICS_ENABLED = True
FEATURE_RANKING_ENABLED = True
OPTIMIZATION_ENABLED = True
CLASSIFICATION_ENABLED = True
CLASSIFICATION_GROUPS_ENABLED = True
BASELINES_ENABLED = True
ONLINE_LEARNING_ENABLED = True

LOADING_USE_MEMORY_CACHE = True
LOADING_USE_DISK_CACHE = False
PREPROCESSING_N_JOBS = -1
FEATURE_RANKING_N_JOBS = -1
CLASSIFICATION_N_JOBS = -1
OPTIMIZATION_N_JOBS = -1

EVALUATION_MAX_POINTS_ON_CURVE = 10000


def _get_globals():
    return globals()
