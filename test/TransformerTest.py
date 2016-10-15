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

import unittest
import pandas as pd
import numpy as np

from pandas.util.testing import assert_frame_equal

from src import transformers


class Test(unittest.TestCase):
    def testMostFrequentElementTransformer(self):
        df = (pd.DataFrame(pd.Series(
            ['abc,def,ghi', 'ghi,abc', 'ghi', 'x1', 'x1,x2,x3', 'x2', 'x2', np.nan])
            .astype('category')))
        
        transformed_df = transformers.MostFrequentTagTransformer().fit_transform(df)
        
        expected_df = (pd.DataFrame(pd.Series(
            ['ghi', 'ghi', 'ghi', 'x1', 'x2', 'x2', 'x2', np.nan])
            .astype('category')))

        assert_frame_equal(expected_df, transformed_df)
        
    def testMostFrequentElementTransformerFit(self):
        df1 = pd.DataFrame(['abc', 'def,abc'])
        
        transformer = transformers.MostFrequentTagTransformer()
        transformer.fit(df1)
        
        df2 = (pd.DataFrame(pd.Series(
            ['new,abc', 'new,def,new', 'x1,x2', 'x3', 'def', 'def', np.nan])
            .astype('category')))
        transformed_df = transformer.transform(df2)
        
        expected_df = (pd.DataFrame(pd.Series(
            ['abc', 'def', 'x1', 'x3', 'def', 'def', np.nan])
            .astype('category')))

        assert_frame_equal(expected_df, transformed_df)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
