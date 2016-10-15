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

import pandas as pd
import unittest
import datetime


class Test(unittest.TestCase):
    
    def getIndexForRevisionIdFromDf(self):
        pass

    def testGetIndexForDateFromDf(self):
        s = pd.Series(['2005-02-25T00:00:00Z',
                       '2005-02-25T00:00:00Z',
                       '2005-02-25T00:00:01Z'])
        s = pd.to_datetime(s, format='%Y-%m-%dT%H:%M:%SZ', utc=True)
        
        ts = pd.Timestamp('2005-02-25T00:00:00Z')
        
        ts2 = datetime.date(2005, 2, 25)
        ts2 = datetime.datetime.combine(ts2, datetime.datetime.min.time())
        
        self.assertEqual(s.searchsorted(ts), 0)
        self.assertEqual(s.searchsorted(ts2), 0)
        
        pass


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
