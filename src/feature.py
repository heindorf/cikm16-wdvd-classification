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


class Feature:
    def __init__(self, inputNames, transformers=None, outputName=None):
        if transformers == None:
            transformers = []
        
        if type(inputNames) is not list:
            inputNames = [inputNames]
        if type(transformers) is not list:
            raise Exception("transformers should be a list, but it was " +
                            str(type(transformers)))
        
        self.__inputNames = inputNames
        self.__transformers = transformers
        
        self.__outputName = outputName
        if self.__outputName == None:
            if len(inputNames) > 1:
                raise Exception("There should only be one input name, " +
                                "if no output name is specified.")
            self.__outputName = str(inputNames[0])
            
        self.__group = None
        self.__subgroup = None
        
    def getInputNames(self):
        return self.__inputNames
    
    def getOutputName(self):
        return self.__outputName
        
    def getTransformers(self):
        return self.__transformers
    
    def getGroup(self):
        return self.__group
    
    def getSubgroup(self):
        return self.__subgroup
    
    def setGroup(self, group):
        self.__group = group
        
    def setSubgroup(self, subgroup):
        self.__subgroup = subgroup
    
    def __str__(self):
        return ("Feature %s (%s, %s)" %
                (self.__outputName,
                 str(self.__inputNames),
                 str(self.__transformers)))
