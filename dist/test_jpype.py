import jpype
import os
import numpy

class BS(object):
    def __init__(self):
        self.path_jvm = '/Library/Java/JavaVirtualMachines/jdk1.7.0_79.jdk/Contents/Home/jre/lib/server/libjvm.dylib'
        self.path_jar = './generation.jar'
        self.max_diff = numpy.int(5)
        self.djavapath = "-Djava.class.path=%s"%os.path.abspath(
            self.path_jar
        )
        jpype.startJVM(
            self.path_jvm, '-ea', self.djavapath
        )
        self.BleuScorerClass = jpype.JClass(
            "cortex.BleuScorer"
        )
        self.BleuScorer = self.BleuScorerClass()
        print "setting threshold"
        self.BleuScorer.setThreshold(self.max_diff)


def main():
    bs = BS()

if __name__ == "__main__": main()
