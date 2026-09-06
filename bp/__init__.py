"""Bar path analysis: plate finding, tracking, rep detection.

The modules in here import each other by bare name (`import v7`), which is how
they were developed and tested as a flat directory. Putting this directory on
sys.path keeps that working when the package is imported as `bp.*`, so the
service and the test harness run byte-identical code.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
