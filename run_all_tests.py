import unittest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "test"))
from test import *

if __name__ == '__main__':
    unittest.main(verbosity=2)
