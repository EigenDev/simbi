#!/usr/bin/env python3
"""Version info"""

with open('pysimbi/VERSION') as vfile:
    __version__       = vfile.readline()
    
if __name__ == '__main__':
    print(__version__)