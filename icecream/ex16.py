from sys import argv
from os.path import exists

scirpt,from_file,to_file = argv

open(to_file,"w").write(open(in_file.read()))
