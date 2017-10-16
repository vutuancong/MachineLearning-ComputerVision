
from sys import argv

script, filename =argv


print "this is the name of file: %r" % filename
print open(filename).read()

print open(raw_input("Writing file 2")).read()

