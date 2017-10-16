from sys import argv

script, file_name = argv

target=open(file_name,"r")

print "we are going to delete the content of the file %r" % file_name
print "hit ctrl c to exit and enter to continue"
raw_input("?")
print "This file %r was deleted" % file_name
target.truncate()

print " We are going to add line on the file %r " % file_name
line1=raw_input("line 1: ")
line2=raw_input("line 2: ")
line3=raw_input("line 3: ")
target.write(line1 + "\n" + line2 + "\n" + line3 + "\n")

print "This is the end of task 1.We close it"
target.close()