ten_thing = "Apples Oranges Crows Telephone Light Sugar"
print "Wait there are not 10 things in that list. Let's fix that."

stuff = ten_thing.split()
more_stuff_ = "Day Night Song Frisbee Corn Banana Girl Boy"
more_stuff = more_stuff_.split()

while len(stuff) !=10:
	next_one = more_stuff.pop()
	print "Adding: ",next_one
	stuff.append(next_one)
	print "There are %d items now." % len(stuff)

print "There we go: ",stuff
print "Let's do some things with stuff."

print stuff[1]
print stuff[-1]

print stuff.pop()
print ' '.join(stuff)
print "#".join(stuff[3:5])
print more_stuff