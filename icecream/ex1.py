my_name = 'Zed A. Shaw'
my_age = 35 #not a line
my_height = 74.8 #inches
my_weight = 180
my_eyes = 'Blue'
my_teeth = 'white'
my_hair = 'Brown'

print "Let's talk about %s." % my_name
print "he's %.2f inches tall." % my_height
print "he's %d pounds heavy." % my_weight
print "actually that's not too heavy."
print "he's got %s eyes and %s hair" % (my_eyes,my_hair)
print "his teeth are usually %s depending on the coffee." % my_teeth

# this line is tricky, try to get it exactly right
print "if i add %d, %d, and %d I get %d." % (
	my_age, my_height, my_weight,my_age + my_height + my_height
	)