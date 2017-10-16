import pickle

data1 = {
	'a' :[1,2,3,0,4 + 6j],
	'b' :('string', 'Unicode string'),
	'c' : None}
list2 = [1,2,7]

list2.append(list2)

output = open('data.pkl','wb')

pickle.dump(data1,output)
pickle.dump(list2,output,-1)
output.close()