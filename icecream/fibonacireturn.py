def fibo2(n):
	print "Fibonaci series:"
	result  = []
	a,b = 0,1
	while b < n:
		result.append(b)
		a,b = b,a+b
	return result
f100 = fibo2(int(raw_input("Enter number of n: ")))
print f100

def f(a, L = []):
	L.append(a)
	return L

print f(1)
print f(2)
print f(3)