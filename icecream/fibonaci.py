def fibo(n):
	print ("Fibonaci series")
	a,b = 0,1
	while b < n:
		print (b,)
		a,b = b, a+b
fibo(int(input("Enter number of n: ")))
