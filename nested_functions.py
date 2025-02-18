def functionA(a, b, c):
    value = a
    value = value + functionB(b, c)  + functionC(x = b, y = c)
    return value

def functionB(x, y): #function B does not care what its arguments are called
    return x * y

def functionC(x = 0, y = 0):
	return x + y

# Oop, we forgot to pass any arguments
result = functionA()
print(result)
