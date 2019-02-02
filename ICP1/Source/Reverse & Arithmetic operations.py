a=input("Enter first number:")
b=input("Enter second number:")

rev=a[::-1]
print("reverse of first number is:",int(rev))

a=int(a)
b=int(b)
print('arithmetic operations:')
numbers=[a,b]
c=sum(numbers)
print("Sum is:",c)
print("subtraction:",(a-b))
print("multiplication:",(a*b))
print("division:",(a/b))
print("mod:",(a%b))
print("power:",(a**b))