str1=input("Enter string:")
nums=0
letters=0
words=0
for i in str1:
    if i.isdigit():
        nums+=1
    elif i.isalpha():
        letters+=1
    elif i.isspace():
       words+=1
    else:
        pass
print("Number of letters:",letters)
print("Number of digits:",nums)
print("Number of words",words+1)
