Python = []                                             #list to maintain students for python class
Web_Application = []                                    #list to maintain student list for web application class
while(True):
    a = input('enter Python student list:')
    if a == 'q':
        break
    else:
        Python += [a]                                   #student names are added to python list

while(True):
    b=input('enter Web Application student list:')
    if b == 'q':
        break
    else:
        Web_Application += [b]                          #student names are added to web application list

print("students attending Web Application class are",Web_Application)
print("students attending Python class are",Python)

print("attending both classes",(set(Python) & set(Web_Application)))        #intersection operator for sets
print("who are not common in both classes",set(Python) ^ set(Web_Application))          #symmetric difference operator for sets
