import numpy

lst=numpy.random.randint(0,20,15)

print(lst)
list1=list(lst)

print(type(list1))
max_count=0
count=0
j=0
for i in list1:
    count=list1.count(i)
    if max_count<count:
        j=i
        max_count=count

print("most frequent num",j,"its count is",max_count)

