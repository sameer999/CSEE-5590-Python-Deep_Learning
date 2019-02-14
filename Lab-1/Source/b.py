a= [( 'John', ('Physics', 80)) , ('Daniel', ('Science', 90)), ('John', ('Science', 95)), ('Mark',('Maths', 100)), ('Daniel', ('History', 75)), ('Mark', ('Social', 95))]
d={}
for key,value in a:
    d.setdefault(key,[]).append(value)              #appending tuples to dict d
print("Dictionary:",d)
b=sorted(d.items(), key=lambda x:x[0])           #key value pairs are sorted using lambda and sorted functions
print("Sorted:",b)                                  #desired dict with sorted tuples