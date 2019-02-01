items = []
min=int(input("enter lower limit"))
max=int(input('enter upper limit'))
for i in range(min, max):
    s = str(i)
    if (int(s[0])%2!=0) and (int(s[1])%2!=0) and (int(s[2])%2!=0):
        items.append(s)
print( ",".join(items))