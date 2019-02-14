l=[]
while(True):
    a= input("enter transactions\n")
    if a=='quit':                        #to exit from loop when transactions are completed
        break
    else:
        l+=[a]                           #adding all the transactions to the list l
print(l)
sum = 0
for i in l:
    words = i.split(' ')
    if words[0]=='Deposit':              #adding all the deposits
        sum += int(words[1])
    elif words[0] == 'Withdrawl':        #subtracting the withdrawl amount
        sum-=int(words[1])
    else:
        exit('enter correct transaction')
print("total amount ",sum)                 #final amount