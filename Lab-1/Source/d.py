def longest_substring(s):

    a = []                  #for current
    b = []                  #for all substrings
    for c in s:
        if c in a:
            b.append(''.join(a))            #finds a char already in the array, it maintains a copy of the array
            d = a.index(c) + 1
            a = a[d:]                       #cuts off beginning of it up to that character and keeps building it.
        a += c
    b.append(''.join(a))

    longest = max(b, key=len)       #max of all substrings is calculated based on length as parameter(key)
    print('b is:',b)                   #print all the substrings found
    return longest
    #print(b, len(b))

longest = longest_substring("pwwkew")
print(longest, len(longest))