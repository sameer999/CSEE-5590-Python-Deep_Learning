with open('abc.txt') as f:
  lines = f.readlines()
  print(lines)
  print('lines =', len(lines))
  for i in lines:
    data = ''.join(i)
    #print(data)

    print('Words = ',len(data.split()))
    char = ''.join(data.split())
    print('characters = ',len(char))