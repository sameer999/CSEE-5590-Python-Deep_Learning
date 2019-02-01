with open('abc.txt') as f:
  lines = f.readlines()
  data = ''.join(lines)
  print('lines =',len(lines))
  print('Words = ',len(data.split()))
  char = ''.join(data.split())
  print('characters = ',len(char))