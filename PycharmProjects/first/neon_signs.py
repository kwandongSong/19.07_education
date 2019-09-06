
str=input("plese enter the string:")

odd_seq=''
even_seq=''

for i in str[::2]:
    odd_seq += i + '='
for i in str[1::2]:
    even_seq += '=' + i

even_seq += "="*(len(odd_seq)-len(even_seq))
border_line = '-+'*(len(odd_seq)//2)

print('|'+border_line+'|')
print('|'+odd_seq+'|')
print('|'+even_seq+'|')
print('|'+border_line+'|')


