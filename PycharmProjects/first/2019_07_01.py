# 2019.07.01
# 1. Basic Components of Python

"""
likes  /* */
"""

# easy swap in python
x=1
y=2
(x,y)=(y,x)
print(x,y)

#python ++ (X)

print('python "가능" 하다')

aa='='*30
print(aa)

test_string = ("Hello World!!!")
print(test_string[0:50])

test_string2 = ("lovely")
print('There are %d %s dogs' %(2, test_string2))

# a.count 문자열에서 문자 갯수  a.index 문자열에서 인덱스 반환 a.find index와 같지만 index는 없을 시 error, find는 -1반환
test_count=test_string.count('e')
test_index=test_string.index('e')
test_find=test_string.find('e')
print("count: %d  \nindex: %d \nfind: %d\n" %(test_count,test_index,test_find))

# Python-list  like string 슬라이싱, 인덱싱 가능하다.
test_list = [1,2,3]
test_list2 = list([4,5,6])
test_list3 = list("Hello")
test_list4 =test_list+test_list2

print(test_list)
print(test_list2)
print(test_list + test_list2)
print(test_list4[0:-1:2])

#min 얻기
min = 100000000000
for i in range (2):
    if min<test_list[i]:
        min=test_list[i]


