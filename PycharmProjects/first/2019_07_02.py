"""
2019.07.02
"""

# sort()

test_list = [ 20, -2, 4, 6]
test_list_s = ['aa','hello','my','world']

print(test_list)
print(test_list_s)

test_list.sort()
test_list_s.reverse()

print(test_list)
print(test_list_s)

# tuple list 와 다르게 수정이 불가,
# 값을 도출할때 주로 사용

t=(1,4)
print(t)

# object, class, and instance

# 서로 주소 같다 imutable -> 만약 하나가 변하면 새로운 주소로 만듬
a=1
b=1
print(id(a),id(b))
b=b+1
print(id(b))

# 서로 주소 다르다 mutable list여서 서로 각각 다르게 주소를 잡음
a1=[1,2]
b1=[1,2]
print(id(a1),id(b1))

# 즉 mutable은 수정이 가능하다. imutable은 수정이 불가하다.

lst1=['a',[1,2]]
lst2=lst1[:]
lst2[1][1]='d'

print("lst1=%d" %(id(lst1[1][1])))
print("lst2=%d" %(id(lst2[1][1])))

# bool data type
# 0 = flase else true

# not or and 그냥 언어로 쓴다.
# in, not in list안에 있는지 확인

# 제어문

value_to_print=''
for char in 'string_values':
    value_to_print += '\n' if char =='_' else char
print(value_to_print)

# enumerate() = 인덱스, 리스값 반환

#for 문 간소화
a = range(10)
result = []
for i in a:
    result.append(i*10)
print(result)

result=[num*10 for num in a]
print(result)

result=[num*10 for num in range(10)]
print(result)

#
def function1(a,b):
    return (a+b)

sum1=function1(1,2)
print(sum1)

