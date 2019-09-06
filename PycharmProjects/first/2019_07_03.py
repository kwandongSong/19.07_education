'''
2019_07_03
'''

# Function
# 람다 함수 - 이름이 필요 없는 일회용 함수

start_index=input("Enter start value:")
end_index=input("Enter end value:")

def check_sum(a,b):  # header
    s=0                     #
    for i in range(a,b):    #
        s += i              # body
    print(s % 5 == 0)       #
    return s%5, s, i
check_sum(int(start_index),int(end_index))


# set default parameter value
def test_para(a=5):
    print(a)

test_para(19)
test_para() #함수 호출할 떄 파라미터를 설정 안해줘도 됨


# possible multiple return value
print(check_sum(1,10))

# global, local variable

global_var=1
another_var=2
print("처음 another_var 주소 %d" %id(another_var))
def test_variable():
    another_var=3
    print("함수내에서 만 변한다: %d %d" %(another_var, id(another_var)))

test_variable()
print("변하지 않는다 %d %d" %(another_var, id(another_var)))

# 제어문이나 반복문에서 생성된 변수는 전역 변수임
for i in range(4):
    print(i)

print(i)

# pass by value pass by reference
# call by value  call by reference


# parameter가 그 형식의 데이터 타입이 맞는지 확인 해주는 assert isinstance
test_list=[1, 2, 3, 4, 5]

def repair_list(test_list):
    # 이걸 해주면 이 형식이 아니면 에러가 나온다.
    assert isinstance(test_list,list)
    test_list.append(0)
    test_list.append(-1)

repair_list(test_list)
print(test_list)

#뭔가 복사 할때는 copy.deepcopy 가 좋다 완전히 복사해줌

c=[3 ,4 [-5, -10]]

def test5(x):
    x[2][0] =100
    x += [30, 40]
    print("inside the function, x is", x)
    print("inside the function, x's id is", id(x))

print("Before the call c is", c)
print("Before the call c's id is", id(c))
test5(c[:])  #test5(copy.deepcopy(c))
print("After the call c is", c)
print("After the call c's id is", id(c))


#
