"""

2019_07_04

"""


# import 할 시 import하는 x.py의 코드를 한번 쭉 실행.
# 그러므로 모듈 만들때 실행하는 코드는 지양
# from x import y
# 하나만 import 해도 그대로 x 코드를 한번 쭉 실행

#(집합)set 데이터 타입 문자열 순서 맘대로 저장됨

# list_obj=[1, 2, 3 ,4 ,4]
# str_obj='apple'
#
# s1=set(list_obj)
# s2=set(str_obj)
#
# print(s1)
# print(s2)


"""

실습

"""
# test_list1=[10, 20, 30, 40, 30, 20, 10]
# test_list2=list(range(0,100))
# test_list2=test_list2[::2]
#
# s1=set(test_list1)
# s2=set(test_list2)
#
# s3=s1.union(s2)
# s4=s1.intersection(s2)
#
# test_list1=list(s3)
# test_list2=list(s4)
#
# test_list1.sort()
# test_list2.sort()
#
# print(test_list1)
# print(test_list2)

"""

Dictionary 데이터 타입 

"""
#순서는 지켜 지지 않는다, sort하고 싶다면 key 값을 sort 하고 그 후 value를 key를 가지고 얻는다.
#key= 키값 , value는 값, items는 key와 값 둘다 가져 올 수 있다.

test_dict = dict(a=1, b=2, c=4)
#temp_list = test_dict.keys()

for key in test_dict.keys():
    print(key)

for value in test_dict.values():
    print(value)

for key,value in test_dict.items():
    print(key, value)


# in, not in 키나 value 가 있는 지 확인 가능