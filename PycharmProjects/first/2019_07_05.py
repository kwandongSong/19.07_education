"""
2019_07_05

"""


# dictionary value 값으로 정렬 하기
# operator를 사용한다.


# import operator
#
# test_dict = {'d' :5 , 'c':1, 'k':3, 'a':2}
#
# sorted_x = sorted(test_dict.items(),key=operator.itemgetter(1))
# print(sorted_x)


"""

numpy

"""

# 따로 지정안하면 32가 default

import numpy as np
#
# list_var =[0, 1, 2, 3, 40,]
# arr_var =np.array(list_var, dtype='int8')
# print(arr_var)
# print(arr_var.dtype)
#

# # numpy array는 띄어쓰기로 구분된다.
# # 0,1,2 데이터 타입이 다르면 string으로 통일 시켜준다.
# list_var2 =[0, 1, 2, 'aa', 'bb',]
# arr_var2 =np.array(list_var2)
# print(arr_var2)
# print(arr_var2.dtype)

#astype 타입 변환
# h=np.array([1,2,3])
# print(h.dtype)
#
# i=h.astype(np.float16)
# print(i)

# np.zeros, np.zeros_like(array) array 와 같은 크기지만 성분은 다 0으로

# h= np.array([(1,2,3), (4,5,6)])
# k=np.zeros_like(h)
#
# print(h)
# print(k)

#np.arange  리스트 range와 다르게 정수, 소수까지 사용 가능

#np.linspace 동일 간격으로 x~y까지 잘라줌

#numpy는 for loop 없이 한번에 연산 가능
# a=[1,2,3]
# b=[4,5,6]
# a_np= np.array(a)
# b_np=np.array(b)
# result_np= a_np + b_np
# print(result_np)

# 큰장점
#
# a=[1,3,5]
# b=[5,3,1]
# # a=a*3
# #1,3,5 가 세번 반복
#
# a_np= np.array(a)
# b_np=np.array(b)
#
# # numpy는 직관적으로 계산 가능하다.
# a_np=a_np*3
# print(a_np)
#
# a=np.array([[1,3,5],[2,4,6],[3,6,9]])
#
# c=a*a
# print(c)

# 실습
# np_a=np.array([[1,4,2,3,4,5],[2,5,6,5,2,1]])
# np_b=np.array([1,2,3,4,5,6])
# np_c=np.array([3.5,2.4])
# np_d=np.ones_like(np_a) * 2.2
#
# print(np_a)
# print(np_b)
# print(np_c)
# print(np_d)
#
# result= np_b.dot(np_c.dot((np_a*3-np_b)))
# print(result)

# b=np.array([[1,3,5,7],[2,4,6,8],[3,6,9,12]])
# print(b[:,:3])
# a=np.array([range(11,17),range(17,23),range(23,29),range(29,35)])
#
# print(a[:2,3:6])
#
# print(a[::2])
#
# print(a[:,4])
#
# print(a[:,1::3])
#
# print(a[0::2,1::2])


#idexing

palette =np.array([[0,0,0],[255,0,0,],[0,255,0],[0,0,255],[255,255,255]])

image=np.array([[0,3,2,3],[0,1,4,0]])

print(palette[image])

#x.reshape(x,y) x.transpose = x.T

print(np.array([5,]))

#dot matmul