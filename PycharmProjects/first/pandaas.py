"""

2019_07_10

"""
import numpy as np
import pandas as pd

#
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a','b','c','d'])
data_s= pd.Series(data)
print(data_s)
# index를 range로 보여준다.
# value보다 index가 많으면 broadcasting 처럼 작동 (부족한곳 채워줌)
print(data.index)
print('\n')

#pandas 는 numpy 구조체 사용한다. type= numpy array
area_dict={'california': 423967, 'Texas': 695662}
area= pd.Series(area_dict)

print(area)
print('\n')
print(type(area.values))
print('\n')

#1d-array 와 가장 큰 차이점 index를 명시적으로 정의 가능

#Series보다 dataframe을 더 많이 쓴다.
#dataframe 각 열이 이름이 있다. 행과 열에 이름이 부여된 2-d array
area={'california': 423967, 'Texas': 695662}
population={'california': 267, 'Texas': 562}

states = pd.DataFrame({'population':population, 'area':area})
print(states)

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print(states['area'])
print(states.area)

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print(states['area']['Texas']) # texas 값 갖고 옴.

# numpy는 이런 연산 가능
states['density']=states['population']/states['area']
print(states['density'])

# row-wise selection
# iloc 첫 row index 번호가 0이며 1씩 증가 포함 안해서 가져옴
# 첫 column의 index 번호도 0부터 1씩 증가

#loc- 명시적 인덱스 사용 포함해서 가져옴

#ix는 지양

#iloc loc로 데이터 수정 가능

#numpy처럼 조건 true false 표현
print(states['density']>100)
states.loc[states.density<100,['density']]=10
print(states)

#dataframe scalar 값 연산 가능
print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print(states)
print(states-500)
# 행렬 크기 다르면 0으로 처리하고 계산 하지않고 에러 안나고 NaN으로 표출

#dropna() na를 삭제 새로운 instance를 반환
#fillna() 도 원래 객체는 변경하지 않는다.
#원래 함수 수정하려면 inplace =True 설정

#csv파일 다루기
df= pd.read_csv('sample_iris.csv')
#print(df.isnull()) Nan 있는지 판단
#print(df.dropna(how='all')) #Nan 삭제 , 원래 것은 보존, how='all'는 모든것이 Nan일 때 삭제. = 한행에 모든 열이 다 Nan일 때
df= df.fillna('Null')
print(df)

#pd.concat([d3,d4],axis=1) default는 밑으로 axis=1은 옆으로 붙임 ignore_index='True' = index를 원래 있던거 지우고 새로 쓴다.
#join='inner' 이너조인 한다. (교집합) 결과만
#join_axex=[df.]
#join_axex=[df7.index]

#join 기능이 merge로 구현되어 있다.
#row column 개수 다를때 알아서 같은거 잘 매칭해준다.

df10 = pd.DataFrame([{'name':'peter','food':'fish'},{'name':'paul','food':'beans'}])
df11 = pd.DataFrame([{'name':'peter','food':'wine'}])
print('%%%%%%%%%%%%%%%원본%%%%%%%%%%%%%%%%')
print(pd.merge(df10,df11))
print('%%%%%%%%%%%%%%%inner%%%%%%%%%%%%%%%%')
print(pd.merge(df10,df11, how='inner'))
print('%%%%%%%%%%%%%%%%outer%%%%%%%%%%%%%%%')
print(pd.merge(df10,df11, how='outer'))
print('%%%%%%%%%%%%%%%%left%%%%%%%%%%%%%%%')
print(pd.merge(df10,df11, how='left'))

# dataframe 에서 mean(), describe() 통계값 사용 가능
