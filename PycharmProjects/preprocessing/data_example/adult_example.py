import pandas as pd
df = pd.read_csv('adult.data', header=None) #read_csv adult.data 를 가져오고 헤더는 no, 따로 경로 지정 안하면 같은 패키지에 있다 가정

# data basic
print(df.size) # data의 elements의 갯수
print(df.shape) # 매트릭스 형태 ?x?
print(df.columns) #컬럼 따로 이름 안줬기 때문에 순서대로 0~ 으로 defalut
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'wage']
# column 명을 넣어줌
print(df.columns) # 바뀜
print(df.dtypes) # column별로 data type
print(df.head()) # 제일 위의 5개 instance
print(df.tail()) # 제일 밑에 5개
# head랑 tail로 데이터 대략 볼 수 있다.

# data summary
print(df.describe()) # column별로 요약 정보
print(df.mean()) #numerical 만 column별로 평균값
print(df.mode()) #카테고리 별로 제일 많은 ???

# Details
print(df.education.unique()) #유니크한 값 경우가 뭐 있나
print(df.education.value_counts()) #유니크한 경우에대해 몇개 씩 있나
print(df['wage'].value_counts()) #
print(df.groupby(['wage'])['age'].mean())
print(df.groupby(['wage'])['age'].std())
print(df['capital-gain'].corr(df['age']))