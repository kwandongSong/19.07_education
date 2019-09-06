# 126명 선수 기럭
# 열 안타 홈런 볼넷 삼진 도루 타율
#
# 데이터를 nupmpy array로 불러오ㅗ plotting
# 음수값=0 o
# 홈런수가 1이상인것만 o
# 세가지 비울 계산
#
# 홈/안 볼/안 삼/안
#
# x푹 홈/안 1 y축 볼/안 2
# rx스타일
#
# x축 홈/안 1 y 삼/안 3
# b.스타일
#안 홈 볼 삼 도 타
from matplotlib import pyplot as plt
import numpy as np

data_arr=np.loadtxt('baseball_data.txt',delimiter=',')
data_arr[data_arr<0]=0

persent1_arr=np.zeros(126)
persent2_arr=np.zeros(126)
persent3_arr=np.zeros(126)

# for i in range(0,126):
#     if data_arr[i][1]<1:
#         np.delete(data_arr, i)
#
#     persent1_arr[i]=data_arr[i][1]/data_arr[i][0]
#     persent2_arr[i]=data_arr[i][2]/data_arr[i][0]
#     persent3_arr[i]=data_arr[i][3]/data_arr[i][0]

data_arr=data_arr[data_arr[:,1]>0]

h,hr,bb,k,sb,avg = data_arr.T

hr_h=hr/h
hr_bb=bb/h
hr_k=k/h

hr=np.max(hr)
hr_bb=np.max(hr_bb)
hr_k=np.max(hr_k)

plt.plot(hr_h,hr_bb,'rx')
plt.plot(hr_h,hr_k,'b.')

plt.show()