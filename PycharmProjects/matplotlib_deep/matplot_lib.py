import numpy as np
from matplotlib import pyplot as plt

# x=np.linspace(-np.pi, np.pi, 100)
#
# c=np.cos(x)
# s=np.sin(x)
#
# plt.plot(x,c,'b:o')
# plt.plot(x,s)
#
# plt.show()

# 샘플 1000개 rp.random.randn()
#standard 부ㄴ포 plotting 스타일은 ro

rand_array= np.random.randn(2,1000)
uni_array= np.random.rand(2,1000)*6-3

plt.plot(rand_array[0],rand_array[1],'ro')
plt.plot(uni_array[0],uni_array[1],'g.')
#uniform 분포[-33] 범위 스타일은 g.

plt.show()