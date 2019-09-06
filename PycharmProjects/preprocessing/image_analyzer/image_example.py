from skimage import data, io
from matplotlib import pyplot as plt
camera = data.camera() # camera image ; grey scale 512x512
io.imshow(camera)
plt.show() # image 떠있어서 뒤에 코드 실행 x

print(type(camera), camera.shape)
print(camera)

cat = data.chelsea() # cat is a 300-by-451 pixel image with three channels (red, green, and blue)
io.imshow(cat)
plt.show()

print(type(cat), cat.shape)
print(cat)