from konlpy.tag import Hannanum
from collections import Counter
import pytagcloud # Add fonts supporting Korean

f = open("../crawler/description.txt", "r", encoding="UTF-8")
# r w a(append) 파일 원래 있던거 뒤에 붙임
description = f.read()
h = Hannanum()
nouns = h.nouns(description) #명사화
count = Counter(nouns) #단어 출현 빈도수
print(count)
tag = count.most_common(100) #가장 빈도수 많은 100개 뽑음
tag_list = pytagcloud.make_tags(tag, maxsize=50) #tag list를 만듬
pytagcloud.create_tag_image(tag_list, 'word_cloud.jpg', size=(900, 600), fontname='Korean', rectangular=False)
#어떤 이름의 파일, 사이즈, 폰트 네임
import webbrowser
webbrowser.open('word_cloud.jpg')
# 웹브라우저로 그림 띄우기