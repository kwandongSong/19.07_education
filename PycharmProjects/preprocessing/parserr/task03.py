# ctrl shift10
# ctrl 함수, 클래스명 ->이동
# ctrl qq 관련된 문서
from bs4 import BeautifulSoup
html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>
<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>
<p class="story">...</p>
"""
# 예제로 html 코드 넣어 놈
# 트리 구조다
soup = BeautifulSoup(html_doc, 'html.parser')
print(soup.prettify()) # html 코드를 구분 쉽게 정렬 되어서 나옴

print("\n------- Task 3-1 --------\n")
tag = soup.a # a라는 태그를 가진 첫번째 한 줄 dictionary로
print(tag)
print(tag.name)
print(tag.attrs) # 전체 태그의 애트리뷰트, 값
print(tag.string) #tag 사이에 있는 string
print(tag['class'])
print(soup.title) # 타이틀
print(soup.title.name) # 타이틀 이름
print(soup.title.string)
print(soup.title.parent.name)  # parent 하나 윗 단계 (head)
print(soup.title.parent.title.string)
print(soup.head.contents[0].string)  # contents : children as a list children 에 접근 가능

print(soup.p) #처음 p태그
print(soup.p['class'])
print(soup.a) #a 처음

# 여기부터 잘알기
print(soup.find_all('a')) #특정 클래스, 태그, 아이디 접근 할떄 find 애트리뷰트가 어떤거 찾기, find _all x 태그 가진 전체 리스트 형태로
print(soup.find(id='link3')) # id가 lin3인거
print(soup.find(id='link3').string)

#a태그 dictionary 돌며 href 링크 따온다
for link in soup.find_all('a'):
    print(link.get('href')) #get사용
    print(link['href']) #key로 찾기
print(soup.get_text()) # get_text soup에 있는 모든 텍스트 갖고 온다.
                       # 보통 soup을 조금 잘라내고 get_text()를 한다.

# href 찾아서 링크 따올 수 있다.