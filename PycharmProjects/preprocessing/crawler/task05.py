# top chart에 있는 게임의 3종류 chart 요약 되어 있슴
# 다볼려면 see more 버튼 눌르면 다 볼수있다
# 하나 하나 설명 가져오기
# selenium 밑으로 스크롤 하고 기다려서 또 추가되는거 갖고올 수 있는 기능있다.

from selenium import webdriver
from bs4 import BeautifulSoup

driver_path = '../resources/chromedriver'  # driver path
url = 'https://play.google.com/store/apps/top/category/GAME'

browser = webdriver.Chrome(executable_path=driver_path)  # Chrome driver
browser.get(url)
page = browser.page_source

soup = BeautifulSoup(page, "html.parser")
links = soup.find_all('div', {'class': 'W9yFB'})  # find all links to rankings
#div 태그 class =w9yfb 찾음

for link in links:
    print(link.a.attrs)  # dictionary 형식으로 저장
    new_url = link.a['href'] # 그 중 a태그 의 href값을 가져옴(링크)
    browser.get(new_url)

browser.quit()