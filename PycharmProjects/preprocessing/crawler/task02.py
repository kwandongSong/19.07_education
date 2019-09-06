"""

여러개 url을 연다.
여러 페이지 에서 크롤링 할 때 유용

"""
from selenium import webdriver
driver_path = '../resources/chromedriver'  # driver path
urls = [
"https://play.google.com/store/apps/category/GAME_EDUCATIONAL",  # categories
"https://play.google.com/store/apps/category/GAME_WORD",
]
browser = webdriver.Chrome(executable_path=driver_path)  # Chrome driver
for url in urls: # for문 list 돌면서 실행
    browser.get(url)
browser.quit()