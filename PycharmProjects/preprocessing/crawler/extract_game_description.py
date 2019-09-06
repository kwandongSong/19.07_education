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
   # print(link.a.attrs)  # dictionary 형식으로 저장
    new_url = link.a['href'] # 그 중 a태그 의 href값을 가져옴(링크)
    browser.get(new_url)

    page2= browser.page_source
    soup2 =BeautifulSoup(page2,"html.parser")
    links2 = soup.find_all('div', {'class': 'b8cIId ReQCgd Q9MA7b'})
    print(links2)
    for link2 in links2:

        new_url2= 'https://play.google.com'+link2.a['href']
        print(new_url2)
        browser.get(new_url2)

        page3 = browser.page_source
        soup3 = BeautifulSoup(page3, "html.parser")
        links3 = soup3.find('meta', {'name': 'description'})
        print(links3)

browser.quit()