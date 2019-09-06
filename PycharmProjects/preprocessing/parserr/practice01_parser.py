from bs4 import BeautifulSoup
from crawler import practice01_crawler

class melon_parser:
    def __init__(self, url):
        self.url = url
    def melon_parsing(self):
        crawler=practice01_crawler.crawling(self.url)
        page = crawler.melon_crawler() #멜론 url 들어가서 크롤링
        # 예제로 html 코드 넣어 놈
        soup = BeautifulSoup(page, 'html.parser')
        return soup # html 코드를 구분 쉽게 정렬 되어서 나옴
#
#
# url="https://www.melon.com/chart/index.htm"
#
# melon=melon_parser(url)
# melon.melon_parsing()



