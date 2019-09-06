from selenium import webdriver

class crawling:
        def __init__(self , url):
            self.url=url
        def melon_crawler(self):
            driver_path = './resources/chromedriver'  #driver #main에서 실행시 ./resources이다?
            url_path = self.url
            browser = webdriver.Chrome(executable_path=driver_path)
            browser.get(url_path)
            page = browser.page_source
            browser.quit()
            return page




