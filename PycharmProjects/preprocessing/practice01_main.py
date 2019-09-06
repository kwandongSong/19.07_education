from parserr import practice01_parser
import re #숫자만 뽑기 위함
from draw_word_cloud import draw
import os #파일 있는지 확인 위함

melon_url = "https://www.melon.com/chart/index.htm"

melon = practice01_parser.melon_parser(melon_url)
soup=melon.melon_parsing() #url전체 가져옴
links=soup.find_all('a',{'class':'btn button_icons type03 song_info'}) #멜론 곡정보 url 담겨있는 a 태그 class = btn button_icons type03 song_info

#%%%%%%%%%%%%맞춤정보 지워야함%%%%%%%%%%%%%%%%
top100_flag=0 # top100만을 뽑기위해서 처음꺼 맞춤추천곡 건너뛰기 위한 flag
song_count=0 #song_count <10 오래걸려서 10개까지만.. 최대 50

# file이 있으면 삭제 후 시작 , 없으면 그냥 시작
file = './genre.txt'
if os.path.isfile(file):
    os.remove(file)

for link in links:
    if top100_flag == 1 and song_count < 10: #flag 1일시 데이터 뽑아 온다.
        new_url = 'https://www.melon.com/song/detail.htm?songId=' #songid를 더해서 new url을 만든다.
        song_count += 1
        get_songid=link.get('href') #get사용
        get_songid_num=re.findall("\d+", get_songid) # javascript 주소에서 songid만 뽑아냄
        new_url=new_url+str(get_songid_num[0]) #각 곡의 곡정보 있는 url 생성
        genre=practice01_parser.melon_parser(new_url) # 새로운 url에서 파싱함.
        soup2=genre.melon_parsing()
        links2=soup2.find_all('dl',{'class':'list'})[0] #<dl> 중에서 class list인 거 찾음
        links2=str(links2)

        #pharsing 부분 엔터로 구분해서 split 후 <dd></dd> 사이 string을 뽑아내기 위함
        #R&amp;B 로 나오는 이유?? 인코딩 문제인듯
        get_genre=links2.split('\n')
        get_real_genre=get_genre[6] #split한 리스트 중 6번째에 장르는 위치함
        get_real_genre=get_real_genre.split('>')
        get_real_genre = get_real_genre[1]
        get_real_genre = get_real_genre.split('<')
        get_real_genre = get_real_genre[0]

        # file open 후 write 'a' 로 append를 한다.
        f=open('./genre.txt',"a",encoding="UTF-8")
        get_real_genre=get_real_genre+' ' #그냥 쓰면 word를 인식 할 때 붙어서 써져있어 발라드댄스힙합 이런식으로 인식해버린다. 그래서 공백 추가
        f.write(get_real_genre)

        print(song_count)
        print(get_real_genre)
        print('\n')


    top100_flag=1 # 맞춤정보 거르기 위한 플래그 1

f.close()
#jpg 파일 있으면 삭제 아니면 고
file2 = './word_cloud.jpg'
if os.path.isfile(file2):
    os.remove(file2)
# word_cloud draw 부분
ddraw=draw.draw_cloud()
ddraw.drawing()

#print(links)
#songid를 크롤링 해서 곡 정보 들어가 장르 받아와야 할 듯
#여기서 실행하려면 driver path가 ./resources이다???