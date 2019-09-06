"""
practice 2

"""

lines = open('example.txt', 'r').readlines() # lines 에 엔터 구별  하나씩 들어감
lines = [line.strip()for line in lines]
words_dict={}

for sentence in lines: # 공백 파싱
    count = 0
    for word in sentence.split(' '):
        words_dict[word] = words_dict.get(word,0)+1 # value를 default 0으로, 그 후 +=1

for key in sorted(words_dict.keys()): # sort for 문
    if words_dict[key] >= 10:
        print(key, words_dict[key])


#test_dict =defaultdict(int)

#value값을 int로 디폴트 해놓겠다, - 0 이됨
