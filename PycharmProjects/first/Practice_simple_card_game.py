import random


cards = list(range(1, 14)) * 4
my_cards, dealer_cards = [], []


def draw_a_card(cards):
    '''
    전체 카드의 리스트를 argument로 받음
    카드 리스트에서 임의의 카드 한장을 뽑고 이를 리스트에서 지움
    random.randint() 함수 활용, list 내장함수 pop() 활용
    :return: 뽑은 카드 값
    '''
    random_value=random.randint(1,len(cards))
    pick=cards[random_value]
    cards.pop(random_value)

    return pick



def get_score(my_cards , name):
    str(name)
    score_sum=0


    score_sum= score_sum + my_cards[i]
    print(my_cards[i])


    print("점수는 %d 이름은 %s" %(score_sum, name))
    return score_sum

    '''
    
    가지고 있는 카드들의 리스트를 argument로 받음
    카드 소유주를 나타내는 문자열을 argument로 받음
    합계 점수를 계산, sum() 함수 활용
    현재 보유중인 카드 리스트를 출력
    점수 합산 결과를 카드 소유주가 누구인지와 함께 출력 (print)
    :return: 계산된 합계 점수
    '''



def print_result(my_sum,dealer_sum):
    '''
    나의 점수와 딜러의 점수를 arguments로 받음
    대소 비교를 통해 결과를 출력 (print)
    :return: None
    '''

    if my_sum>dealer_sum:
       print("내가 이겼다 \n")
    elif my_sum==dealer_sum:
       print("draw \n")
    else:
       print("니가 이겼다 \n")



for i in range(2):
    my_cards.append(draw_a_card(cards))
    dealer_cards.append(draw_a_card(cards))

my_sum = get_score(my_cards, 'me')
dealer_sum = get_score(dealer_cards, 'dealer')

print_result(my_sum, dealer_sum)