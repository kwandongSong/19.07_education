import copy

list=[1, 2, 5, 2, 0]
list2=copy.deepcopy(list)

list.reverse()
#== [::-1]
if list==list2:
    print("good")
else:
    print("bad")