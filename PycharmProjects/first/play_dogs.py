from dogs import *

dog=dogs('unknown', 'female', 10)
cat=cat('cat', 'female', 10)

dog.what_mung()
cat.what_mung()


# from dogs import *
#
# class Dog:
#     def __init__(self, name, weight, happiness, hunger):
#         self._name = name
#         self._weight = weight
#         self._happiness = happiness
#         self._hunger = hunger
#
#     def bark(self):
#         print(".....")
#
# dogs_list = []
# dogs_list.append(Dog("unknown", weight=15, happiness=1, hunger=10))
# dogs_list.append(Terrier("Sam", weight=10, happiness=3, hunger=8))
# dogs_list.append(Husky("Ellie", weight=20, happiness=5, hunger=4))
#
# for dog in dogs_list:
#     dog.bark()
