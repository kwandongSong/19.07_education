
class dogs:
    def __init__ (self, name, sex, hungry_value):
        self.name=name
        self.sex=sex
        self.hungry_value=hungry_value
    def what_name(self):
        print("dog name is %s\n" %self.name)
    def what_sex(self):
        print("dog sex is %s\n" %self.sex)
    def what_mung(self):
        print("mung mung mung\n")
    def play(self):
        self.hungry_value -= 1
        print("hungry_value -1, current hungry_value= %d\n" %self.hungry_value)
        if(self.hungry_value==0):
            print("----dog die----\n")

class jindo(dogs):
    def __init__(self, name, sex,  hungry_value):
        super().__init__(name, sex,  hungry_value)
    def what_mung(self):
        print("i'm jindo~\n")

class wolf(dogs):
    def __init__(self, name, sex,  hungry_value):
        super().__init__(name, sex, hungry_value)
    def what_mung(self):
        print("owwwww~\n")

class cat(dogs):
    def __init__(self, name, sex,  hungry_value):
        super().__init__(name, sex,  hungry_value)

    def what_mung(self):
        print("ya ong~\n")

#
#
#
#
#
# class Dog:
#     def __init__(self, name, weight, happiness, hunger):
#         self._name = name
#         self._weight = weight
#         self._happiness = happiness
#         self._hunger = hunger
#
#     def play(self):
#         print("%s is playing~" % self._name)
#         print("happiness +1 ==> %d" % self._happiness)
#         self._happiness += 1
#
#     def eat(self):
#         print("%s is eating~" % self._name)
#         print("hunger -1 ==> %d" % self._hunger)
#         self._hunger -= 1
#
#     def bark(self):
#         print("walwal")
#
#
# class Husky(Dog):
#     def __init__(self, name, weight, happiness, hunger):
#         super().__init__(name, weight, happiness, hunger)
#
#     def bark(self):
#         print("grrrr")
#
#
# class Terrier(Dog):
#     def __init__(self, name, weight, happiness, hunger):
#         super().__init__(name, weight, happiness, hunger)
#
#     def bark(self):
#         print("bowwow")
#
#
# class Bulldog(Dog):
#     def bark(self):
#         super().bark()
#
#     def __init__(self, name, weight, happiness, hunger):
#         super().__init__(name, weight, happiness, hunger)
#
#     def play(self):
#         super().play()
#
#     def eat(self):
#         super().eat()
#
# new_dog = Bulldog("10", 10,10,10)
# new_dog.play()