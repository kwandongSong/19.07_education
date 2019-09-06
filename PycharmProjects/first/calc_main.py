from calc import addition

print("outside if statement", __name__)
if __name__ == '__main__':
    print(addition(100, 200))
    print("inside if statement", __name__)

