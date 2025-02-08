

class Rectangle:

    length = 0
    width = 0

    def __init__(self, len, wid):
        self.length = len
        self.width = wid
        
    def calcualte_area(self):
        return self.length * self.width
    

obj = Rectangle(5,6)

print(obj.calcualte_area());