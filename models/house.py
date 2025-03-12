class House:
    def __init__(self, price, sqtf, rooms, age):
        self.price = price
        self.sqtf = sqtf
        self.rooms = rooms
        self.age = age
    
    def __eq__(self, other):
        return self.price == other.price and self.sqtf == other.sqtf