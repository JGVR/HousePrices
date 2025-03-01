class House:
    def __init__(self, price, sqtf):
        self.price = price
        self.sqtf = sqtf
    
    def __eq__(self, other):
        return self.price == other.price and self.sqtf == other.sqtf