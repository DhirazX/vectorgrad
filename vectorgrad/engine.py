class Value:
    def __init__(self, data):
        self.data = data

    def __add__(self, other):
        out = Value(self.data + other.data)
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data)
        return out


a = Value(2.0)
b = Value(-3.0)
c = a + b
print(c.data)
