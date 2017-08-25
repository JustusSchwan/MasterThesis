class GrowingArray(list):
    def __init__(self):
        super(GrowingArray, self).__init__()

    def __call__(self):
        return self

    def clear(self):
        del self[:]
