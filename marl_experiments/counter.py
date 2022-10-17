class Counter:
    def __init__(self, x: int = 0):
        self.x = x

    def __add__(self, other):
        self.x += other.x

    def __str__(self):
        return f"Counter({self.x})"
