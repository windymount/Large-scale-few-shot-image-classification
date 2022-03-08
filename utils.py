class running_average():
    def __init__(self):
        self.value = 0
        self.counter = 0
    
    def update(self, new_value, num=1):
        self.value = self.value * self.counter + new_value
        self.counter += num
        self.value /= self.counter