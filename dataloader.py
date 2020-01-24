class SlidedDataLoader:
    def __init__(self, data, slides):
        self.slides = slides
        self.data = data
        self.slide_size = int(len(self.data) / slides)
        self.current_index = 0

    def __getitem__(self, index):
        return self.data[index * self.slide_size, index * self.slide_size + self.slide_size]

    def __iter__(self):
        return self

    def __len__(self):
        return self.slides
