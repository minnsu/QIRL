# 시계열 데이터를 순서대로 반환

class environment:
    def __init__(self, chart, close_price_idx):
        self.chart = chart
        self.day_idx = -1
        self.cp_idx = close_price_idx
        self.recent_data = None
    
    def reset(self, chart=None):
        if chart is not None:
            self.chart = chart
        self.day_idx = -1
    
    def next(self):
        self.day_idx += 1
        if self.day_idx < len(self.chart):
            return True
        else:
            reset()
            return False

    def get(self):
        if next():
            self.recent_data = self.chart.iloc[self.day_idx]
            return self.recent_data
        else:
            return None

    def price(self):
        return self.recent_data.iloc[self.cp_idx]