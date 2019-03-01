class Oracle:
    def __init__(self, y, max_queries_cnt=20):
        self.labels = y
        self.queries_cnt = 0
        self.max_queries_cnt = max_queries_cnt

    def query(self, i, j):
        if self.queries_cnt < self.max_queries_cnt: # query to search for a must link between i,j
            self.queries_cnt += 1
            return self.labels[i] == self.labels[j]
        else:
            return False

    def can_query(self):
        if self.queries_cnt < self.max_queries_cnt:
            return True
        else:
            return False