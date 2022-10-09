
"""
# Game Theory - duplicates: 2, total sent (unique): 11, total sent value (unique): 10.198405364824104, total duplicate Value: 2.000518252254722
# Maximum - duplicates: 5, total sent (unique): 11, total sent value (unique): 9.967952582598397, total duplicate Value: 5.736302852082445
# Total not sent messages: 2, total missing value: 1.2943220395748838

"""
import pickle

class Metrics:
    def __init__(self, approach_name):
        self.approach_name = approach_name
        self.number_sent_unique = 0
        self.total_sent_value = 0
        self.number_duplicate = 0
        self.total_duplicate_value = 0

        self.num_not_sent_msgs = 0
        self.total_not_sent_value = 0
        self.approaches_counts = [0, 0, 0]

    def print(self):
        print(f"#{self.approach_name} - total sent (unique): {self.number_sent_unique},"
              f" total sent value (unique): {self.total_sent_value},"
              f" duplicates: {self.number_duplicate},"
              f" total duplicate Value: {self.total_duplicate_value}")

        print(f"#Total not sent messages: {self.num_not_sent_msgs},"
              f" total missing value: {self.total_not_sent_value}")

        if self.approaches_counts != [0, 0, 0]:
            print(f"Dominant approaches: {self.approaches_counts[0]},"
                f"Dominant - other approaches: {self.approaches_counts[1]}, "
                f"Mixed Strategy approaches: {self.approaches_counts[2]}")

    def __add__(self, other):
        ret = Metrics(self.approach_name)
        ret.number_sent_unique = self.number_sent_unique + other.number_sent_unique
        ret.total_sent_value = self.total_sent_value + other.total_sent_value
        ret.number_duplicate = self.number_duplicate + other.number_duplicate
        ret.total_duplicate_value = self.total_duplicate_value + other.total_duplicate_value
        ret.num_not_sent_msgs = self.num_not_sent_msgs + other.num_not_sent_msgs
        ret.total_not_sent_value = self.total_not_sent_value + other.total_not_sent_value
        ret.approaches_counts = [self.approaches_counts[0] + other.approaches_counts[0],
                                 self.approaches_counts[1] + other.approaches_counts[1],
                                 self.approaches_counts[2] + other.approaches_counts[2]]
        return ret

    def __truediv__(self, other):
        if not isinstance(other, int):
            raise TypeError("Only possible division by int")

        ret = Metrics(self.approach_name)
        ret.number_sent_unique = self.number_sent_unique / other
        ret.total_sent_value = self.total_sent_value / other
        ret.number_duplicate = self.number_duplicate / other
        ret.total_duplicate_value = self.total_duplicate_value / other
        ret.num_not_sent_msgs = self.num_not_sent_msgs / other
        ret.total_not_sent_value = self.total_not_sent_value / other
        ret.approaches_counts = [self.approaches_counts[0] / other,
                                 self.approaches_counts[1] / other,
                                 self.approaches_counts[2] / other]
        return ret

class Stats:
    def __init__(self, list_approaches=["max", "gt", "distance", "random"]):
        self.approaches_metrics = {a:Metrics(a) for a in list_approaches}

    def print(self):
        for approach, metrics in self.approaches_metrics.items():
            metrics.print()

    def __add__(self, other):
        ret = Stats(list(self.approaches_metrics.keys()))

        for approach in self.approaches_metrics:
            ret.approaches_metrics[approach] = self.approaches_metrics[approach] + other.approaches_metrics[approach]

        return ret

    def __truediv__(self, other):
        if not isinstance(other, int):
            raise TypeError("Only possible division by int")

        ret = Stats(list(self.approaches_metrics.keys()))
        for approach in self.approaches_metrics:
            ret.approaches_metrics[approach] = self.approaches_metrics[approach] / other

        return ret