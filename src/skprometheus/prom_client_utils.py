
def observe_many(metric, observations):
    for observation in observations:
        metric.observe(observation)


def add_labels(metric, labels=None):
    if not labels:
        return metric
    return metric.labels(**labels)
