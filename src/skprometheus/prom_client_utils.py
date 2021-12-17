
def observe_many(metric, observations):
    for observation in observations:
        metric.observe(observation)


def add_labels(metric, labels):
    if not labels:
        return metric
    return metric.labels(**labels)
