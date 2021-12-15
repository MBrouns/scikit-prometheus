def probas_to_metric(metric, probas, classes, **labels):
    print(classes)
    for idx, class_ in enumerate(classes):
        for proba in probas:
            metric.labels(class_=class_, **labels).observe(proba[idx])
