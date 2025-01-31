def generate_statistics(emotions):
    stats = {}
    for emotion in emotions:
        if emotion not in stats:
            stats[emotion] = 0
        stats[emotion] += 1
    return stats
