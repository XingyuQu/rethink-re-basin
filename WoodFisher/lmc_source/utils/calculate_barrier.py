def calculate_barrier(lmc_stat, metric):
    """Calculate the barrier for a given metric.

        lmc_stat: array-like
        metric: 'loss', 'top1', 'top5'
    """
    if metric in ['loss']:
        barrier = max(lmc_stat) - (lmc_stat[0] + lmc_stat[-1])/2
    elif metric in ['top1', 'top5']:
        barrier = (lmc_stat[0] + lmc_stat[-1])/2 - min(lmc_stat)
    else:
        raise ValueError(f'Invalid metric {metric}')

    return barrier.item()
