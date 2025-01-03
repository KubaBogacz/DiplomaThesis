def customDtwPlotTwoWay(dtw_obj, xts=None, yts=None,
                        offset=0,
                        ts_type="l",
                        match_indices=None,
                        match_col="gray",
                        xlab="Index",
                        ylab="",
                        **kwargs):
    
    import matplotlib.pyplot as plt
    from matplotlib import collections as mc
    import numpy as np

    if xts is None or yts is None:
        try:
            xts = dtw_obj.query
            xts_mean = np.mean(xts)
            xts_min = np.min(xts)
            xts_max = np.max(xts)
            yts = dtw_obj.reference
            yts_mean = np.mean(yts)
            yts_min = np.min(yts)
            yts_max = np.max(yts)
        except:
            raise ValueError("Original timeseries are required")

    # ytso = yts + offset
    offset = -offset

    xtimes = np.arange(len(xts))
    ytimes = np.arange(len(yts))

    # Increase figure size by 1.5 times
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    
    # ax.set_xticks(range(6))
    # ax.set_xticklabels([500 * i for i in range(6)])
    ax.set_xticks(np.arange(0, 301, 50))
    ax.set_xticklabels([f'{i * 500}' for i in range(7)])
    
    ax.set_yticks([xts_min, xts_mean, xts_max])

    ax.axhline(y=xts_mean, color='r', linestyle='--', linewidth=0.5, label='xts_mean')
    ax.axhline(y=yts_mean - offset, color='g', linestyle='--', linewidth=0.5, label='yts_mean')
    
    
    ax.plot(xtimes, np.array(xts), color='k', **kwargs)
    ax.plot(ytimes, np.array(yts) - offset, color='k', **kwargs)  # Plot with offset applied

    if offset != 0:
        # Create an offset axis
        ax2 = ax.twinx()
        ax2.tick_params('y', colors='k')
        ax2.set_yticks([yts_min , yts_mean, yts_max])
        ql, qh = ax.get_ylim()
        ax2.set_ylim(ql + offset, qh + offset)

    if match_indices is None:
        idx = np.linspace(0, len(dtw_obj.index1) - 1)
    elif not hasattr(match_indices, "__len__"):
        idx = np.linspace(0, len(dtw_obj.index1) - 1, num=match_indices)
    else:
        idx = match_indices
    idx = np.array(idx).astype(int)

    col = []
    for i in idx:
        col.append([(dtw_obj.index1[i], xts[dtw_obj.index1[i]]),
                    (dtw_obj.index2[i], -offset + yts[dtw_obj.index2[i]])])

    lc = mc.LineCollection(col, linewidths=1, linestyles=":", colors=match_col)
    ax.add_collection(lc)

    return ax


def dropDtwPlotTwoWay(xts=None, yts=None,
                      offset=0,
                      matched_indices=None,
                      match_col="gray",
                      xlab="Index",
                      ylab="",
                      dropped1=None,
                      dropped2=None,
                      **kwargs):
    
    import matplotlib.pyplot as plt
    from matplotlib import collections as mc
    import numpy as np

    #Exclude dropped values from matched indeces
    match_indices = [(i, j) for i, j in matched_indices if i not in dropped1 and j not in dropped2]
    
    if xts is None or yts is None:
        raise ValueError("Original timeseries are required")
    
    xts_mean = np.mean(xts)
    xts_min = np.min(xts)
    xts_max = np.max(xts)
    yts_mean = np.mean(yts)
    yts_min = np.min(yts)
    yts_max = np.max(yts)

    offset = -offset

    xtimes = np.arange(len(xts))
    ytimes = np.arange(len(yts))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    
    ax.set_xticks(np.arange(0, 301, 50))
    ax.set_xticklabels([f'{i * 500}' for i in range(7)])
    
    ax.set_yticks([xts_min, xts_mean, xts_max])

    ax.axhline(y=xts_mean, color='r', linestyle='--', linewidth=0.5, label='xts_mean')
    ax.axhline(y=yts_mean - offset, color='g', linestyle='--', linewidth=0.5, label='yts_mean')
    
    ax.plot(xtimes, np.array(xts), color='k', **kwargs)
    ax.plot(ytimes, np.array(yts) - offset, color='k', **kwargs)

    if offset != 0:
        ax2 = ax.twinx()
        ax2.tick_params('y', colors='k')
        ax2.set_yticks([yts_min , yts_mean, yts_max])
        ql, qh = ax.get_ylim()
        ax2.set_ylim(ql + offset, qh + offset)

    col = []
    for i, j in match_indices:
        if i < len(xts) and j < len(yts):
            col.append([(i, xts[i]), (j, -offset + yts[j])])

    lc = mc.LineCollection(col, linewidths=1, linestyles=":", colors=match_col)
    ax.add_collection(lc)

    if dropped1 is not None:
        for d in dropped1:
            if d < len(xts):
                ax.plot(d, xts[d], 'ro', label='Dropped from series1' if d == dropped1[0] else "")

    if dropped2 is not None:
        for d in dropped2:
            if d < len(yts):
                ax.plot(d, yts[d] - offset, 'bo', label='Dropped from series2' if d == dropped2[0] else "")

    return ax


def dropDtwPlotThreeWay(xts, yts, matched_indices=None,
                        match_col="#1f77b4",
                        xlab="Query index",
                        ylab="Reference index",
                        dropped1=None,
                        dropped2=None,
                        **kwargs):
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import collections as mc
    import numpy as np
    
    if xts is None or yts is None:
        raise ValueError("Original timeseries are required")

    # Exclude dropped values from matched indices
    match_indices = [(i, j) for i, j in matched_indices if i not in dropped1 and j not in dropped2]

    nn = len(xts)
    mm = len(yts)
    nn1 = np.arange(nn)
    mm1 = np.arange(mm)

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3], height_ratios=[3, 1])
    axr = plt.subplot(gs[0])
    ax = plt.subplot(gs[1])
    axq = plt.subplot(gs[3])

    axq.plot(nn1, xts, **kwargs)  # query, horizontal, bottom
    axq.set_xlabel(xlab)

    axr.plot(yts, mm1, **kwargs)  # ref, vertical
    axr.invert_xaxis()
    axr.set_ylabel(ylab)

    # Mark dropped values in the query plot
    if dropped1 is not None:
        axq.scatter(dropped1, [xts[i] for i in dropped1], color='r', zorder=5, label='Dropped Query Points')

    # Mark dropped values in the reference plot
    if dropped2 is not None:
        valid_dropped2 = [d for d in dropped2 if d < len(yts)]
        axr.scatter([yts[i] for i in valid_dropped2], valid_dropped2, color='b', zorder=5, label='Dropped Reference Points')

    if match_indices is None:
        idx = []
    else:
        idx = np.array(match_indices).astype(int)

    col = []
    for i, j in idx:
        col.append([(i, 0), (i, j)])
        col.append([(0, j), (i, j)])

    # lc = mc.LineCollection(col, linewidths=1, linestyles=":", colors=match_col)
    # ax.add_collection(lc)

    # Plot the warping path as a line following matched indices
    if match_indices:
        path_x, path_y = zip(*match_indices)
        ax.plot(path_x, path_y, color=match_col)

    return ax