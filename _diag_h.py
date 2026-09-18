# Temporary: H field statistics across LODs (deleted after use).
import sys
import numpy as np
sys.path.insert(0, '/home/niclas/terrain_explorer')
import _port_waterways as P

for lon, lat, tag in ((20.25, 10.4, 'lowland'), (72.16, 14.5, 'focus'), (20.0, -30.0, 'south'), (140.0, 45.0, 'east')):
    for fp, exact in ((.24, False), (.06, False), (.012, False), (None, True)):
        r = P.render('tmp3_' + tag, lon, lat, 24.0, 260, fp=fp, exact=exact)
        v = r['fields']['J']
        q = np.quantile(v, [.5, .9, .95, .99])
        print('%-8s %-6s J[p50=%.3f p90=%.3f p95=%.3f p99=%.3f max=%.3f]' % (tag, 'exact' if exact else ('%.3f' % fp), q[0], q[1], q[2], q[3], v.max()))
