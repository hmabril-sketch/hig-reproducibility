"""Reference implementation of the intrinsic construction of the HIG (Algorithm 1 of the paper).

construct(image) returns (ids, edges, geom, cuts, carriers, holes, m01, m12):
  ids       region id of every pixel: 4-connected same-label components in raster order of their
            first pixel; OUT receives the id len(regions);
  edges     frontier edges as (vertex, vertex, (region, region), on_frame);
  geom      geometric 1-nodes (continuation classes) with their type ("seg", "enc1", "enc0"),
            region pair and edges;
  cuts      cut nodes (frame-free frontier components, as lists of edge indices);
  carriers  carrier region of every cut node (region of the pixel above the uppermost, then
            leftmost, horizontal edge of the component);
  holes     number of cut nodes of every region (a Counter); on images of at most 10 000 pixels
            it is checked against an explicit search of the bounded 8-connected components of
            the complement of every region, and a ValueError is raised if they differ;
  m01, m12  HIGM_{0,1} (junctions then virtual 0-nodes x geometric then cut 1-nodes) and
            HIGM_{1,2} (1-nodes x regions, OUT last), as scipy sparse matrices.
The compiled version hig_fast.construct_fast returns the same HIG.
"""
from collections import Counter, defaultdict, deque
import numpy as np
from scipy import sparse

def region_ids(image):
    """Four-connected equal-label components; OUT receives the final id."""
    h, w = image.shape
    ids = np.full((h, w), -1, dtype=np.int32)
    nr = 0
    for y in range(h):
        for x in range(w):
            if ids[y, x] != -1:
                continue
            label = image[y, x]
            ids[y, x] = nr
            queue = deque([(y, x)])
            while queue:
                yy, xx = queue.popleft()
                for ny, nx in ((yy-1, xx), (yy+1, xx), (yy, xx-1), (yy, xx+1)):
                    if 0 <= ny < h and 0 <= nx < w and ids[ny, nx] == -1 and image[ny, nx] == label:
                        ids[ny, nx] = nr
                        queue.append((ny, nx))
            nr += 1
    return ids, nr


def construct(image):
    ids, out = region_ids(image)
    h, w = image.shape
    edges = []                 # (first grid vertex, second, region pair, on frame)
    at = defaultdict(list)
    def add(a, b, u, v, frame):
        if u == v:
            return
        i = len(edges)
        pair = tuple(sorted((int(u), int(v))))
        edges.append((a, b, pair, frame))
        at[a].append(i)
        at[b].append(i)
    for y in range(h):
        for x in range(w+1):
            add((y,x), (y+1,x), out if x == 0 else ids[y,x-1],
                out if x == w else ids[y,x], x in (0,w))
    for y in range(h+1):
        for x in range(w):
            add((y,x), (y,x+1), out if y == 0 else ids[y-1,x],
                out if y == h else ids[y,x], y in (0,h))
    junctions = sorted(v for v, incident in at.items() if len(incident) >= 3)
    jset = set(junctions)
    jindex = {v:i for i,v in enumerate(junctions)}
    geom = []
    seen = set()
    for start in range(len(edges)):
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        group = []
        for i in queue:
            group.append(i)
            a,b,pair,_ = edges[i]
            for v in (a,b):
                if v in jset:
                    continue
                for nxt in at[v]:
                    if nxt not in seen and edges[nxt][2] == pair:
                        seen.add(nxt)
                        queue.append(nxt)
        endpoints = []
        for i in group:
            for v in edges[i][:2]:
                if v in jset:
                    endpoints.append(v)
        if len(endpoints) == 0:
            typ = "enc0"
        elif len(endpoints) == 2:
            typ = "enc1" if endpoints[0] == endpoints[1] else "seg"
        else:
            raise ValueError(f"Continuation has {len(endpoints)} endpoint occurrences")
        geom.append({"type":typ,"pair":pair,"endpoints":endpoints,"edges":group})
    cuts = []
    seen = set()
    for start in range(len(edges)):
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        group = []
        frame = False
        for i in queue:
            group.append(i)
            a,b,_,on_frame = edges[i]
            frame |= on_frame
            for v in (a,b):
                for nxt in at[v]:
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
        if not frame:
            cuts.append(group)
    # Pixel-side to frontier edge; supports local validation on small images.
    pixel_sides = defaultdict(list)
    for i,(a,b,pair,_) in enumerate(edges):
        if a[0] == b[0]:
            yy=a[0]; xx=a[1]
            for py in (yy-1,yy):
                if 0<=py<h:pixel_sides[(py,xx)].append(i)
        else:
            yy=a[0]; xx=a[1]
            for px in (xx-1,xx):
                if 0<=px<w:pixel_sides[(yy,px)].append(i)
    # Each frame-free frontier component corresponds to exactly one
    # bounded 8-component of the complement of exactly one 4-region.
    # Establish the carrier by matching its complete R-boundary to the cut.
    # Its uppermost horizontal edge has the carrier pixel immediately above.
    # This gives a linear-time carrier extraction without flooding the
    # complement separately for every region (which is prohibitive for WSI).
    carriers=[]
    for group in cuts:
        horizontal=[i for i in group if edges[i][0][0]==edges[i][1][0]]
        i=min(horizontal,key=lambda k:(edges[k][0][0],edges[k][0][1]))
        yy,xx=edges[i][0]
        carriers.append(int(ids[yy-1,xx]))
    holes=Counter(carriers)
    if h*w<=10000:
      cut_sets = [set(g) for g in cuts]
      carrier_matches = [[] for _ in cuts]
      verified_holes=Counter()
      for region in range(out):
        visited = np.zeros((h,w), dtype=bool)
        for y in range(h):
            for x in range(w):
                if ids[y,x] == region or visited[y,x]:
                    continue
                queue = deque([(y,x)])
                visited[y,x] = True
                pixels = set()
                frame = False
                while queue:
                    yy,xx = queue.popleft()
                    pixels.add((yy,xx))
                    frame |= yy in (0,h-1) or xx in (0,w-1)
                    for dy in (-1,0,1):
                        for dx in (-1,0,1):
                            ny,nx = yy+dy,xx+dx
                            if 0 <= ny < h and 0 <= nx < w and not visited[ny,nx] and ids[ny,nx] != region:
                                visited[ny,nx] = True
                                queue.append((ny,nx))
                if frame:
                    continue
                verified_holes[region] += 1
                boundary = set()
                for py,px in pixels:
                    for i in pixel_sides[(py,px)]:
                        if region in edges[i][2]:boundary.add(i)
                # C can include frontiers between regions *inside* the hole;
                # only the R--H boundary must lie in that component.
                matches = [k for k,g in enumerate(cut_sets) if boundary and boundary <= g]
                if len(matches) != 1:
                    raise ValueError(f"Cut-hole correspondence failed at region {region}: {len(matches)} matches")
                carrier_matches[matches[0]].append(region)
      if any(m != [carriers[k]] for k,m in enumerate(carrier_matches)) or holes != verified_holes:
          raise ValueError("Fast carrier disagrees with explicit 8-hole verification")
    n0 = len(junctions) + sum(g["type"] == "enc0" for g in geom)
    n1 = len(geom) + len(cuts)
    n2 = out + 1
    ii01,jj01,vv01,ii12,jj12,vv12 = [],[],[],[],[],[]
    virtual = len(junctions)
    for k,g in enumerate(geom):
        endpoints = g["endpoints"]
        if g["type"] == "enc0":
            ii01.append(virtual); jj01.append(k); vv01.append(2)
            virtual += 1
        else:
            for j,count in Counter(jindex[v] for v in endpoints).items():
                ii01.append(j); jj01.append(k); vv01.append(count)
        for region in g["pair"]:
            ii12.append(k); jj12.append(region); vv12.append(1)
    for k,region in enumerate(carriers, start=len(geom)):
        ii12.append(k); jj12.append(region); vv12.append(2)
        # Intentionally NO 0--1 incidence for cut k.
    m01 = sparse.csr_matrix((vv01,(ii01,jj01)),shape=(n0,n1),dtype=np.int32)
    m12 = sparse.csr_matrix((vv12,(ii12,jj12)),shape=(n1,n2),dtype=np.int32)
    return ids, edges, geom, cuts, carriers, holes, m01, m12

