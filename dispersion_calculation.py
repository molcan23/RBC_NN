import numpy as np


def euklidian_distance(v1, v2):
    return np.sqrt((v1['x'] - v2['x'])**2 + (v1['y'] - v2['y'])**2 + (v1['z'] - v2['z'])**2 )


def dispersion_calculation(filae_name):
    vertices = {}
    edges = {i: [] for i in range(374)}
    lines = []

    with open(filae_name, 'r') as f:
        for idx, line in enumerate(f):
            lines.append(line.split())

        for i in range(5, 379):
            vertices[i-5] = {'x': float(lines[i][0]),
                             'y': float(lines[i][1]),
                             'z': float(lines[i][2])}

        for e in range(380, len(lines)):
            a = int(lines[e][1])
            b = int(lines[e][2])
            c = int(lines[e][3])
            edges[a].append(b)
            edges[a].append(c)
            edges[b].append(a)
            edges[b].append(c)
            edges[c].append(a)
            edges[c].append(b)

    distances = []

    for v1, e in enumerate(edges):
        for v2 in edges[e]:
            distances.append(euklidian_distance(vertices[v1], vertices[v2]))

    distances = np.array(distances)
    print(distances.mean())
    print(distances.std())
    print(distances.var())


dispersion_calculation('rbc0_98000 (1).vtk')