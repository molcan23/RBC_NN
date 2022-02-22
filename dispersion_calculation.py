import numpy as np
import pandas as pd
import os
import re


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
    return distances.mean(), distances.std(), distances.var()


def dispersion_for_vtk_dataset(number_of_cells=18, path='.', sim_name=''):

    write_cycle = True
    max_cycle = 0

    only_cell_files = [f for f in os.listdir(f'{path}/vtk') if re.match("rbc[0-9]+_.+.vtk", f)]
    cycles = []
    vtk_cell_info = {i: [] for i in range(number_of_cells)}

    for i in only_cell_files:
        # print(i)
        _mean, _std, _var = dispersion_calculation(f'{path}/vtk/{i}')
        _cycle = int(i.split('_')[1].split('.')[0])

        if _cycle % 100000 == 0:
            print(_cycle)

        if write_cycle:
            if max_cycle > _cycle:
                write_cycle = False
            else:
                cycles.append(_cycle)

        rbc = int(i.split('_')[0].split('.')[0].split('c')[1])

        vtk_cell_info[rbc].append(_mean)

    if not os.path.exists(sim_name):
        os.makedirs(sim_name)

    for i in range(number_of_cells):
        df = pd.DataFrame(data={'cycle': sorted(list(cycles)),
                                'info': [x for _, x in sorted(zip(cycles, vtk_cell_info[i]))]})
        df.to_csv(f'{sim_name}/rbc{i}.csv', index=False)


dispersion_for_vtk_dataset(number_of_cells=20, sim_name='simmix20')
