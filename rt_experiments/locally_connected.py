from collections import defaultdict, OrderedDict

import common

import logging

import numpy as np

from rig.geometry import (minimise_xyz, to_xyz, shortest_torus_path_length,
                          standard_system_dimensions)
from rig.netlist import Net
from rig.place_and_route import Cores, Machine, route
from rig.routing_table import routing_tree_to_tables

from six import iterkeys, iteritems

logger = logging.getLogger(__name__)


def get_targets(x, y, probs, width, height, rng, vertices):
    # Get the target co-ordinates
    xs, ys, ps = np.where(rng.uniform(0, 1, size=probs.shape) < probs)
    xs = (xs + x) % machine.width
    ys = (ys + y) % machine.height

    # Get the sinks
    return list(vertices[xyp] for xyp in zip(xs, ys, ps))


def get_network(machine, rng):
    # Compute the distance between chips
    dists = np.zeros((machine.width, machine.height, 17), dtype=np.uint32)

    for x, y in machine:
        dists[x, y, :] = shortest_torus_path_length(
            (0, 0, 0), minimise_xyz(to_xyz([x, y])),
            machine.width, machine.height
        )

    # Compute the probability of a target for each distance
    probs = 0.475**(dists + 1)
    centroid_probs = 0.25**(dists + 1)

    # Create the vertices for each core (and allocations and placements)
    vertices = OrderedDict()
    placements = OrderedDict()
    allocations = OrderedDict()
    resources = defaultdict(lambda: {Cores: 1})

    for x, y in machine:
        for p in range(17):
            # Create the new vertex for this core
            vx = common.Vertex(x, y, p)

            # Place and allocate
            vertices[x, y, p] = vx
            placements[vx] = (x, y)
            allocations[vx] = {Cores: slice(p, p+1)}

    # Keep track of targets which are the result of centroid(s)
    centroid_sinks = defaultdict(lambda: set())

    # Now construct the nets
    nets = list()
    for x, y, p in ((i, j, k) for k in range(17) for i, j in machine):
        # Get the source
        source = vertices[x, y, p]

        # Get the (local) sinks
        sinks = get_targets(x, y, probs, machine.width, machine.height,
                            rng, vertices)

        # Add centroid links if we have any (we have a 5% chance of connecting
        # to 1 centroid)
        if rng.uniform(0, 1) < 0.05:
            # Get the target co-ordinates
            c_sinks = get_targets(x, y, centroid_probs, machine.width,
                                  machine.height, rng, vertices)

            # Mark these as sinks relating to a centroid
            centroid_sinks[x, y, p].update(c_sinks)

            # Add to the general list of sinks
            sinks += c_sinks

        # Add the net
        nets.append(Net(source, sinks))

    # Route the nets
    logger.info("Routing...")

    # Get the centroid routes
    centroid_routes = route(resources, nets, machine, list(),
                            placements, allocations)

    # Prune back to get the local-only connections
    logger.info("Pruning...")
    local_routes = {
        net: common.get_pruned_routing_tree(tree, centroid_sinks[net.source])
        for net, tree in iteritems(centroid_routes)
    }

    return centroid_routes, local_routes


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Get the height and width of the system
    machine = Machine(*standard_system_dimensions(24))

    # Build the network
    logger.info("Building netlist...")
    rng = np.random.RandomState(seed=2804)
    centroid_routes, local_routes = get_network(machine, rng)

    # Find the necessary number of random bits
    rnd_bits = int(np.ceil(np.log2(machine.width * machine.height * 17)))

    # Construct the keys and routing tables
    algs = (common.assign_xyp_keys,
            common.assign_xyzp_keys,
            common.assign_hilbert_keys,
            common.assign_random_keys,
            common.assign_random_keys)
    all_args = (tuple(), tuple(), (machine, ),
                (machine, 2804, 21), (machine, 2804, rnd_bits))
    fps = ("xyp", "xyzp", "hilbert", "rnd21", "rnd{:d}".format(rnd_bits))

    for routes, prefix in ((centroid_routes, "centroid"),
                           (local_routes, "locally_connected")):
        for alg, args, fp in zip(algs, all_args, fps):
            logger.info("Generating keys...")
            keys = alg(iterkeys(routes), *args)

            # Generate the routing tables
            logger.info("Building routing tables...")
            tables = routing_tree_to_tables(routes, keys)

            # Dump to file
            logger.info("Writing to file...")
            with open("benchmarks/{}_{}.bin".format(prefix, fp), 'wb+') as f:
                common.dump_routing_tables(f, tables)
