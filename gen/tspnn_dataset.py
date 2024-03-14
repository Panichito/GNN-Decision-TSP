import sys, os, argparse, time, datetime
import numpy as np
import random
import networkx as nx
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from scipy.spatial import distance_matrix   
from tqdm import tqdm

def create_data_model(matrix):
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"]=matrix
    data["num_vehicles"] = 1
    data["depot"] = 0
    return data

def print_solution(manager, routing, solution, isOutput):
    """Prints solution on console."""
    path_list = []
    if isOutput:
        print(f"Objective: {solution.ObjectiveValue()} miles")
    index = routing.Start(0)
    plan_output = "Route for vehicle 0:\n"
    route_distance = 0
    while not routing.IsEnd(index):
        path_list.append(index)
        plan_output += f" {manager.IndexToNode(index)} ->"
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    path_list.append(index%len(path_list))
    plan_output += f" {manager.IndexToNode(index)}\n"
    if isOutput:
        print(plan_output)
    plan_output += f"Route distance: {route_distance}miles\n"
    return path_list

def solve(distance_matrix, time_limit=5, isOutput=False):
    num_nodes = len(distance_matrix)
    num_vehicles = 1  # TSP has only one vehicle
    depot = 0  # Starting node
    data = create_data_model(distance_matrix)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data["distance_matrix"]), data["num_vehicles"], data["depot"])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Define cost of each arc
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = time_limit
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        return print_solution(manager, routing, solution, isOutput=isOutput)
    else:
        raise Exception('Unsolvable')

def create_graph(num_points, seed):
    np.random.seed(seed)
    coords = np.random.rand(num_points, 2)
    distance_matrix_np = distance_matrix(coords, coords)
    distance_matrix_denormalize = np.round(distance_matrix_np*1e16).astype(int)

    adjacency_matrix = np.ones((num_points, num_points))
    np.fill_diagonal(adjacency_matrix, 0)

    # Solve
    route = solve(distance_matrix_denormalize)
    if route is None:
        raise Exception('Unsolvable')

    return np.triu(adjacency_matrix), distance_matrix_np, route, coords

def create_dataset(path, nmin, nmax, samples=1000, verbose=False):

    if not os.path.exists(path):
        os.makedirs(path)

    for i in tqdm(range(samples)):

        n = random.randint(nmin, nmax)
        n = 20  # IF you want to fixed node value
        if verbose:
            print('Creating instance {} with {} vertices'.format(i,n), flush=True)

        # Create graph
        Ma, Mw, route, coords = create_graph(n, seed=i)
        # Write graph to file
        write_graph(Ma, Mw, filepath="{}/{}.graph".format(path, i), route=route)

def write_graph(Ma, Mw, filepath, route=None, int_weights=False, bins=10**6):
    with open(filepath,"w") as out:

        n, m = Ma.shape[0], len(np.nonzero(Ma)[0])
        
        out.write('TYPE : TSP\n')

        out.write('DIMENSION: {n}\n'.format(n = n))

        out.write('EDGE_DATA_FORMAT: EDGE_LIST\n')
        out.write('EDGE_WEIGHT_TYPE: EXPLICIT\n')
        out.write('EDGE_WEIGHT_FORMAT: FULL_MATRIX \n')
        
        # List edges in the (generally not complete) graph
        out.write('EDGE_DATA_SECTION:\n')
        for (i,j) in zip(list(np.nonzero(Ma))[0], list(np.nonzero(Ma))[1]):
            out.write("{} {}\n".format(i,j))
        out.write('-1\n')

        # Write edge weights as a complete matrix
        out.write('EDGE_WEIGHT_SECTION:\n')
        for i in range(n):
            for j in range(n):
                if Ma[i,j] == 1:
                    out.write(str( int(bins*Mw[i,j]) if int_weights else Mw[i,j]))
                else:
                    out.write(str(n*bins+1 if int_weights else 0))
                out.write(' ')
            out.write('\n')

        if route is not None:
            # Write route
            out.write('TOUR_SECTION:\n')
            out.write('{}\n'.format(' '.join([str(x) for x in route])))

        out.write('EOF\n')

if __name__ == '__main__':

    # Define argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-samples', default=2**10, type=int, help='How many samples?')
    parser.add_argument('-path', help='Save path', required=True)
    parser.add_argument('-nmin', default=20, type=int, help='Min. number of vertices')
    parser.add_argument('-nmax', default=40, type=int, help='Max. number of vertices')

    # Parse arguments from command line
    args = parser.parse_args()

    print('Creating {} instances'.format(vars(args)['samples']), flush=True)
    create_dataset(
        vars(args)['path'],
        vars(args)['nmin'], 
        vars(args)['nmax'],
        samples=vars(args)['samples'],
    )