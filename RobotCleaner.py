# %%
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from queue import PriorityQueue
import numpy as np
import requests
import chardet
from math import sqrt
from flask import Flask, Response
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# %%



# Clase del Agente
class Robot(Agent):
    def __init__(self, unique_id, model, start_pos, papelera_pos):
        super().__init__(unique_id, model)
        self.position = start_pos
        self.capacity = 5
        self.filled = 0
        self.papelera_pos = papelera_pos
        self.flag = True
        self.last_visited_positions = []

    # La funcion detectX, se encarge de que el agente pueda localizar los obstaculos a su alrededor
    def detectX(self):
        neighborhood = self.model.grid.get_neighborhood(
            self.position, moore=True, include_center=False)
        for neighbor in neighborhood:
            row, col = neighbor
            if self.model.grid_data[row][col] == "X":
                self.model.robot_knowledge[row][col] = self.model.grid_data[row][col]
                self.model.obstacles.add((row, col))
        self.model.all_positions = [(row, col) for row, col in self.model.all_positions if (
            row, col) not in self.model.obstacles]
        

     # La funcion detectX, se encarge de que el agente pueda comuiar si hay basura en la posicion en la que se encuentra
    def detectTrash(self):
        row, col = self.position
        value = self.model.grid_data[row][col]
        if value == "S" or value == "P":
            pass
        else:
            value = int(value)
            self.model.robot_knowledge[row][col] = value
            if value > 0:
                #En caso de que la basura sea mayor a 0, agrega esa posicion a un arrgelo con las posiciones de la basura  
                self.model.trashes.add((row, col))



    # La funcion collect_trash, se encarga de que los agentes limpien el mapa
    def collect_trash(self):
        row, col = self.position
        value = self.model.grid_data[row][col]

        # Se verifica que el agente aun tenga capacidad
        if self.filled < self.capacity:
            trash = int(value)
            remaining_capacity = self.capacity - self.filled
             # En caso de que la basura sea mayor que la capcidad actual se recolecta la maxima cantidad y se actualiza el mapa
            if trash >= remaining_capacity:
                self.filled = self.capacity
                self.model.robot_knowledge[row][col] = str(
                    trash - remaining_capacity)
                self.model.grid_data[row][col] = str(
                    trash - remaining_capacity)
                        
            # En caso que se pueda recoger la cantidad completa de basura, se elimna esa posicion del arreglo y se actualiza el mapa
            else:
                self.filled += trash
                self.model.robot_knowledge[row][col] = "0"
                self.model.grid_data[row][col] = "0"
                self.model.trashes.remove(self.position)

    # La funcion calculate_distance, calcula la distancia entre dos posiciones, la funcion regresa un valor que representa la distancia de 
    # Manhattan entre dos puntos, para despues ser comparado y obtener el valor minimo.
    def calculate_distance(self, pos1, pos2):
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) + abs(y1 - y2)
    

    #La funcion stepsCloser, recibe la posicion actual y la posicon que se desea ir, y calcula a donde se debe mover el agente. La funcion
    #regresa la posicion en la que se debeb mover
    def stepsCloser(self, pos1, pos2):
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]

        new_x = pos1[0] + (1 if dx > 0 else -1 if dx < 0 else 0)
        new_y = pos1[1] + (1 if dy > 0 else -1 if dy < 0 else 0)

        return (new_x, new_y)
    

    # La funcion moveExplore se encarga de el movimiento de los agentes para poder mapear.
    def moveExplore(self):
        # Se mandan a llamar las fuciones inicales
        self.detectX()
        self.detectTrash()

        neighborhood = self.model.grid.get_neighborhood(
            self.position, moore=True, include_center=False)

        # Crear listas para almacenar la atracción y las posiciones disponibles
        attractiveness = []
        available_positions = set()

        for neighbor in neighborhood:
            row, col = neighbor
            neighbor = (row, col)  
            if (self.model.robot_knowledge[row][col] == ".") and neighbor not in self.model.occupied_positions:
                if self.model.grid_data[row][col] != "X" and self.model.robot_knowledge[row][col] not in self.model.visited_positions:
                    attraction = 1  # Atracción base
                    available_positions.add(neighbor)

            else:
                attractiveness.append(0)

        available_positions = [
            pos for pos in available_positions if pos not in self.last_visited_positions]

        if any(available_positions):
            # Usamos el numero de vecinos disponibles como peso para cada posición
            weights = [
                1 if pos in available_positions else 0 for pos in neighborhood]
            new_position = self.random.choices(
                neighborhood, weights=weights)[0]
        else:
            # Si no hay posiciones disponibles, buscamos cual es la posicion no vistada mas cercana.
            if self.model.all_positions:
                closest_unvisited = min(
                    self.model.all_positions, key=lambda pos: self.calculate_distance(self.position, pos))
                next_pos = self.stepsCloser(self.position, closest_unvisited)

                # print("Pos, closesr, nextmove: ",self.position, closest_unvisited, next_pos)
                if self.model.grid_data[next_pos[0]][next_pos[1]] == "X" or next_pos in self.model.occupied_positions:
                    neighbor_positions = self.model.grid.get_neighborhood(
                        self.position, moore=True, include_center=False)

                    valid_neighbors = [
                        pos for pos in neighbor_positions
                        if (0 <= pos[0] < self.model.grid.width and 0 <= pos[1] < self.model.grid.height)
                        and self.model.grid_data[pos[0]][pos[1]] != "X"
                        and pos not in self.model.occupied_positions
                        and pos != self.position
                        and pos not in self.last_visited_positions
                    ]
                    # En caso de que no haya sido valida la poscion, mandamos a llamar la funcion de nuevo eliminando las posiciones no validas
                    if valid_neighbors:
                        new_position = min(valid_neighbors, key=lambda pos: self.calculate_distance(
                            pos, closest_unvisited))

                    else:
                        new_position = self.position
                else:
                    new_position = next_pos
            else:
                new_position = self.position


        # Agregamos la posicion en el arreglo de visited_positions
        new_position = tuple(new_position)
        self.model.visited_positions.add(new_position)


        # Eliminamos la posicion del arreglo con las posciones totales
        if new_position in self.model.all_positions:
            self.model.all_positions.remove(new_position)


        # Actualizamos el arreglo de lugares ocupados
        if self.position in self.model.occupied_positions:
            self.model.occupied_positions.remove(self.position)
        self.model.occupied_positions.add(new_position)

        # Movemos al agente y actualizamos su posicion
        self.model.grid.move_agent(self, new_position)
        self.position = new_position

        #Actualizamos el arreglo con los ultimos moviemntos, este arreglo evita el problema de moverser y regresar a la misma casilla
        if len(self.last_visited_positions) >= 5:
            self.last_visited_positions.pop(0)
        self.last_visited_positions.append(new_position)



    # Funcion de recojer basura
    def moveToTrash(self):


        # En la primera iteracion hacemos la ultima actualizacion del conocimeinto del robot
        if self.flag:
            if self.position not in self.model.visited_positions:
                self.model.visited_positions.add(self.position)
            row, col = self.position
            value = self.model.grid_data[row][col]
            self.model.robot_knowledge[row][col] = value
            if value != "0" and value != "P" and value != "R" and value != "X":
                self.model.trashes.add(self.position)
                self.flag = False
            self.flag = False


        # Verificamos si el agente se encuentre en una poscion con basura
        if self.position in self.model.trashes:
            self.collect_trash()

        # Vaciamos la capacidad del agente cuando esta en la papelera
        if self.position == self.papelera_pos:
            self.filled = 0


        if self.filled == self.capacity or not self.model.trashes:
            # Si ya no hay basura por recoger, decismos que los agantes se moverian a la esquina superior derecha para no estorbar cerca de la papelera
            if self.filled == 0:
                # Definimos la esquina superior como el punto de encuentro en caso de que los agantes ya no tengan tareas
                corner_position = (0, 0)
                next_pos = self.stepsCloser(self.position,  corner_position)

                # El agente busca el siguiente paso haciaa la esquina, tomando en cuenta los movimientos no validos
                if self.model.grid_data[next_pos[0]][next_pos[1]] == "X" or next_pos in self.model.occupied_positions:
                    neighbor_positions = self.model.grid.get_neighborhood(
                        self.position, moore=True, include_center=False)

                    valid_neighbors = [
                        pos for pos in neighbor_positions
                        if (0 <= pos[0] < self.model.grid.width and 0 <= pos[1] < self.model.grid.height)
                        and self.model.grid_data[pos[0]][pos[1]] != "X"
                        and pos not in self.model.occupied_positions
                        and pos != self.position
                    ]

                    if valid_neighbors:
                        new_position = min(
                            valid_neighbors, key=lambda pos: self.calculate_distance(pos, corner_position))
                    else:
                        new_position = self.position
                else:
                    new_position = next_pos

            # En caso que los robots estan llenos y deben dirigirse a la papelera
            else:
                 # Calculamos el sigueinte movimiento tomando en cuenta las posicion no validas
                next_pos = self.stepsCloser(self.position,  self.papelera_pos)
                if self.model.grid_data[next_pos[0]][next_pos[1]] == "X" or next_pos in self.model.occupied_positions:
                    neighbor_positions = self.model.grid.get_neighborhood(
                        self.position, moore=True, include_center=False)

                    valid_neighbors = [
                        pos for pos in neighbor_positions
                        if (0 <= pos[0] < self.model.grid.width and 0 <= pos[1] < self.model.grid.height)
                        and self.model.grid_data[pos[0]][pos[1]] != "X"
                        and pos not in self.model.occupied_positions
                        and pos != self.position
                        and pos not in self.last_visited_positions
                    ]

                    if valid_neighbors:
                        new_position = min(valid_neighbors, key=lambda pos: self.calculate_distance(
                            pos, self.papelera_pos))
                    else:
                        new_position = self.position
                else:
                    new_position = next_pos


        # Caso en el que aun hay basuras y el agente aun tiene capacidad
        else:
            # Calculamos cual es la basura mas cercana
            if self.model.trashes:
                closest_unvisited = min(
                    self.model.trashes, key=lambda pos: self.calculate_distance(self.position, pos))
                next_pos = self.stepsCloser(self.position, closest_unvisited)


                # Verificamos que la poscion dada sea valida
                if self.model.grid_data[next_pos[0]][next_pos[1]] == "X" or next_pos in self.model.occupied_positions:
                    neighbor_positions = self.model.grid.get_neighborhood(
                        self.position, moore=True, include_center=False)

                    valid_neighbors = [
                        pos for pos in neighbor_positions
                        if (0 <= pos[0] < self.model.grid.width and 0 <= pos[1] < self.model.grid.height)
                        and self.model.grid_data[pos[0]][pos[1]] != "X"
                        and pos not in self.model.occupied_positions
                        and pos != self.position
                        and pos not in self.last_visited_positions
                    ]

                    # En caso de que la poscion no sea valida, la recalculamos eliminando los lugares no validos
                    if valid_neighbors:
                        new_position = min(valid_neighbors, key=lambda pos: self.calculate_distance(
                            pos, closest_unvisited))

                    else:
                        new_position = self.position
                else:
                    new_position = next_pos
            else:
                new_position = self.position

        new_position = tuple(new_position)


        # Actualizamos el arreglo de occupied_positions
        if self.position in self.model.occupied_positions:
            self.model.occupied_positions.remove(self.position)
        self.model.occupied_positions.add(new_position)

        # Movemos al agente y actualizamos su posicion
        self.model.grid.move_agent(self, new_position)
        self.position = new_position

        #Actualizamos el arreglo con los ultimos moviemntos, este arreglo evita el problema de moverser y regresar a la misma casilla
        if len(self.model.trashes) != 0:
            if len(self.last_visited_positions) >= 5:
                self.last_visited_positions.pop(0)
            self.last_visited_positions.append(new_position)


# %%

# Modelo
class OfficeCleaningModel(Model):
    def __init__(self, grid_data, height, width):

        self.grid_data = grid_data
        self.height = height
        self.width = width
        self.robot_knowledge = [["R" if value == "S" else "P" if value ==
                                 "P" else "." for value in row] for row in self.grid_data]
        self.all_positions = [(row, col) for row in range(
            self.height) for col in range(self.width)]
        self.grid = MultiGrid(self.height, self.width, torus=False)
        self.schedule = RandomActivation(self)
        self.agent_count = 0

        self.occupied_positions = set()
        self.visited_positions = set()
        self.obstacles = set()
        self.trashes = set()
        self.unallowed_positions = set()

        self.create_agents()
        self.papelera_pos = self.find_papelera()

        self.exploring = True
        self.mision_complete = False

        self.datacollector = DataCollector(
            agent_reporters={}
        )

    # La funcion de find_papelera se encarga de obtener la posicion de la papelera.
    def find_papelera(self):
        for row in range(self.height):
            for col in range(self.width):
                if self.grid_data[row][col] == "P":
                    papelera_pos = (row, col)
                    if papelera_pos in self.all_positions:
                        self.all_positions.remove(papelera_pos)
                    return papelera_pos
                

    # La funcion create_agentes, crea a los agentes en base al txt que nos asigno
    def create_agents(self):
        for row in range(self.height):
            for col in range(self.width):
                if self.grid_data[row][col] == "S":
                    self.all_positions.remove((row, col))
                    self.visited_positions.add((row, col))
                    for i in range(5):
                        robot = Robot(i, self, (row, col),
                                      self.find_papelera())
                        self.schedule.add(robot)
                        self.grid.place_agent(robot, (row, col))

    # La funcion print_grid, se encarga de impirmir las diferentes matrices, en este caso es la matriz que se envia a Unity.
    def print_grid(self):
        with open("input.txt", "w") as file:
            for row_idx, row in enumerate(self.grid_data):
                for col_idx, cell_value in enumerate(row):
                    position = (row_idx, col_idx)
                    cell_contents = self.grid.get_cell_list_contents(position)
                    if cell_contents:
                        agent_count = len(cell_contents)
                        if agent_count > 1:
                            file.write("A ")
                        else:
                            agent_ids = [str(agent.unique_id)
                                         for agent in cell_contents]
                            agent_ids_str = " ".join(agent_ids)
                            file.write(f"A{agent_ids_str} ")
                    else:
                        file.write(f"{cell_value} ")
                file.write("\n")

    # La funcion print_grid_withAgents se encarga de imprimir la matriz con los agentes, se uso en el desarrollo, para tener una nocion de los
    # agentes y sus posiciones.
    def print_grid_withAgents(self):
        self.exploringmap = []
        for row in range(self.grid.width):
            for col in range(self.grid.height):
                position = (row, col)
                cell_contents = self.grid.get_cell_list_contents(position)
                if cell_contents:
                    agent_count = len(cell_contents)
                    print(f"{agent_count}", end=" ")
                else:
                    if position in self.visited_positions:
                        print("#", end=" ")
                    elif position in self.obstacles:
                        print("X", end=" ")
                    elif position == self.find_papelera():
                        print("P", end=" ")
                    else:
                        print(".", end=" ")
            print()

    # La funcion step, se encarga de mandar a llamar todas las fucniones, actualizacion de agentes, registro de datos, etc...
    def step(self):
        if len(self.all_positions) != 0:
            for agent in self.schedule.agents:
                agent.moveExplore()
        else:
            self.exploring = False

            if len(model.trashes) != 0:
                for agent in self.schedule.agents:
                    agent.moveToTrash()
            else:
                if all(agent.filled == 0 for agent in self.schedule.agents):
                    self.mision_complete = True
                else:
                    for agent in self.schedule.agents:
                        agent.moveToTrash()

        self.print_grid()
        self.datacollector.collect(self)
        self.schedule.step()

# %%
# Abrimos el archivo


def detect_encoding(file_path):
    with open(file_path, "rb") as file:
        result = chardet.detect(file.read())
        return result['encoding']


# Variable con el nombre de nuestro archivo a analizar
mapa = "input1.txt"
file_encoding = detect_encoding(mapa)

with open(mapa, "r", encoding=file_encoding) as file:
    first_line = file.readline().strip()

# Extraemos la primera linea con el width y height
height, width = map(int, first_line.split())

# Leemos el resto de archivo
with open(mapa, "r", encoding=file_encoding) as file:
    lines = file.readlines()[1:]  # Ignore the first line
    grid_data = [line.strip().split() for line in lines]

# %%
app = Flask(__name__)
model = OfficeCleaningModel(grid_data, height, width)


def read_matrix_from_file(filename):
    matrix_data = []
    with open(filename, 'r') as file:
        for line in file:
            row = line.strip().split()
            matrix_data.append(row)
    return matrix_data


@app.route('/get_matrix')
def publish_matrix():
    matrix = read_matrix_from_file('input.txt')
    matrix_str = "\n".join([" ".join(row) for row in matrix])

    return Response(matrix_str, content_type='text/plain')


@app.route('/step')
def run_step():
    if not model.mision_complete:
        model.step()
        return "", 200
    else:
        return "", 400


if __name__ == '__main__':
    app.run()


# %%
