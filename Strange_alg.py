import numpy as np


def move_forward():
    print("1")
    response = int(input())
    assert response == 1
def rotate_clockwise():
    print("2 1")
    response = int(input())
    assert response == 1

def rotate_counter_clockwise():
    print("2 0")
    response = int(input())
    assert response == 1

def light_fire():
    print("3")
    response = []
    for _ in range(K * 2 + 1):
        response.append(input())

    return response


def double_matrix(matrix):
    old_size = len(matrix)
    new_size = old_size * 2
    new_matrix = np.zeros((new_size, new_size), dtype=matrix.dtype)
    new_matrix[old_size // 2:old_size // 2 + old_size, old_size // 2:old_size // 2 + old_size] = matrix
    return new_matrix


def check_boundary_ones(matrix):
    size = matrix.shape[0]
    if not np.all(matrix[0] == 1) or not np.all(matrix[size - 1] == 1):
        return False
    if not np.all(matrix[:, 0] == 1) or not np.all(matrix[:, size - 1] == 1):
        return False

    return True

def check_zeros(matrix):
    return np.any(matrix == 0)

#собирает сет с соседями(то есть для каждой точки список тех точек, в которые можем перейти по стороне клетки)
def find_neighbors(point, point_set):
    x, y = point
    neighbors = set()

    for neighbor_point in point_set:
        if (abs(neighbor_point[0] - x) == 1 and neighbor_point[1] == y) or (neighbor_point[0] == x and abs(neighbor_point[1] - y) == 1):
            neighbors.add(neighbor_point)

    return neighbors

#нахождения цикла в графе
def find_cycle(graph):
    visited = set()

    def dfs(node, path):
        visited.add(node)
        path.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, path):
                    return True
            elif neighbor in path:
                path.append(neighbor)
                return True

        path.pop()
        return False

    for node in graph:
        if node not in visited:
            path = []
            if dfs(node, path):
                return path

    return None


def point_inside_polygon(point, polygon):
    x = point[0]
    y = point[1]
    n = len(polygon)
    odd_nodes = False
    j = n - 1

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if yi < y and yj >= y or yj < y and yi >= y:
            if xi + (y - yi) / (yj - yi) * (xj - xi) < x:
                odd_nodes = not odd_nodes

        j = i

    return odd_nodes


#проверяем есть ли еще места где можно разжечь костер(то есть проверяем есть ли у нас цикл из стен вокруг начальной точки минотавра)
def check_circle(point_set, center):
    our_copy = point_set.copy()
    our_copy.remove(center)
    #список точек без центральной(проверим образуют ли они цикл)
    points_with_neighbors = {point: find_neighbors(point, point_set) for point in point_set}
    result = find_cycle(points_with_neighbors)
    if result is None:
        return False
    #теперь проверим лежит ли центральная точка внутри этого цикла
    if (point_inside_polygon(center, point_set) == False):
        return False
    return True


def check_stop(set_v, set_i, set_w, set_g):
    if (set_v != set_g):
        return False
    if (check_circle(set_w) == False):
        return False
    return True



#вариант когда нам выгоднее разжигать костер
def first_variant(x, y, x_1, y_1, A, B, C, K):
    # давайте в visited 0 - это непосещенные, 1 - посещенные
    # давайте в labyrinth_map 0 - это дырка, 1 - стена
    visited = np.full((2 * (2 * K + 1), 2 * (2 * K + 1)), 0)
    labyrinth_map = np.full((2 * (2 * K + 1), 2 * (2 * K + 1)), 0)
    vis_set = set()
    vis_set.append((K, K))
    #сет того про что мы знаем информацию
    inf_set = set()
    #сет с проходами
    go_set = set()
    #сет со стенами
    wall_set = set()

    # требуется ли расширение нашей матрицы
    flag_expand = 0
    our_timing = 0 # текущее время
    our_x = K
    our_y = K

    #давайте будем считать что верх - это поврот 0, право - 1, низ - 2/-2, лево - -1
    delta_x = x_1 - x
    delta_y = y_1 - y
    if (delta_x < 0):
        rotation = -1
    if (delta_x > 0):
        rotation = 1
    if (delta_y > 0):
        rotation = 0
    if (delta_y < 0):
        rotation = 2

    up = (0, 1)
    down = (0, -1)
    right = (1, 0)
    left = (-1, 0)

    while (check_stop(visited, labyrinth_map) == False):

        if (rotation == 0):
            x_next = our_x
            y_next = our_y + 1

        if (rotation == 1):
            x_next = our_x + 1
            y_next = our_y

        if (rotation == -1):
            x_next = our_x - 1
            y_next = our_y

        if (rotation == 2):
            x_next = our_x
            y_next = our_y - 1

        if (((x_next, y_next) in go_set) and ((x_next, y_next) in vis_set) == False):
            our_x = x_next
            our_y = y_next
            vis_set.add((x_next, y_next))

        else:
            if (rotation == 0):
                x_next = our_x + 1
                y_next = our_y

            if (rotation == 1):
                x_next = our_x
                y_next = our_y - 1

            if (rotation == -1):
                x_next = our_x
                y_next = our_y + 1

            if (rotation == 2):
                x_next = our_x - 1
                y_next = our_y

            if (((x_next, y_next) in go_set) and ((x_next, y_next) in vis_set) == False):
                our_x = x_next
                our_y = y_next
                vis_set.add((x_next, y_next))

            else:
                if (rotation == 0):
                    x_next = our_x
                    y_next = our_y - 1

                if (rotation == 1):
                    x_next = our_x - 1
                    y_next = our_y

                if (rotation == -1):
                    x_next = our_x + 1
                    y_next = our_y

                if (rotation == 2):
                    x_next = our_x
                    y_next = our_y + 1

                if (((x_next, y_next) in go_set) and ((x_next, y_next) in vis_set) == False):
                    our_x = x_next
                    our_y = y_next
                    vis_set.add((x_next, y_next))

                else:
                    if (rotation == 0):
                        x_next = our_x - 1
                        y_next = our_y

                    if (rotation == 1):
                        x_next = our_x
                        y_next = our_y + 1

                    if (rotation == -1):
                        x_next = our_x
                        y_next = our_y - 1

                    if (rotation == 2):
                        x_next = our_x + 1
                        y_next = our_y

                    if (((x_next, y_next) in go_set) and ((x_next, y_next) in vis_set) == False):
                        our_x = x_next
                        our_y = y_next
                        vis_set.add((x_next, y_next))

                    else:
                        response = light_fire()
                        inf_set, wall_set, go_set = remember_map_fire(wall_set, inf_set, go_set, response, our_x, our_y)






def remember_map_fire(w_set, i_set, g_set, response, x, y):
    copy_w = w_set.copy()
    copy_i = i_set.copy()
    copy_g = g_set.copy()
    for i, line in enumerate(response):
        for j, char in enumerate(line):
            copy_i.add((x - K + i, y - K + j))
            if char == '#':
                copy_w.add((x - K + i, y - K + j))
            if char == '_':
                copy_g.add((x - K + i, y - K + j))
    return copy_i, copy_w, copy_g


x, y, x_1, y_1, A, B, C, K = map(int, input().split(', '))

def solve(x, y, x_1, y_1, A, B, C, K):
    if (3 * A + B > C):
        #тогда нам выгоднее тыкаться в стены и проверять так(не разжигая костер)
        return first_variant(x, y, x_1, y_1, A, B, C, K)
    else:
        return second_variant(x, y, x_1, y_1, A, B, C, K)


