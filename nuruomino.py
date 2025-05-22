# nuruomino.py: Template para implementação do projeto de Inteligência Artificial 2024/2025.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes sugeridas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 110633 Filipe Oliveira
# 110720 Francisco Andrade

import sys
from search import Problem, Node, depth_first_tree_search, astar_search
from copy import deepcopy


# --- Definições Globais e Constantes ---
TETROMINO_BASE_SHAPES = {
    'L': frozenset([(0,0), (1,0), (2,0), (2,1)]),
    'I': frozenset([(0,0), (1,0), (2,0), (3,0)]),
    'T': frozenset([(0,0), (0,1), (0,2), (1,1)]),
    'S': frozenset([(0,1), (1,1), (1,0), (2,0)])
}
ALLOWED_TETROMINO_TYPES = ['L', 'I', 'T', 'S']
TETROMINO_VARIANTS = {}


# --- Funções de Manipulação de Tetraminós ---
def _normalize_shape(shape_coords):
    """Translada a peça para que a sua célula mais a cima e à esquerda esteja em (0,0)."""
    if not shape_coords: return frozenset()
    min_r = min(r for r, c in shape_coords)
    min_c = min(c for r, c in shape_coords)
    return frozenset([(r - min_r, c - min_c) for r, c in shape_coords])

def _rotate_shape_90_clockwise(shape_coords):
    """Roda uma peça 90 graus no sentido horário em torno da origem (0,0) da peça, depois normaliza."""
    # (r,c) -> (c, -r) para rotação horária.
    # Depois normalizar para manter coordenadas positivas e relativas ao canto superior esquerdo.
    if not shape_coords: return frozenset()
    # max_r_before_rotation = max(r for r,c in shape_coords) # Para rotação em torno de um ponto diferente
    # return _normalize_shape(frozenset([(c, max_r_before_rotation - r) for r, c in shape_coords]))
    return _normalize_shape(frozenset([(c, -r) for r, c in shape_coords]))


def _reflect_shape_vertical_axis(shape_coords):
    """Reflete uma peça em torno do eixo y (vertical) que passa pela "origem" da peça, depois normaliza."""
    # (r,c) -> (r, -c) e depois normaliza
    return _normalize_shape(frozenset([(r, -c) for r, c in shape_coords]))

def generate_all_tetromino_variants():
    """Preenche o dicionário global TETROMINO_VARIANTS com todas as orientações únicas."""
    global TETROMINO_VARIANTS
    if TETROMINO_VARIANTS: # Já gerado
        return

    for name, base_shape in TETROMINO_BASE_SHAPES.items():
        variants = set()
        current_shape_to_reflect = base_shape
        for _ in range(2): # Original e refletido (se a reflexão gerar uma nova forma)
            shape_to_rotate = current_shape_to_reflect
            for _ in range(4): # 4 rotações
                variants.add(shape_to_rotate)
                shape_to_rotate = _rotate_shape_90_clockwise(shape_to_rotate)
            # Prepara para a próxima iteração: reflete a forma base original
            current_shape_to_reflect = _reflect_shape_vertical_axis(base_shape)
            if current_shape_to_reflect == base_shape and name in ['I', 'S', 'T']: # Algumas peças não mudam com reflexão e rotação (ou S vira Z)
                 # O S e Z são reflexos. T e I são simétricos em relação a um eixo após algumas rotações.
                 # O objetivo é ter todas as 8 (ou menos) orientações distintas.
                 pass


        TETROMINO_VARIANTS[name] = [list(variant) for variant in variants]


class NuruominoState:
    state_id_counter = 0

    def __init__(self, board: 'Board'):
        self.board = board
        self.id = NuruominoState.state_id_counter
        NuruominoState.state_id_counter += 1

    def __lt__(self, other):
        """ Este método é utilizado em caso de empate na gestão da lista
        de abertos nas procuras informadas. """
        return self.id < other.id
    
    def __eq__(self, other):
        if not isinstance(other, NuruominoState):
            return NotImplemented
        return self.board == other.board
    
    def __hash__(self):
        return hash(self.board)
    


class Board:
    """Representação interna de um tabuleiro do Puzzle Nuruomino."""
    def __init__(self, N, initial_grid_ids, regions_map, assignments=None):
        self.N = N
        self.initial_grid_ids = initial_grid_ids # Grelha original com números de região (strings)
        self.regions_map = regions_map # {'id_str': {'cells': set((r,c))}, ...}
        
        # assignments: {'region_id_str': {'type': 'L', 'abs_cells': frozenset((r,c))}}
        self.assignments = assignments if assignments is not None else {}

        # solution_grid é a grelha para output: L,I,T,S ou números de região originais
        # Esta parte pode ser mais complexa e depender de outros métodos,
        # mas para o parse_instance, o importante é armazenar os dados de entrada.
        self.solution_grid = [row[:] for row in initial_grid_ids] # Inicializa com os IDs das regiões
        
        # Se houver assignments (não neste caso do parse_instance inicial), eles seriam aplicados aqui.
        for region_id, data in self.assignments.items():
            tetromino_type = data['type']
            for r_cell, c_cell in data['abs_cells']:
                if 0 <= r_cell < self.N and 0 <= c_cell < self.N:
                    self.solution_grid[r_cell][c_cell] = tetromino_type
        

    def get_value(self, row: int, col: int) -> str:
        """Retorna o valor (char) preenchido numa determinada posição da solution_grid."""
        if 0 <= row < self.N and 0 <= col < self.N:
            return self.solution_grid[row][col]
        raise IndexError("Posição fora da grelha")
    

    def get_region_cells(self, region_id: str) -> set:
        """Retorna o conjunto de células (r,c) para uma dada região."""
        return self.regions_map.get(region_id, {}).get('cells', set())


    def adjacent_regions(self, region_id:str) -> list:
        """Devolve uma lista das regiões que fazem fronteira com a região enviada no argumento."""
        adj_regs = set()
        region_cells = self.get_region_cells(region_id)
        if not region_cells:
            return []

        for r, c in region_cells:
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]: # Ortogonais
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.N and 0 <= nc < self.N:
                    neighbor_region_id = self.initial_grid_ids[nr][nc]
                    if neighbor_region_id != region_id: # Se pertence a outra região
                        adj_regs.add(neighbor_region_id)
        return sorted(list(adj_regs)) # Exemplo pede lista ordenada
    
    
    def adjacent_positions(self, row:int, col:int) -> list:
        """Devolve as posições adjacentes à região, em todas as direções, incluindo diagonais."""
        adj_pos = []
        deltas = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        for dr, dc in deltas:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.N and 0 <= nc < self.N:
                adj_pos.append((nr, nc))
        return adj_pos
    

    def adjacent_values(self, row:int, col:int) -> list:
        """Devolve os valores das celulas adjacentes à região, em todas as direções, incluindo diagonais."""
        values = []
        for r_adj, c_adj in self.adjacent_positions(row, col):
            values.append(self.get_value(r_adj, c_adj))
        return values
    

    @staticmethod
    def parse_instance():
        """Lê a instância do problema do standard input (stdin)
        e retorna uma instância da classe Board.
            
            Por exemplo:
            $ python3 pipe.py < test-01.txt
            > from sys import stdin
            > line = smtdin.readline().split()
        """
        input_lines_str = []
        for line in sys.stdin:
            stripped_line = line.strip()
            if stripped_line: # Ignora linhas vazias
                input_lines_str.append(stripped_line.split())
        
        if not input_lines_str:
            raise ValueError("Input vazio ou inválido: nenhuma linha de dados encontrada.")

        N = len(input_lines_str) # Número de linhas determina N
        
        # Verifica se todas as linhas têm N colunas
        for r_idx, row_list in enumerate(input_lines_str):
            if len(row_list) != N:
                raise ValueError(f"Input inválido: linha {r_idx+1} tem {len(row_list)} colunas, esperado {N}.")

        initial_grid_ids = input_lines_str # Esta já é a nossa grelha de IDs

        regions_map = {} # Dicionário para mapear ID da região para as suas células
                         # Formato: {'region_id_str': {'cells': set_of_coords (r,c)}}
        
        for r in range(N):
            for c in range(N):
                region_id = initial_grid_ids[r][c]
                if region_id not in regions_map:
                    regions_map[region_id] = {'cells': set()}
                regions_map[region_id]['cells'].add((r, c))
        
        # Neste ponto, N, initial_grid_ids, e regions_map estão prontos.
        # assignments será um dicionário vazio para uma instância recém-parseada.
        return Board(N, initial_grid_ids, regions_map, {})    

    
    def print_board(self):
        """Imprime a grelha (solution_grid) no formato de output especificado."""
        output = ""
        for r in range(self.N):
            output += "\t".join(self.solution_grid[r]) + "\n"
        return output.strip() # Remove a última nova linha
    

    def __eq__(self, other):
        if not isinstance(other, Board): return NotImplemented
        # Compara N, grelha inicial e assignments (que determinam a solution_grid)
        return (self.N == other.N and
                self.initial_grid_ids == other.initial_grid_ids and # Grelhas são listas de listas
                self.assignments == other.assignments)
    

    def __hash__(self):
        # Para que Board possa ser usado em sets (e.g., closed list em graph search)
        # É crucial que os assignments sejam representados de forma canónica (ordenada)
        frozen_assignments = []
        for region_id, data in sorted(self.assignments.items()): # Ordenar por region_id
            # As células (abs_cells) já são um frozenset, que é hasheável.
            # Tipo também é hasheável.
            frozen_data_tuple = (data['type'], data['abs_cells'])
            frozen_assignments.append((region_id, frozen_data_tuple))
            
        # initial_grid_ids é list of lists of strings. Precisa ser tupla de tuplas.
        grid_tuple = tuple(map(tuple, self.initial_grid_ids))
        
        return hash((self.N, grid_tuple, tuple(frozen_assignments)))


    # TODO: outros metodos da classe Board





class Nuruomino(Problem):
    def __init__(self, initial_board: Board):
        """O construtor especifica o estado inicial."""
        # Chama o construtor da superclasse (Problem)
        super().__init__(NuruominoState(initial_board))
        
        # Pré-processamento útil:
        self.all_region_ids = sorted(list(initial_board.regions_map.keys()))
        self.N = initial_board.N
        
        # Gera as variantes de tetraminós uma vez, se ainda não foram geradas
        if not TETROMINO_VARIANTS:
            generate_all_tetromino_variants()


    def actions(self, state: NuruominoState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento. 
        Uma ação é (region_id_to_fill, tetromino_type, frozenset_of_absolute_cells_to_fill)
        """
        board = state.board # O estado contém um objeto Board
        
        # Encontrar a primeira região (por ordem canónica) ainda não preenchida
        region_to_fill_id = None
        for r_id in self.all_region_ids:
            if r_id not in board.assignments:
                region_to_fill_id = r_id
                break
        
        if region_to_fill_id is None: # Todas as regiões preenchidas, não há mais ações
            return []

        possible_actions = []
        region_cells_set = board.get_region_cells(region_to_fill_id)

        for tetromino_type in ALLOWED_TETROMINO_TYPES:
            if tetromino_type not in TETROMINO_VARIANTS: continue # Segurança

            for variant_rel_coords_list in TETROMINO_VARIANTS[tetromino_type]:
                variant_rel_coords = frozenset(variant_rel_coords_list) # Garantir frozenset
                
                # Tentar ancorar a variante em cada célula da região
                # A âncora pode ser a célula mais a cima e à esquerda da peça,
                # e esta âncora da peça deve estar numa célula da região.
                
                # Como as variantes já estão normalizadas (canto sup-esq em (0,0) relativo)
                # podemos iterar pelas células da região como o ponto de ancoragem para (0,0) da peça.
                for anchor_r, anchor_c in region_cells_set:
                    abs_cells_to_fill = frozenset(
                        (r_rel + anchor_r, c_rel + anchor_c)
                        for r_rel, c_rel in variant_rel_coords
                    )
                    
                    # Validações para esta colocação:
                    # 1. Peça cabe inteiramente na região (todas as abs_cells estão em region_cells_set)
                    if not abs_cells_to_fill.issubset(region_cells_set):
                        continue

                    # 2. Regra de adjacência de tetraminós do mesmo tipo
                    violates_adjacency = False
                    for r_cell, c_cell in abs_cells_to_fill:
                        for dr_adj, dc_adj in [(0,1), (0,-1), (1,0), (-1,0)]:
                            nr, nc = r_cell + dr_adj, c_cell + dc_adj
                            if 0 <= nr < board.N and 0 <= nc < board.N:
                                neighbor_val = board.solution_grid[nr][nc]
                                neighbor_region_id = board.initial_grid_ids[nr][nc]
                                if (neighbor_val in ALLOWED_TETROMINO_TYPES and
                                    neighbor_region_id != region_to_fill_id and # Diferente região
                                    board.assignments.get(neighbor_region_id, {}).get('type') == tetromino_type):
                                    violates_adjacency = True
                                    break
                        if violates_adjacency: break
                    
                    if violates_adjacency:
                        continue
                    
                    # 3. Regra de não formar 2x2 com esta nova peça E as já existentes
                    # Esta verificação é mais complexa aqui, pois envolve verificar todos os possíveis 2x2
                    # que esta nova peça poderia criar. Pode ser mais eficiente verificar no goal_test
                    # ou após a colocação, mas para pruning, pode ser feita aqui (parcialmente).
                    # Por agora, vamos deixar esta verificação mais pesada para o goal_test.

                    action = (region_to_fill_id, tetromino_type, abs_cells_to_fill)
                    possible_actions.append(action)
        
        return possible_actions


    def result(self, state: NuruominoState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state).
        A ação é (region_id, tetromino_type, frozenset_of_absolute_cells).
        """
        region_id, tetromino_type, abs_cells = action
        
        # Cria uma cópia profunda dos assignments atuais para não modificar o estado original
        new_assignments = deepcopy(state.board.assignments)
        new_assignments[region_id] = {'type': tetromino_type, 'abs_cells': abs_cells}
        
        # Cria um novo Board com os assignments atualizados.
        # O __init__ do Board irá reconstruir a solution_grid com base nos novos assignments.
        new_board = Board(state.board.N, state.board.initial_grid_ids, 
                          state.board.regions_map, new_assignments)
        
        return NuruominoState(new_board)
        

    def goal_test(self, state: NuruominoState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas de acordo com as regras do problema."""
   
        board = state.board

        # 1. Todas as regiões devem estar preenchidas
        if len(board.assignments) != len(self.all_region_ids):
            return False

        # Se chegou aqui, todas as regiões têm uma peça atribuída.
        # Agora, verificar as regras globais do Nuruomino:
        
        all_filled_cells = set()
        for region_data in board.assignments.values():
            all_filled_cells.update(region_data['abs_cells'])

        if not all_filled_cells: # Se não há células preenchidas (e.g. 0 regiões)
             return len(self.all_region_ids) == 0 # É um goal se não havia regiões para preencher


        # 2. Conectividade de todas as células preenchidas (formar um único poliminó)
        q = [next(iter(all_filled_cells))] # Pega uma célula preenchida qualquer para iniciar BFS/DFS
        visited_connected = set()
        head = 0
        while head < len(q):
            r, c = q[head]
            head += 1
            if (r,c) in visited_connected:
                continue
            visited_connected.add((r,c))

            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in all_filled_cells and (nr, nc) not in visited_connected:
                    q.append((nr, nc))
        
        if len(visited_connected) != len(all_filled_cells):
            return False # Nem todas as células preenchidas estão conectadas

        # 3. Não pode haver áreas 2x2 de células preenchidas
        for r in range(self.N - 1):
            for c in range(self.N - 1):
                is_2x2_filled = True
                for dr_block in [0, 1]:
                    for dc_block in [0, 1]:
                        if (r + dr_block, c + dc_block) not in all_filled_cells:
                            is_2x2_filled = False
                            break
                    if not is_2x2_filled: break
                if is_2x2_filled:
                    return False # Encontrou um bloco 2x2 preenchido
        
        # Se passou por todas as verificações, é um estado objetivo
        return True


    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*.
           Estima o "custo" para alcançar o objetivo a partir do estado atual.
           Uma heurística simples é o número de regiões ainda não preenchidas.
        """
        state = node.state # O estado é um NuruominoState
        board = state.board
        
        # Número de regiões que ainda precisam de uma peça.
        # Quanto menor, mais perto do objetivo (em termos de peças colocadas).
        unassigned_regions = len(self.all_region_ids) - len(board.assignments)
        return unassigned_regions




if __name__ == '__main__':
    # Nomes dos ficheiros de input e output esperado
    expected_output_filename = "test-01.out"

    DEBUG_MODE = False
    DEBUG_MODE_COMPARISON = False

    def dprint(message):
        if DEBUG_MODE:
            print(message, file=sys.stderr)
    
    def dprint_c(message):
        if DEBUG_MODE_COMPARISON:
            print(message, file=sys.stderr)

    dprint("--- NURUOMINO Solver ---")
    
    # 1. Gerar as variantes dos tetraminós
    try:
        generate_all_tetromino_variants()
        dprint("Variantes de tetraminós geradas.")
    except Exception as e:
        print(f"ERRO FATAL ao gerar variantes: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Ler a instância do problema
    initial_board_obj = None
    try:
        dprint("A ler a instância do problema do standard input...")
        initial_board_obj = Board.parse_instance()
        dprint(f"Instância lida: Grelha {initial_board_obj.N}x{initial_board_obj.N} com {len(initial_board_obj.regions_map)} regiões.")
    except Exception as e:
        print(f"ERRO FATAL ao ler instância: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Criar a instância do problema Nuruomino
    nuruomino_problem = None
    try:
        nuruomino_problem = Nuruomino(initial_board_obj)
        dprint("Instância do problema Nuruomino criada.")
    except Exception as e:
        print(f"ERRO FATAL ao criar problema: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Resolver o problema
    goal_node = None
    dprint(f"\nA tentar resolver o problema com depth_first_tree_search...")
    try:
        goal_node = astar_search(nuruomino_problem)
    except Exception as e:
        print(f"ERRO FATAL durante a procura: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 5. Apresentar a solução
    if goal_node:
        dprint("\nSOLUÇÃO ENCONTRADA pelo algoritmo de procura!")
        
        my_solution_str = goal_node.state.board.print_board()

        print(my_solution_str)

        # --- Comparação com o ficheiro .out (para debug/teste local) ---
        if DEBUG_MODE_COMPARISON:
            try:
                with open(expected_output_filename, 'r') as f:
                    expected_solution_str = f.read().strip()
                
                my_solution_str_normalized = my_solution_str.replace('\r\n', '\n').strip()
                expected_solution_str_normalized = expected_solution_str.replace('\r\n', '\n').strip()

                if my_solution_str_normalized == expected_solution_str_normalized:
                    dprint_c(f"\nVERIFICAÇÃO: Output do programa CORRESPONDE ao ficheiro '{expected_output_filename}'. SUCESSO!")
                else:
                    dprint_c(f"\nVERIFICAÇÃO: Output do programa DIFERE do ficheiro '{expected_output_filename}'. FALHA!")
                    dprint_c("--- Output Esperado ---")
                    dprint_c(expected_solution_str_normalized)
                    dprint_c("--- Output Obtido ---")
                    dprint_c(my_solution_str_normalized)
            except FileNotFoundError:
                dprint_c(f"\nAVISO: Ficheiro de output esperado '{expected_output_filename}' não encontrado para verificação.")
            except Exception as e:
                dprint_c(f"\nERRO ao ler ou comparar com o ficheiro de output esperado: {e}")

    else:
        dprint("\nNenhuma solução encontrada pelo algoritmo de procura.")

    if DEBUG_MODE:
        dprint("\n--- Fim da Execução ---")