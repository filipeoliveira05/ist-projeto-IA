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
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1

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
            output += " ".join(self.solution_grid[r]) + "\n"
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
    print("--- Iniciando Testes dos Exemplos ---")
    
    # ---------------------------------------------------------------------------
    # Preparação Comum: Ler a grelha uma vez para todos os exemplos
    # ---------------------------------------------------------------------------
    initial_board_obj = None
    try:
        # Simula a leitura do stdin apenas uma vez no início
        # Se for executar com redirecionamento, Board.parse_instance() lerá do stdin real
        print("A ler a instância do problema (stdin)...")
        initial_board_obj = Board.parse_instance()
        print("Instância lida com sucesso.")
        generate_all_tetromino_variants() # Gerar variantes de tetraminós
        print("Variantes de tetraminós geradas.")
    except Exception as e:
        print(f"ERRO FATAL ao ler instância ou gerar variantes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) # Termina se não conseguir ler a grelha

    # ---------------------------------------------------------------------------
    # Exemplo 1: board.adjacent_regions()
    # ---------------------------------------------------------------------------
    print("\n--- Teste do Exemplo 1 ---")
    if initial_board_obj:
        try:
            # Usar strings para IDs de região
            region_id_1_str = "1"
            region_id_3_str = "3"
            
            print(f"initial_board_obj.adjacent_regions('{region_id_1_str}')")
            print(initial_board_obj.adjacent_regions(region_id_1_str)) # Output: [2, 3]

            print(f"initial_board_obj.adjacent_regions('{region_id_3_str}')")
            print(initial_board_obj.adjacent_regions(region_id_3_str)) # Output: [1, 2, 4, 5]
        except Exception as e:
            print(f"Erro no Exemplo 1: {e}")
            import traceback; traceback.print_exc()

    # ---------------------------------------------------------------------------
    # Exemplo 2: problem.result() e board.get_value(), board.adjacent_values()
    # ---------------------------------------------------------------------------
    print("\n--- Teste do Exemplo 2 ---")
    if initial_board_obj:
        try:
            problem_ex2 = Nuruomino(deepcopy(initial_board_obj)) # Usar uma cópia para não afetar outros testes
            initial_state_ex2 = problem_ex2.initial # Equivalente a NuruominoState(board)

            print(f"initial_state_ex2.board.get_value(2, 1)")
            # A grelha da Figura 5/Input tem '1' na região (2,1) no input
            # Célula (2,0) é da região '1', (2,1) é da região '3' no input fornecido.
            # Corrigindo o exemplo para (2,0) que é da região '1'
            print(initial_state_ex2.board.get_value(2, 0)) # Output: 1 (ID da região original)

            # Ação: (region_id, 'L', frozenset_of_absolute_cells)
            # Para a região "1" (células (0,0),(0,1),(1,0),(2,0) no input)
            # A peça 'L' na solução da Figura 4b ocupa (0,0),(1,0),(2,0),(2,1)
            # No entanto, a célula (2,1) NÃO pertence à região "1" do input.
            # A região "1" no input é: (0,0), (0,1), (1,0), (2,0)
            # Uma peça L válida para a região "1" seria, por exemplo:
            # frozenset([(0,0), (1,0), (2,0), (0,1)]) - que é um L rodado
            # O Exemplo 2 diz "cuja forma é [[1,1],[1,0],[1,0]]". Isso é confuso.
            # Vamos usar as coordenadas absolutas da PEÇA L da SOLUÇÃO para a REGIÃO "1":
            # Região "1": (0,0), (0,1), (1,0), (2,0)
            # Peça L na solução: (0,0), (1,0), (2,0), (0,1) [Esta é uma peça L que cabe na região "1"]
            # O enunciado do exemplo 2 (1, 'L', [[1, 1],[1, 0],[1, 0]]) não especifica as coordenadas absolutas.
            # Vou definir uma peça L que cabe na região '1' e corresponde à solução
            action_L_region1 = ("1", 'L', frozenset([(0,0), (1,0), (2,0), (0,1)])) 
                                        # Esta forma de L ((0,0),(1,0),(2,0),(0,1)) é uma rotação/reflexão do L base
                                        # e cabe perfeitamente na região "1" do input: (0,0),(0,1),(1,0),(2,0)

            print(f"problem_ex2.result(initial_state_ex2, {action_L_region1})")
            s1_ex2 = problem_ex2.result(initial_state_ex2, action_L_region1)
            
            print(f"s1_ex2.board.get_value(2, 0)") # (2,0) agora deve ser 'L'
            print(s1_ex2.board.get_value(2, 0))   # Output: L
            
            print(f"s1_ex2.board.adjacent_values(2, 2)") # Posição (2,2) é da região "3" (input)
                                                         # Vizinhos de (2,2) após colocar L em (2,0):
                                                         # (1,1)R2, (1,2)R2, (1,3)R2
                                                         # (2,1)R3,           (2,3)R2 <--- L foi colocado na reg 1
                                                         # (3,1)R3, (3,2)R3, (3,3)R3
                                                         # Na solution_grid do s1_ex2:
                                                         # (2,0) é L. (0,0)L, (1,0)L, (0,1)L
                                                         # (2,2) continua '3'.
                                                         # Vizinhos de (2,2) em s1_ex2.solution_grid:
                                                         # (1,1)='2', (1,2)='2', (1,3)='2'
                                                         # (2,1)='3',            (2,3)='2'
                                                         # (3,1)='3', (3,2)='3', (3,3)='3'
                                                         # O exemplo de output [L,L,2,L,2,L,3,3] parece implicar que a peça L foi colocada
                                                         # de forma a afetar os vizinhos de (2,2) e que (2,1) virou L.
                                                         # Isso só aconteceria se a região 1 incluísse (2,1) E a peça L o cobrisse.
                                                         # Com a Região "1" do input, o output esperado é diferente.
                                                         # Output do Exemplo: [L,L,2,L,2,L,3,3]
                                                         # Meu Output esperado com a peça L em (0,0),(1,0),(2,0),(0,1):
                                                         # Vizinhos de (2,2) na solution_grid de s1_ex2:
                                                         # (1,1)R2 (1,2)R2 (1,3)R2
                                                         # (2,1)R3         (2,3)R2
                                                         # (3,1)R3 (3,2)R3 (3,3)R3
                                                         # Se (2,1) é '3' e (2,0) é 'L':
                                                         # adj_vals(2,2) => ['2','2','2', '3','2', '3','3','3']
            print(s1_ex2.board.adjacent_values(2,2)) # O output dependerá da peça L exata
        except Exception as e:
            print(f"Erro no Exemplo 2: {e}")
            import traceback; traceback.print_exc()

    # ---------------------------------------------------------------------------
    # Exemplo 3: Sequência de ações para resolver e goal_test
    # ---------------------------------------------------------------------------
    print("\n--- Teste do Exemplo 3 ---")
    if initial_board_obj:
        try:
            problem_ex3 = Nuruomino(deepcopy(initial_board_obj))
            s0_ex3 = problem_ex3.initial

            # Definir as peças da SOLUÇÃO da Figura 4b para cada região do INPUT da Figura 5
            # Região "1" (input): (0,0),(0,1),(1,0),(2,0) -> Peça L na Solução: (0,0),(1,0),(2,0),(0,1)
            # Região "2" (input): (0,2),(0,3),(1,1),(1,2),(1,3),(2,3) -> Peça S na Solução: (0,2),(0,3),(1,2),(1,3)
            # Região "3" (input): (0,4)... -> Peça T na Solução: (3,3),(4,2),(4,3),(4,4) (Precisa verificar se cabe!)
            # Região "4" (input): (4,0),(4,1),(4,2),(5,0) -> Peça L na Solução: (4,0),(5,0),(4,1),(4,2) (outro L)
            # Região "5" (input): (2,5),(3,5),(4,5),(5,5) -> Peça I na Solução: (2,5),(3,5),(4,5),(5,5)
            
            # Ações (region_id_str, type, frozenset_abs_coords)
            # Estas são as peças da SOLUÇÃO DA FIGURA 4b, mapeadas para as regiões do INPUT DA FIGURA 5
            action1 = ("1", 'L', frozenset([(0,0),(1,0),(2,0),(0,1)])) # L rosa em (a1,a2,a3,b1)
            action2 = ("2", 'S', frozenset([(0,2),(0,3),(1,2),(1,3)])) # S verde em (c1,d1,c2,d2)
            # Para a Região "3" do input, a peça T da solução é (3,3),(4,2),(4,3),(4,4)
            # Células da Região "3" (input): (0,4),(0,5),(1,4),(1,5),(2,1),(2,2),(2,4),(3,0),(3,1),(3,2),(3,3),(3,4),(4,3),(4,4),(5,1),(5,2),(5,3),(5,4)
            # Esta peça T ((3,3),(4,2),(4,3),(4,4)) não cabe na região "3". (4,2) não está na região "3".
            # O Exemplo 3 parece usar uma numeração de regiões e/ou forma de peças diferente da Figura 4b/5.
            # Vou usar as peças da *solução dada no output do exemplo 3*:
            # Peça L (rosa) na Região 1: (0,0),(1,0),(2,0),(0,1) -> corresponde à Figura 4b (0,0),(1,0),(2,0),(2,1) se a região 1 fosse maior
            # Peça S (verde) na Região 2: (0,2),(0,3),(1,2),(1,3)
            # Peça T (roxo claro) na Região ??: (3,3),(4,3),(5,3),(4,2) - esta é (d4,d5,d6,c5)
            # Peça L (rosa escuro) na Região ??: (4,0),(5,0),(5,1),(5,2) - esta é (a5,a6,b6,c6)
            # Peça I (azul) na Região ??: (2,5),(3,5),(4,5),(5,5)
            
            # Mapeando a SOLUÇÃO do Exemplo 3 para as regiões do INPUT (Figura 5)
            # Região "1" (input) -> L: (0,0),(1,0),(2,0),(0,1)
            # Região "2" (input) -> S: (0,2),(1,2),(0,3),(1,3)
            # Região "3" (input) -> T: A peça T na sol. exemplo 3 é (d4,d5,d6,c5) = (3,3),(4,3),(5,3),(4,2)
                                     # Células da Região 3 (input): (0,4),(0,5),(1,4),(1,5),(2,1),(2,2),(2,4), (3,0),(3,1),(3,2),(3,3),(3,4), (4,3),(4,4), (5,1),(5,2),(5,3),(5,4)
                                     # (4,2) NÃO ESTÁ na Região 3 do input. (5,3) ESTÁ.
                                     # O T da solução do exemplo está em (3,3),(4,3),(5,3) e (4,2).
                                     # Para caber na região 3 do input, um T poderia ser (3,3),(4,3),(5,3),(3,2) ou (3,3),(4,3),(5,3),(4,4)
                                     # O T da solução é (d4)T (d5)T (d6)T (c5)T. (3,3)T (4,3)T (5,3)T (4,2)T
                                     # No input, (4,2) é Região "4".
                                     # ISTO INDICA QUE A NUMERAÇÃO DE REGIÕES DO EXEMPLO 3 PODE NÃO SER A DA FIGURA 5.
                                     # OU as peças do exemplo são simbólicas e não representam as coordenadas exatas.

            # Para Exemplo 3, vou assumir que as ações são para as regiões 1,2,3,4,5 do *problema*
            # e que as peças são as da solução final.
            # As peças da solução final são:
            sol_L1 = ("1", 'L', frozenset([(0,0),(1,0),(2,0),(0,1)])) # Região "1" do input
            sol_S2 = ("2", 'S', frozenset([(0,2),(1,2),(0,3),(1,3)])) # Região "2" do input
            # Para a solução do Exemplo 3: T roxo é (c5,d4,d5,d6) = (4,2),(3,3),(4,3),(5,3)
            # Região "3" do input contém (3,3),(4,3),(5,3) e (3,2),(3,4),(4,4),(5,1),(5,2),(5,4) etc. NÃO contém (4,2)
            # Região "4" do input contém (4,0),(4,1),(4,2),(5,0). A peça T (4,2) está aqui.
            # Região "5" do input contém (2,5),(3,5),(4,5),(5,5)
            
            # Tentativa de mapear as peças da SOLUÇÃO dada no output do Exemplo 3
            # para as REGIÕES do INPUT (Figura 5)
            # L rosa claro (a1,b1,a2,a3) -> (0,0),(0,1),(1,0),(2,0) -> Região 1
            s1_ex3 = problem_ex3.result(s0_ex3, ("1", 'L', frozenset([(0,0),(0,1),(1,0),(2,0)])))
            
            # S verde (c1,d1,c2,d2) -> (0,2),(0,3),(1,1),(1,2) -> Região 2 (note que (1,1) é da reg 2 no input)
            s2_ex3 = problem_ex3.result(s1_ex3, ("2", 'S', frozenset([(0,2),(0,3),(1,1),(1,2)])))
            
            # T roxo claro (c5,d4,d5,d6) -> (4,2),(3,3),(4,3),(5,3)
            # A célula (4,2) é da Região "4". As células (3,3),(4,3),(5,3) são da Região "3".
            # Esta peça T não pode ser colocada numa única região do input.
            # O EXEMPLO 3 PARECE ASSUMIR UMA DEFINIÇÃO DE REGIÕES DIFERENTE DA FIGURA 5 / test-01.txt
            # OU as ações dadas são simbólicas e o `problem.result` deve encontrar uma variante.
            # Dado que `problem.result` recebe coordenadas absolutas, o Exemplo 3 está mal definido
            # se usarmos o input de test-01.txt que corresponde à Figura 5.

            # VOU IGNORAR AS AÇÕES ESPECÍFICAS DO EXEMPLO 3 E TESTAR APENAS COM O goal_node DO EXEMPLO 4
            # porque as ações do Exemplo 3 não são consistentes com o input test-01.txt / Figura 5
            # E a definição de 'problem.result'.
            print("AVISO: As ações do Exemplo 3 não são consistentes com o input 'test-01.txt' (Figura 5).")
            print("A testar goal_test com um estado parcialmente preenchido e um estado objetivo (se Exemplo 4 funcionar).")
            # Para testar goal_test(s2), precisamos de um s2 válido.
            # Se s2_ex3 acima for válido (apenas peças em Região 1 e 2):
            if 's2_ex3' in locals():
                 print(f"problem_ex3.goal_test(s2_ex3)")
                 print(problem_ex3.goal_test(s2_ex3)) # Output: False

        except Exception as e:
            print(f"Erro no Exemplo 3: {e}")
            import traceback; traceback.print_exc()

    # ---------------------------------------------------------------------------
    # Exemplo 4: depth_first_tree_search
    # ---------------------------------------------------------------------------
    print("\n--- Teste do Exemplo 4 ---")
    if initial_board_obj:
        try:
            problem_ex4 = Nuruomino(deepcopy(initial_board_obj))
            print("A executar depth_first_tree_search... (Pode demorar!)")
            goal_node_ex4 = depth_first_tree_search(problem_ex4)
            
            if goal_node_ex4:
                print("Nó objetivo encontrado pela procura em profundidade!")
                print(f"problem_ex4.goal_test(goal_node_ex4.state)")
                is_goal = problem_ex4.goal_test(goal_node_ex4.state)
                print(is_goal) # Output: True
                
                print("Solution:\n", goal_node_ex4.state.board.print_board(), sep="")
                # Output: (a grelha da solução)

                # Agora podemos usar este goal_node.state para o teste final do Exemplo 3
                if 'problem_ex3' in locals() and is_goal: # Se Exemplo 3 foi inicializado
                    print(f"\nRetomando Exemplo 3 com estado objetivo do Exemplo 4:")
                    print(f"problem_ex3.goal_test(goal_node_ex4.state)") # Reusa problem_ex3 para consistência
                    print(problem_ex3.goal_test(goal_node_ex4.state)) # Output: True
            else:
                print("Nenhum nó objetivo encontrado pela procura em profundidade.")
        except Exception as e:
            print(f"Erro no Exemplo 4: {e}")
            import traceback; traceback.print_exc()


    print("\n--- Fim de Todos os Testes ---")