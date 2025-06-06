# nuruomino.py: Projeto de Inteligência Artificial 2024/2025
# Grupo 132:
# 110633 Filipe Oliveira
# 110720 Francisco Andrade

import sys
from search import Problem, best_first_graph_search
from collections import deque

# Global debug flag
DEBUG_MODE = False

# --- Global Definitions ---
TETROMINO_BASE_SHAPES = {
    "L": {
        "shape": frozenset(
            [
                (0, 0),
                (1, 0),
                (2, 0),
                (2, 1),
            ]
        ),
        "forbidden_relative": [(1, 1)],
    },
    "I": {
        "shape": frozenset([(0, 0), (1, 0), (2, 0), (3, 0)]),
        "forbidden_relative": [],
    },
    "T": {
        "shape": frozenset([(0, 0), (0, 1), (0, 2), (1, 1)]),
        "forbidden_relative": [
            (1, 0),
            (1, 2),
        ],
    },
    "S": {
        "shape": frozenset([(0, 1), (1, 1), (1, 0), (2, 0)]),
        "forbidden_relative": [
            (0, 0),
            (2, 1),
        ],
    },
}
ALLOWED_TETROMINO_TYPES = ["I", "T", "S", "L"]
TETROMINO_VARIANTS = {}


# --- Tetromino Manipulation Functions ---
def _normalize_shape(shape_coords):
    """
    Normalizes a tetromino shape by shifting its cells so that the top-leftmost cell
    (minimum row, then minimum column) is at (0,0).
    Ensures consistent representation for shape comparison regardless of absolute position.
    """
    if not shape_coords:
        return frozenset()

    min_r = min(r for r, c in shape_coords)
    min_c = min(c for r, c in shape_coords)

    return frozenset([(r - min_r, c - min_c) for r, c in shape_coords])


def _rotate_shape_90_clockwise(shape_coords):
    """
    Rotates a shape 90 degrees clockwise around the origin (0,0).
    (r, c) becomes (c, -r). Normalization is applied afterwards.
    """
    return _normalize_shape(frozenset([(c, -r) for r, c in shape_coords]))


def _reflect_shape_vertical_axis(shape_coords):
    """
    Reflects a shape across the vertical axis (c -> -c).
    Normalization is applied afterwards.
    """
    return _normalize_shape(frozenset([(r, -c) for r, c in shape_coords]))


def generate_all_tetromino_variants():
    global TETROMINO_VARIANTS
    if TETROMINO_VARIANTS:
        return

    for name, info in TETROMINO_BASE_SHAPES.items():
        base_shape = info["shape"]
        variants = set()
        current_shape_for_rotation = base_shape

        for i in range(4):
            normalized_rotated = _normalize_shape(current_shape_for_rotation)
            variants.add(normalized_rotated)

            reflected = _reflect_shape_vertical_axis(current_shape_for_rotation)
            variants.add(_normalize_shape(reflected))

            current_shape_for_rotation = _rotate_shape_90_clockwise(
                current_shape_for_rotation
            )

        TETROMINO_VARIANTS[name] = [list(variant) for variant in variants]
        if DEBUG_MODE:
            print(
                f"Generated {len(TETROMINO_VARIANTS[name])} variants for {name}",
                file=sys.stderr,
            )


def _check_if_creates_2x2_block(N, existing_filled_cells, new_cells_to_add):
    """
    Checks if adding 'new_cells_to_add' to 'existing_filled_cells' creates a 2x2 block.
    N is the board size.
    """
    all_filled = existing_filled_cells | new_cells_to_add

    for r_new, c_new in new_cells_to_add:
        if r_new + 1 < N and c_new + 1 < N:
            if all(
                (_r, _c) in all_filled
                for _r, _c in [
                    (r_new, c_new),
                    (r_new + 1, c_new),
                    (r_new, c_new + 1),
                    (r_new + 1, c_new + 1),
                ]
            ):
                return True
            
        if r_new + 1 < N and c_new - 1 >= 0:
            if all(
                (_r, _c) in all_filled
                for _r, _c in [
                    (r_new, c_new - 1),
                    (r_new + 1, c_new - 1),
                    (r_new, c_new),
                    (r_new + 1, c_new),
                ]
            ):
                return True

        if r_new - 1 >= 0 and c_new + 1 < N:
            if all(
                (_r, _c) in all_filled
                for _r, _c in [
                    (r_new - 1, c_new),
                    (r_new, c_new),
                    (r_new - 1, c_new + 1),
                    (r_new, c_new + 1),
                ]
            ):
                return True

        if r_new - 1 >= 0 and c_new - 1 >= 0:
            if all(
                (_r, _c) in all_filled
                for _r, _c in [
                    (r_new - 1, c_new - 1),
                    (r_new, c_new - 1),
                    (r_new - 1, c_new),
                    (r_new, c_new),
                ]
            ):
                return True
    return False


# --- Classes ---
class NuruominoState:
    """Represents a state in the Nuruomino problem."""

    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = NuruominoState.state_id
        NuruominoState.state_id += 1

    def __lt__(self, other):
        """ This method is used in the event of a tie in the management of the list
        of open cases in reported searches. """
        return self.id < other.id


class Board:
    """Represents the Nuruomino game board, its regions, and current assignments."""

    def __init__(
        self,
        N,
        initial_grid_ids,
        regions_map,
        cell_to_region_id_map,
        assignments=None,
        forbidden_cells=None,
        filled_cells=None,
    ):
        self.N = N
        self.initial_grid_ids = [row[:] for row in initial_grid_ids]
        self.regions_map = {k: {"cells": set(v["cells"]), "valid_placements": list(v["valid_placements"])} for k, v in regions_map.items()}
        self.cell_to_region_id_map = cell_to_region_id_map
        self.assignments = (
            assignments or {}
        )

        self._filled_cells = (
            filled_cells
            if filled_cells is not None
            else self._calculate_initial_filled_cells()
        )
        self._forbidden_cells = (
            set(forbidden_cells) if forbidden_cells is not None else set()
        )

        self.solution_grid = [
            row[:] for row in self.initial_grid_ids
        ]
        for region_id, data in self.assignments.items():
            for r, c in data["abs_cells"]:
                self.solution_grid[r][c] = data["type"]

    def _calculate_initial_filled_cells(self):
        """Helper to calculate filled_cells from assignments if not provided."""
        filled = set()
        for data in self.assignments.values():
            filled.update(data["abs_cells"])
        return frozenset(filled)

    def print_board(self):
        """Prints the grid (solution_grid) showing 'X' for prohibited cells."""
        output = ""
        for r in range(self.N):
            row_str = []
            for c in range(self.N):
                # if (r, c) in self.forbidden_cells:
                #     row_str.append("X")

                row_str.append(self.solution_grid[r][c])
            output += "\t".join(row_str) + "\n"
        return output.strip()

    @staticmethod
    def parse_instance():
        """Parses the Nuruomino puzzle input from stdin."""
        lines = [line.strip().split() for line in sys.stdin if line.strip()]
        if not lines:
            raise ValueError("Empty input")

        N = len(lines)
        for i, row in enumerate(lines):
            if len(row) != N:
                raise ValueError(
                    f"Invalid input: row {i+1} has {len(row)} columns, expected {N}"
                )

        initial_grid_ids = lines

        regions_map = {}
        cell_to_region_id_map = {}
        for r in range(N):
            for c in range(N):
                rid = initial_grid_ids[r][c]
                if rid not in regions_map:
                    regions_map[rid] = {"cells": set(), "valid_placements": []}
                regions_map[rid]["cells"].add((r, c))
                cell_to_region_id_map[(r, c)] = rid

        generate_all_tetromino_variants()

        for rid, data in regions_map.items():
            cells = data["cells"]
            valid_placements = set()

            for tet_type in ALLOWED_TETROMINO_TYPES:
                for variant_coords_list in TETROMINO_VARIANTS[tet_type]:
                    variant_coords_frozen = frozenset(variant_coords_list)

                    for vr_anchor, vc_anchor in variant_coords_frozen:
                        for r_region_cell, c_region_cell in cells:
                            offset_r = r_region_cell - vr_anchor
                            offset_c = c_region_cell - vc_anchor

                            current_placement_abs_cells = frozenset(
                                [
                                    (vr + offset_r, vc + offset_c)
                                    for vr, vc in variant_coords_frozen
                                ]
                            )

                            if not current_placement_abs_cells.issubset(cells):
                                continue

                            if len(current_placement_abs_cells) != 4:
                                continue

                            valid_placements.add(
                                (tet_type, current_placement_abs_cells)
                            )

            data["valid_placements"] = list(valid_placements)
            if DEBUG_MODE:
                print(
                    f"Region {rid}: {len(data['valid_placements'])} initial valid tetromino placements.",
                    file=sys.stderr,
                )
            if not data["valid_placements"]:
                print(
                    f"WARNING: Region {rid} has no initial valid placements possible! This puzzle is likely unsolvable.",
                    file=sys.stderr,
                )

        return Board(
            N,
            initial_grid_ids,
            regions_map,
            cell_to_region_id_map,
            {},
            set(),
            frozenset(),
        )
    
    def __eq__(self, other):
        if not isinstance(other, Board):
            return False
        return (
            self.assignments == other.assignments
            and self._filled_cells == other._filled_cells
            and self._forbidden_cells == other._forbidden_cells
        )

    def __hash__(self):
        assignments_tuple = tuple(
            (rid, data["type"], data["abs_cells"])
            for rid, data in sorted(self.assignments.items())
        )
        
        forbidden_tuple = frozenset(self._forbidden_cells)
        filled_tuple = frozenset(self._filled_cells)
        return hash((self.N, assignments_tuple, forbidden_tuple, filled_tuple))

    def get_filled_cells(self):
        """Returns a frozenset of all currently filled cells on the board."""
        return self._filled_cells

    def get_forbidden_cells(self):
        """Returns a frozenset of all currently forbidden cells on the board."""
        return self._forbidden_cells

    def mark_forbidden_2x2(self, new_piece_abs_cells):
        """
        Identifies and updates cells that would complete a 2x2 block if filled,
        given a newly placed piece. Updates self._forbidden_cells.
        """
        n = self.N

        self._forbidden_cells -= new_piece_abs_cells

        cells_to_scan = set(new_piece_abs_cells)
        for r, c in new_piece_abs_cells:
            for dr, dc in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    cells_to_scan.add((nr, nc))

        newly_forbidden_candidates = set()

        for r_target, c_target in cells_to_scan:
            for dr_offset, dc_offset in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
                r_top_left, c_top_left = r_target + dr_offset, c_target + dc_offset

                if 0 <= r_top_left < n - 1 and 0 <= c_top_left < n - 1:
                    block_cells = frozenset(
                        [
                            (r_top_left, c_top_left),
                            (r_top_left + 1, c_top_left),
                            (r_top_left, c_top_left + 1),
                            (r_top_left + 1, c_top_left + 1),
                        ]
                    )

                    filled_in_block = block_cells.intersection(self._filled_cells)

                    if len(filled_in_block) == 3:
                        empty_cells_in_block = block_cells - filled_in_block
                        forbidden_cell = next(iter(empty_cells_in_block))

                        if forbidden_cell not in self._filled_cells:
                            newly_forbidden_candidates.add(forbidden_cell)

        self._forbidden_cells.update(newly_forbidden_candidates)
        if DEBUG_MODE:
            print(
                f"MarkForbidden - Placed: {new_piece_abs_cells}, Current Forbidden: {sorted(list(self._forbidden_cells))}",
                file=sys.stderr,
            )

    def recalculate_valid_placements(self, region_id):
        """
        Recalculates the valid tetromino placements for a given region,
        considering the currently filled and forbidden cells in THIS board state.
        This is the core of forward checking. It filters out invalid options
        from the region's initial domain.
        Returns the updated list of valid placements.
        """
        if region_id not in self.regions_map:
            return []

        region_cells = self.regions_map[region_id]["cells"]
        filtered_placements = []

        initial_placements_for_region = self.regions_map[region_id]["valid_placements"]

        occupied_or_forbidden = self._filled_cells | self._forbidden_cells

        for tet_type, abs_cells_candidate in initial_placements_for_region:
            if not abs_cells_candidate.isdisjoint(occupied_or_forbidden):
                if DEBUG_MODE:
                    print(
                        f"  Filtering {tet_type} at {sorted(list(abs_cells_candidate))}: Overlaps or uses forbidden cell.",
                        file=sys.stderr,
                    )
                continue

            if _check_if_creates_2x2_block(
                self.N, self._filled_cells, abs_cells_candidate
            ):
                if DEBUG_MODE:
                    print(
                        f"  Filtering {tet_type} at {sorted(list(abs_cells_candidate))}: Creates 2x2.",
                        file=sys.stderr,
                    )
                continue

            violates_adjacency = False
            for r_cand, c_cand in abs_cells_candidate:
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r_cand + dr, c_cand + dc
                    if (
                        nr,
                        nc,
                    ) in self._filled_cells:
                        neighbor_region_id = self.cell_to_region_id_map.get((nr, nc))
                        if (
                            neighbor_region_id
                            and neighbor_region_id in self.assignments
                        ):
                            neighbor_piece_type = self.assignments[neighbor_region_id][
                                "type"
                            ]
                            if (
                                neighbor_piece_type == tet_type
                                and neighbor_region_id != region_id
                            ):
                                violates_adjacency = True
                                break
                if violates_adjacency:
                    break

            if violates_adjacency:
                if DEBUG_MODE:
                    print(
                        f"  Filtering {tet_type} at {sorted(list(abs_cells_candidate))}: Violates same-type adjacency.",
                        file=sys.stderr,
                    )
                continue

            filtered_placements.append((tet_type, abs_cells_candidate))

        self.regions_map[region_id]["valid_placements"] = filtered_placements
        if DEBUG_MODE:
            print(
                f"Region {region_id}: Recalculated {len(filtered_placements)} valid placements.",
                file=sys.stderr,
            )
        return filtered_placements


class Nuruomino(Problem):
    """
    Represents the Nuruomino puzzle as a search problem.
    Uses Best-First Graph Search with a heuristic.
    """

    def __init__(self, initial_board):
        super().__init__(NuruominoState(initial_board))
        self.N = initial_board.N
        self.all_region_ids = sorted(initial_board.regions_map.keys())
        self.initial_regions_map_template = {k: {"cells": set(v["cells"]), "valid_placements": list(v["valid_placements"])} for k, v in initial_board.regions_map.items()}
        self.cell_to_region_id_map = initial_board.cell_to_region_id_map

        self.region_adjacencies = self._calculate_region_adjacencies(initial_board)
        if DEBUG_MODE:
            print(
                f"Region adjacencies precomputed: {self.region_adjacencies}",
                file=sys.stderr,
            )

        self.precomputed_future_region_cells_by_id = {}
        for rid, data in self.initial_regions_map_template.items():
            self.precomputed_future_region_cells_by_id[rid] = frozenset(data["cells"])

    def _calculate_region_adjacencies(self, board):
        """
        Precomputes a map of region_id -> set of adjacent region_ids.
        Used for the Degree Heuristic.
        """
        adjacencies = {rid: set() for rid in board.regions_map.keys()}

        for r in range(board.N):
            for c in range(board.N):
                current_rid = board.initial_grid_ids[r][c]
                for dr, dc in [
                    (0, 1),
                    (0, -1),
                    (1, 0),
                    (-1, 0),
                ]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < board.N and 0 <= nc < board.N:
                        neighbor_rid = board.initial_grid_ids[nr][nc]
                        if neighbor_rid != current_rid:
                            adjacencies[current_rid].add(neighbor_rid)
                            adjacencies[neighbor_rid].add(current_rid)

        return {rid: frozenset(adj_rids) for rid, adj_rids in adjacencies.items()}

    def _check_future_connectivity(self, board_state, new_placed_cells=frozenset()):
        """
        Heuristic: Checks if the currently filled cells PLUS the cells of ALL unassigned regions
        (and the potentially new placed cells for this action) form a single connected component.
        If not, it's impossible to form a connected final board.
        """
        hypothetical_filled = board_state.get_filled_cells() | new_placed_cells

        all_potential_filled_cells = set(hypothetical_filled)
        for rid in self.all_region_ids:
            if rid not in board_state.assignments:
                all_potential_filled_cells.update(
                    self.precomputed_future_region_cells_by_id[rid]
                )

        if not all_potential_filled_cells:
            return True if len(self.all_region_ids) == 0 else False

        start_cell = next(iter(all_potential_filled_cells))
        visited = set()
        queue = deque([start_cell])

        while queue:
            r, c = queue.popleft()
            if (r, c) in visited:
                continue
            visited.add((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in all_potential_filled_cells and (nr, nc) not in visited:
                    queue.append((nr, nc))

        if len(visited) != len(all_potential_filled_cells):
            if DEBUG_MODE:
                print(
                    f"    Future Connectivity PRUNED: Visited {len(visited)} of {len(all_potential_filled_cells)} total potential cells.",
                    file=sys.stderr,
                )
            return False
        return True

    def actions(self, state):
        board = state.board

        if not self._check_future_connectivity(board):
            if DEBUG_MODE:
                print(f"PRUNED STATE: Board connectivity impossible.", file=sys.stderr)
            return []

        unassigned_regions_info = []

        for rid in self.all_region_ids:
            if rid not in board.assignments:
                num_valid_placements = len(board.regions_map[rid]["valid_placements"])

                degree = 0
                for adj_rid in self.region_adjacencies.get(rid, frozenset()):
                    if adj_rid not in board.assignments:
                        degree += 1

                unassigned_regions_info.append((num_valid_placements, -degree, rid))

        if not unassigned_regions_info:
            return []
        

        unassigned_regions_info.sort(key=lambda x: (x[0], x[1], x[2]))
        target_region_id = unassigned_regions_info[0][2]

        if DEBUG_MODE:
            print(
                f"\n--- Generating actions for State {state.id}, Target Region (MRV+Deg): {target_region_id} ---",
                file=sys.stderr,
            )
            print(
                f"Current Assignments: {len(board.assignments)} regions assigned.",
                file=sys.stderr,
            )
            print(
                f"Valid Placements for Region {target_region_id}: {len(board.regions_map[target_region_id]['valid_placements'])} options (after MAC).",
                file=sys.stderr,
            )

        scored_actions = []

        unassigned_neighbors_of_target = [
            adj_rid
            for adj_rid in self.region_adjacencies.get(target_region_id, frozenset())
            if adj_rid not in board.assignments
            and adj_rid != target_region_id
        ]

        initial_regions_map_for_lcv_eval = self.initial_regions_map_template

        for tet_type, abs_cells_candidate in board.regions_map[target_region_id]["valid_placements"]:
            forbidden_overlap = len(abs_cells_candidate & board.get_forbidden_cells())
            touches_edge = any(r == 0 or r == board.N-1 or c == 0 or c == board.N-1 for r, c in abs_cells_candidate)
            scored_actions.append((forbidden_overlap + touches_edge, tet_type, abs_cells_candidate))

        scored_actions.sort(key=lambda x: (x[0], x[1], x[2], sorted(x[2])))

        actions = [
            (target_region_id, tet_type, abs_cells_candidate)
            for score, tet_type, abs_cells_candidate in scored_actions
        ]

        if DEBUG_MODE:
            print(
                f"Generated {len(actions)} actions for region {target_region_id} in state {state.id}",
                file=sys.stderr,
            )
            for score, tet_type, abs_cells_candidate in scored_actions:
                print(
                    f"  Action: {tet_type} at {sorted(list(abs_cells_candidate))}, LCV Score: {-score}",
                    file=sys.stderr,
                )

        return actions

    def result(self, state, action):
        """Returns the new state after applying an action."""
        region_id, tet_type, abs_cells = action

        new_assignments = dict(state.board.assignments)
        new_assignments[region_id] = {"type": tet_type, "abs_cells": abs_cells}

        new_filled_cells = set(state.board.get_filled_cells())
        new_filled_cells.update(abs_cells)

        new_board = Board(
            state.board.N,
            state.board.initial_grid_ids,
            state.board.regions_map,
            state.board.cell_to_region_id_map,
            new_assignments,
            set(state.board.get_forbidden_cells()),
            new_filled_cells,
        )
        new_board.mark_forbidden_2x2(
            abs_cells
        )

        propagation_queue = deque()
        for neighbor_rid in self.region_adjacencies.get(region_id, frozenset()):
            if neighbor_rid not in new_board.assignments:
                propagation_queue.append(neighbor_rid)

        while propagation_queue:
            rid_to_check = propagation_queue.popleft()

            old_domain_size = len(new_board.regions_map[rid_to_check]['valid_placements'])
            
            new_board.recalculate_valid_placements(rid_to_check)
            
            new_domain_size = len(new_board.regions_map[rid_to_check]['valid_placements'])

            if new_domain_size < old_domain_size and new_domain_size > 0:
                for neighbor_of_neighbor in self.region_adjacencies.get(rid_to_check, frozenset()):
                    if neighbor_of_neighbor not in new_board.assignments and neighbor_of_neighbor not in propagation_queue:
                            propagation_queue.append(neighbor_of_neighbor)

        return NuruominoState(new_board)

    def goal_test(self, state):
        """Checks if the current state is a goal state."""
        board = state.board

        if len(board.assignments) != len(self.all_region_ids):
            return False

        #print(board.print_board(), file=sys.stderr)
        #print("\n", file=sys.stderr)
     
        if _check_if_creates_2x2_block(self.N, board.get_filled_cells(), frozenset()):
            if DEBUG_MODE:
                print("Goal test failed: 2x2 block found.", file=sys.stderr)
            return False

        if not self._check_future_connectivity(board):
            if DEBUG_MODE:
                print(
                    "Goal test failed: final shaded cells not connected.",
                    file=sys.stderr,
                )
            return False

        if DEBUG_MODE:
            print(f"Goal test PASSED for state {state.id}!", file=sys.stderr)
        return True

    def h(self, node):
        board = node.state.board
        unassigned_count = 0
        min_placements = 1000 # Start with a large number

        for rid in self.all_region_ids:
            if rid not in board.assignments:
                unassigned_count += 1
                placements_count = len(board.regions_map[rid]['valid_placements'])
                
                if placements_count == 0:
                    return float('inf') # Dead end, prune immediately
                
                if placements_count < min_placements:
                    min_placements = placements_count
        
        if unassigned_count == 0:
            return 0
            
        # A simple, fast combination of progress (unassigned_count) and MRV (min_placements)
        # The weights prioritize finishing the puzzle above all else.
        return (unassigned_count * 100) + min_placements


def solve_nuruomino():
    DEBUG_MODE_COMPARISON = False
    expected_output_file = "test01.out"

    try:
        generate_all_tetromino_variants()
        if DEBUG_MODE:
            print("Tetromino variants generated.", file=sys.stderr)

        board = Board.parse_instance()
        if DEBUG_MODE:
            print(
                f"Input parsed: {board.N}x{board.N} grid, {len(board.regions_map)} regions.",
                file=sys.stderr,
            )
        problem = Nuruomino(board)
        if DEBUG_MODE:
            print("Nuruomino problem created.", file=sys.stderr)
            print("\nSolving with best-first graph search...", file=sys.stderr)

        goal_node = best_first_graph_search(
            problem, lambda n: problem.h(n), display=False
        )

        if goal_node:
            if DEBUG_MODE:
                print("\nSolution found!", file=sys.stderr)
            solution_str = goal_node.state.board.print_board()
            print(solution_str)

            if DEBUG_MODE_COMPARISON:
                try:
                    with open(expected_output_file, "r") as f:
                        expected_str = f.read().strip()
                    if solution_str.strip() == expected_str:
                        print("\n[SUCCESS] Output igual ao ficheiro esperado!")
                    else:
                        print("\n[FAIL] Output diferente do ficheiro esperado!")
                        print("----- Esperado -----")
                        print(expected_str)
                        print("----- Obtido -----")
                        print(solution_str)
                except Exception as e:
                    print(
                        f"[WARN] Não foi possível comparar com {expected_output_file}: {e}"
                    )

        else:
            if DEBUG_MODE:
                print("\nNo solution found.", file=sys.stderr)
            print("No solution.")

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)

    if DEBUG_MODE:
        print("\n--- End of Execution ---", file=sys.stderr)

DEBUG_MODE_CPROFILE = False
if __name__ == "__main__":
    if DEBUG_MODE_CPROFILE:
        import cProfile
        import pstats
        with cProfile.Profile() as pr:
            solve_nuruomino()
        stats = pstats.Stats(pr)
        stats.sort_stats("cumulative").print_stats(30)
    if not DEBUG_MODE_CPROFILE:
        solve_nuruomino()