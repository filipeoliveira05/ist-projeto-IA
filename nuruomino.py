# nuruomino.py: Projeto de Inteligência Artificial 2024/2025
# Grupo 00:
# 110633 Filipe Oliveira
# 110720 Francisco Andrade

import sys
from search import Problem, Node, best_first_graph_search
from copy import deepcopy
from collections import deque  # For BFS in connectivity check

# Global debug flag - Set to False for production to suppress stderr output
DEBUG_MODE = False

# --- Global Definitions ---
TETROMINO_BASE_SHAPES = {
    "L": frozenset([(0, 0), (1, 0), (2, 0), (2, 1)]),  # 3x2 L-shape
    "I": frozenset([(0, 0), (1, 0), (2, 0), (3, 0)]),  # 4x1 line
    "T": frozenset([(0, 0), (0, 1), (0, 2), (1, 1)]),  # 2x3 T-shape
    "S": frozenset(
        [(0, 1), (1, 1), (1, 0), (2, 0)]
    ),  # 3x2 S-shape (also Z-shape, depending on orientation)
}
ALLOWED_TETROMINO_TYPES = ["L", "I", "T", "S"]
TETROMINO_VARIANTS = {}  # Stores all rotated/reflected variants for each type


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
    # Normalize after rotation to ensure consistency for comparison in the set
    return _normalize_shape(frozenset([(c, -r) for r, c in shape_coords]))


def _reflect_shape_vertical_axis(shape_coords):
    """
    Reflects a shape across the vertical axis (c -> -c).
    Normalization is applied afterwards.
    """
    # Normalize after reflection to ensure consistency for comparison in the set
    return _normalize_shape(frozenset([(r, -c) for r, c in shape_coords]))


def generate_all_tetromino_variants():
    global TETROMINO_VARIANTS
    if TETROMINO_VARIANTS:
        return

    for name, base_shape in TETROMINO_BASE_SHAPES.items():
        variants = set()
        current_shape_for_rotation = base_shape

        for i in range(4):
            normalized_rotated = _normalize_shape(current_shape_for_rotation)
            variants.add(normalized_rotated)
            reflected = _reflect_shape_vertical_axis(current_shape_for_rotation)
            variants.add(_normalize_shape(reflected))

            if DEBUG_MODE:
                print(
                    f"Generated (rotation {i}, {name}): {sorted(list(normalized_rotated))}"
                )
                print(
                    f"Generated (reflection of rotation {i}, {name}): {sorted(list(_normalize_shape(reflected)))}"
                )
                if len(normalized_rotated) != 4:
                    print(f"ERROR: {name} variant has {len(normalized_rotated)} cells!")
                if len(_normalize_shape(reflected)) != 4:
                    print(
                        f"ERROR: Reflected {name} variant has {len(_normalize_shape(reflected))} cells!"
                    )

            current_shape_for_rotation = _rotate_shape_90_clockwise(
                current_shape_for_rotation
            )

        TETROMINO_VARIANTS[name] = [list(variant) for variant in variants]
        if DEBUG_MODE:
            print(f"Generated {len(TETROMINO_VARIANTS[name])} variants for {name}")


def _check_if_creates_2x2_block(N, existing_filled_cells, new_cells_to_add):
    """
    Checks if adding 'new_cells_to_add' to 'existing_filled_cells' creates a 2x2 block.
    N is the board size.
    """
    all_filled = existing_filled_cells | new_cells_to_add

    # Iterate only over possible top-left corners that include a newly added cell
    # This optimization reduces checks significantly.
    for r_new, c_new in new_cells_to_add:
        # Check all 4 possible 2x2 block origins that involve (r_new, c_new)
        # Block 1: (r_new, c_new) is top-left
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
        # Block 2: (r_new, c_new) is top-right
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
        # Block 3: (r_new, c_new) is bottom-left
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
        # Block 4: (r_new, c_new) is bottom-right
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

    state_id_counter = (
        0  # Unique ID for each state instance (useful for debugging and __lt__)
    )

    def __init__(self, board):
        self.board = board
        self.id = NuruominoState.state_id_counter
        NuruominoState.state_id_counter += 1

    def __lt__(self, other):
        # Used by PriorityQueue for tie-breaking when costs are equal
        return self.id < other.id

    def __eq__(self, other):
        return isinstance(other, NuruominoState) and self.board == other.board

    def __hash__(self):
        return hash(self.board)


class Board:
    """Represents the Nuruomino game board, its regions, and current assignments."""

    def __init__(self, N, initial_grid_ids, regions_map, assignments=None):
        self.N = N
        self.initial_grid_ids = initial_grid_ids  # N x N list of strings (region IDs)
        self.regions_map = (
            regions_map  # Dict: region_id -> {cells: set, valid_placements: list}
        )
        self.assignments = (
            assignments or {}
        )  # Dict: region_id -> {type: str, abs_cells: frozenset}

        # The solution grid for printing (populated dynamically)
        self.solution_grid = [row[:] for row in initial_grid_ids]
        for region_id, data in self.assignments.items():
            for r, c in data["abs_cells"]:
                if 0 <= r < N and 0 <= c < N:
                    self.solution_grid[r][c] = data["type"]  # Fill with tetromino type

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

        initial_grid_ids = lines  # Stores the original region IDs

        regions_map = {}
        for r in range(N):
            for c in range(N):
                rid = initial_grid_ids[r][c]
                if rid not in regions_map:
                    regions_map[rid] = {"cells": set(), "valid_placements": []}
                regions_map[rid]["cells"].add((r, c))

        # Ensure variants are generated before calculating placements
        generate_all_tetromino_variants()

        # Pre-calculate all valid placements for each region
        for rid, data in regions_map.items():
            cells = data["cells"]  # set of (r,c) cells for this region

            valid_placements = set()  # Use a set to avoid duplicates

            # Iterate through all allowed tetromino types
            for tet_type in ALLOWED_TETROMINO_TYPES:
                # For each type, iterate through all its generated variants (rotations/reflections)
                for variant_coords_list in TETROMINO_VARIANTS[
                    tet_type
                ]:  # variant_coords_list is a list of (r,c) tuples
                    variant_coords_frozen = frozenset(
                        variant_coords_list
                    )  # Use frozenset for consistent hashing/comparison

                    # For each cell in the current 'variant_coords_frozen' tetromino shape,
                    # try to align it with every cell in the current 'region_cells'.
                    for (
                        vr_anchor,
                        vc_anchor,
                    ) in variant_coords_frozen:  # A cell from the tetromino variant
                        for (
                            r_region_cell,
                            c_region_cell,
                        ) in cells:  # A cell from the board region

                            # Calculate the offset needed to move the 'vr_anchor, vc_anchor' of the variant
                            # to the 'r_region_cell, c_region_cell' of the board.
                            offset_r = r_region_cell - vr_anchor
                            offset_c = c_region_cell - vc_anchor

                            # Calculate the absolute coordinates of all cells of this variant, shifted by the offset
                            current_placement_abs_cells = frozenset(
                                [
                                    (vr + offset_r, vc + offset_c)
                                    for vr, vc in variant_coords_frozen
                                ]
                            )

                            # Check if all cells of this shifted variant are contained within the current region
                            if current_placement_abs_cells.issubset(cells):
                                # It's a valid placement if it fits perfectly
                                if len(current_placement_abs_cells) == 4:
                                    valid_placements.add(
                                        (tet_type, current_placement_abs_cells)
                                    )

            data["valid_placements"] = list(
                valid_placements
            )  # Convert back to list for consistency
            if DEBUG_MODE:
                print(
                    f"Region {rid}: {len(data['valid_placements'])} valid tetromino placements (pre-calculated)",
                    file=sys.stderr,
                )
            if not data["valid_placements"]:
                print(
                    f"WARNING: Region {rid} has no valid placements possible! This puzzle is likely unsolvable.",
                    file=sys.stderr,
                )

        return Board(N, initial_grid_ids, regions_map, {})

    def print_board(self):
        """Imprime a grelha (solution_grid) no formato de output especificado."""
        output = ""
        for r in range(self.N):
            output += "\t".join(self.solution_grid[r]) + "\n"
        return output.strip()  # Remove a última nova linha

    def __eq__(self, other):
        if not isinstance(other, Board):
            return False
        # Only compare assignments for state equality. N and initial_grid_ids are static for a problem.
        return self.assignments == other.assignments

    def __hash__(self):
        # Hash based on the current assignments to ensure unique states
        assignments_tuple = tuple(
            (rid, data["type"], data["abs_cells"])
            for rid, data in sorted(self.assignments.items())
        )
        # Include N and initial_grid_ids for a full problem hash, though not strictly needed for state comparison
        grid_tuple = tuple(tuple(row) for row in self.initial_grid_ids)
        return hash((self.N, grid_tuple, assignments_tuple))

    def count_almost_2x2_blocks_internal(self):
        """Counts 3-cell almost-2x2 blocks within the current filled cells."""
        filled = set()
        for data in self.assignments.values():
            filled.update(data["abs_cells"])

        count = 0
        # Iterate only over possible top-left corners of 2x2 blocks
        for r in range(self.N - 1):
            for c in range(self.N - 1):
                block_cells = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
                filled_in_block = sum(1 for cell in block_cells if cell in filled)
                if filled_in_block == 3:  # Check for 3 cells
                    count += 1
        return count

    def get_filled_cells(self):
        """Returns a set of all currently filled cells on the board."""
        filled_cells = set()
        for data in self.assignments.values():
            filled_cells.update(data["abs_cells"])
        return filled_cells


class Nuruomino(Problem):
    """
    Represents the Nuruomino puzzle as a search problem.
    Uses Best-First Graph Search with a heuristic.
    """

    def __init__(self, initial_board):
        super().__init__(NuruominoState(initial_board))
        self.all_region_ids = sorted(initial_board.regions_map.keys())
        self.N = initial_board.N

    def actions(self, state):
        """
        Generates valid actions (placing a tetromino) from the current state.
        Actions are (region_id, tet_type, abs_cells).
        Uses MRV (Minimum Remaining Values) heuristic to select which region to assign next.
        """
        board = state.board
        current_filled_cells = board.get_filled_cells()

        # --- Pruning 1: Check if the current board state already has an invalid 2x2 block ---
        if _check_if_creates_2x2_block(self.N, current_filled_cells, frozenset()):
            if DEBUG_MODE:
                print(
                    f"PRUNED STATE (2x2 violation): State {state.id} already contains an invalid 2x2 block.",
                    file=sys.stderr,
                )
            return []  # No actions possible from an invalid state

        actions = []
        unassigned_regions_info = []

        for rid in self.all_region_ids:
            if rid not in board.assignments:  # Process only unassigned regions
                num_valid_placements_for_rid = 0
                if DEBUG_MODE:
                    print(
                        f"Checking valid placements for region {rid}...",
                        file=sys.stderr,
                    )
                for tet_type_candidate, abs_cells_candidate in board.regions_map[rid][
                    "valid_placements"
                ]:
                    if DEBUG_MODE:
                        print(
                            f"  Trying {tet_type_candidate} at {sorted(list(abs_cells_candidate))}",
                            file=sys.stderr,
                        )
                        if len(abs_cells_candidate) != 4:
                            print(
                                f"    ERROR: Candidate has {len(abs_cells_candidate)} cells!",
                                file=sys.stderr,
                            )

                    # 1. Adjacency check (similar to goal_test)
                    violates_adj = False
                    for r_candidate, c_candidate in abs_cells_candidate:
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = r_candidate + dr, c_candidate + dc
                            if 0 <= nr < self.N and 0 <= nc < self.N:
                                n_rid_neighbor = board.initial_grid_ids[nr][nc]
                                if n_rid_neighbor in board.assignments:
                                    neighbor_type = board.assignments[n_rid_neighbor][
                                        "type"
                                    ]
                                    if neighbor_type == tet_type_candidate:
                                        violates_adj = True
                                        if DEBUG_MODE:
                                            print(
                                                f"    Violates adjacency with region {n_rid_neighbor} of type {neighbor_type} at neighbor ({nr},{nc}) of candidate cell ({r_candidate},{c_candidate})",
                                                file=sys.stderr,
                                            )
                                        break
                        if violates_adj:
                            break
                    if violates_adj:
                        continue

                    # 2. 2x2 block creation with current filled cells
                    if _check_if_creates_2x2_block(
                        self.N, current_filled_cells, abs_cells_candidate
                    ):
                        if DEBUG_MODE:
                            print("    Creates a 2x2 block", file=sys.stderr)
                        continue

                    num_valid_placements_for_rid += 1
                    if DEBUG_MODE:
                        print("    Valid placement!", file=sys.stderr)

                if DEBUG_MODE:
                    print(
                        f"Region {rid} has {num_valid_placements_for_rid} valid placements in the current state.",
                        file=sys.stderr,
                    )

                unassigned_regions_info.append((num_valid_placements_for_rid, rid))

        if not unassigned_regions_info:  # All regions assigned, no actions possible
            return []

        # Sort by number of valid placements (MRV). Pick the region with fewest options.
        unassigned_regions_info.sort()
        target_region_id = unassigned_regions_info[0][1]
        if DEBUG_MODE:
            print(f"Selected target region: {target_region_id} (MRV)", file=sys.stderr)

        filtered_candidate_placements = []
        for tet_type, abs_cells in board.regions_map[target_region_id][
            "valid_placements"
        ]:
            if DEBUG_MODE:
                print(
                    f"Considering placement for target {target_region_id}: {tet_type}, {sorted(list(abs_cells))}"
                )
                if len(abs_cells) != 4:
                    print(
                        f"    ERROR: Candidate has {len(abs_cells)} cells!",
                        file=sys.stderr,
                    )

            # Re-check validity for the target region based on the current state
            violates_adjacency_target = False
            for r_candidate, c_candidate in abs_cells:
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r_candidate + dr, c_candidate + dc
                    if 0 <= nr < self.N and 0 <= nc < self.N:
                        n_rid_neighbor = board.initial_grid_ids[nr][nc]
                        if (
                            n_rid_neighbor in board.assignments
                            and n_rid_neighbor != target_region_id
                        ):
                            neighbor_type = board.assignments[n_rid_neighbor]["type"]
                            if neighbor_type == tet_type:
                                violates_adjacency_target = True
                                if DEBUG_MODE:
                                    print(
                                        f"    Violates adjacency with region {n_rid_neighbor} of type {neighbor_type} at neighbor ({nr},{nc})",
                                        file=sys.stderr,
                                    )
                                break
                if violates_adjacency_target:
                    break
            if violates_adjacency_target:
                continue

            # 2x2 block creation
            if _check_if_creates_2x2_block(self.N, current_filled_cells, abs_cells):
                if DEBUG_MODE:
                    print("    Creates a 2x2 block", file=sys.stderr)
                continue

            filtered_candidate_placements.append((tet_type, abs_cells))

        if DEBUG_MODE and not filtered_candidate_placements:
            print(
                f"DEBUG: No valid filtered placements for target region {target_region_id}.",
                file=sys.stderr,
            )

        for tet_type, abs_cells in filtered_candidate_placements:
            actions.append((target_region_id, tet_type, abs_cells))
            if DEBUG_MODE:
                print(
                    f"Generated action: Place {tet_type} at {sorted(list(abs_cells))} in region {target_region_id}",
                    file=sys.stderr,
                )

        if DEBUG_MODE:
            print(
                f"Generated {len(actions)} actions for state {state.id} (target region {target_region_id})",
                file=sys.stderr,
            )

        if DEBUG_MODE:
            print(board.print_board(), file=sys.stderr)

        return actions

    def result(self, state, action):
        """Returns the new state after applying an action."""
        region_id, tet_type, abs_cells = action
        new_assignments = deepcopy(
            state.board.assignments
        )  # Deep copy to avoid modifying original state
        new_assignments[region_id] = {"type": tet_type, "abs_cells": abs_cells}

        # Create a new Board instance with the updated assignments
        new_board = Board(
            state.board.N,
            state.board.initial_grid_ids,  # Pass initial_grid_ids to new Board
            state.board.regions_map,  # regions_map is immutable, can be shared
            new_assignments,
        )
        return NuruominoState(new_board)

    def goal_test(self, state):
        """Checks if the current state is a goal state."""
        board = state.board

        # 1. All regions must be assigned
        if len(board.assignments) != len(self.all_region_ids):
            return False

        # 2. No 2x2 blocks of filled cells
        filled_cells = board.get_filled_cells()
        if _check_if_creates_2x2_block(
            self.N, filled_cells, frozenset()
        ):  # Check existing blocks without adding new_cells
            if DEBUG_MODE:
                print("Goal test failed: 2x2 block found.", file=sys.stderr)
            return False

        # 3. All shaded cells must be connected
        if (
            not filled_cells
        ):  # If no cells are filled (empty board, 0 regions), it's connected IF there are no regions
            return True if len(self.all_region_ids) == 0 else False

        # Perform a BFS/DFS to check 4-connectivity of filled cells
        start_cell = next(iter(filled_cells))  # Pick an arbitrary starting cell
        visited = set()
        queue = deque([start_cell])  # Use deque for BFS

        while queue:
            r, c = queue.popleft()  # Pop from left for BFS
            if (r, c) in visited:
                continue
            visited.add((r, c))

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # 4-connectivity
                nr, nc = r + dr, c + dc
                if (nr, nc) in filled_cells and (nr, nc) not in visited:
                    queue.append((nr, nc))

        if len(visited) != len(filled_cells):
            if DEBUG_MODE:
                print(
                    f"Goal test failed: shaded cells not connected. Visited {len(visited)} of {len(filled_cells)}.",
                    file=sys.stderr,
                )
            return False

        return True  # All conditions met

    def h(self, node):
        """
        Heuristic function for the Nuruomino problem.
        Estimates the cost from the current state to the goal.
        This heuristic aims to be admissible or at least highly informative for best-first search.
        """
        board = node.state.board

        # H1: Number of unassigned regions (admissible, counts remaining tasks)
        unassigned_regions = [
            rid for rid in self.all_region_ids if rid not in board.assignments
        ]
        h_unassigned = len(unassigned_regions)

        # If all regions are assigned, and goal_test might pass, heuristic cost is 0
        if h_unassigned == 0:
            return 0

        # H2: Check for immediate unfillable regions (returning a large finite number instead of infinity)
        current_filled_cells = board.get_filled_cells()
        for rid in unassigned_regions:
            has_valid_placement_found_in_h = False
            for tet_type_candidate, abs_cells_candidate in board.regions_map[rid][
                "valid_placements"
            ]:
                # 1. Adjacency check (similar to goal_test)
                violates_adj = False
                for r_candidate, c_candidate in abs_cells_candidate:
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r_candidate + dr, c_candidate + dc
                        if 0 <= nr < self.N and 0 <= nc < self.N:
                            n_rid_neighbor = board.initial_grid_ids[nr][nc]
                            if n_rid_neighbor in board.assignments:
                                neighbor_type = board.assignments[n_rid_neighbor][
                                    "type"
                                ]
                                if neighbor_type == tet_type_candidate:
                                    violates_adj = True
                                    break
                    if violates_adj:
                        break
                if violates_adj:
                    continue

                # 2. 2x2 block creation
                if _check_if_creates_2x2_block(
                    self.N, current_filled_cells, abs_cells_candidate
                ):
                    continue

                has_valid_placement_found_in_h = True
                break

            if not has_valid_placement_found_in_h:
                if DEBUG_MODE:
                    print(
                        f"HEURISTIC WARNING: Region {rid} has 0 valid placements!",
                        file=sys.stderr,
                    )
                return float(
                    len(self.all_region_ids) * 100
                )  # Return a large finite number

        # H3: Connectivity Heuristic (number of disjoint shaded components)
        h_connectivity = 0
        filled_cells = current_filled_cells
        if filled_cells:
            num_components = 0
            visited_connectivity = set()

            for r_start, c_start in filled_cells:
                if (r_start, c_start) not in visited_connectivity:
                    num_components += 1
                    q_bfs = deque([(r_start, c_start)])
                    visited_connectivity.add((r_start, c_start))

                    while q_bfs:
                        r, c = q_bfs.popleft()
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if (nr, nc) in filled_cells and (
                                nr,
                                nc,
                            ) not in visited_connectivity:
                                visited_connectivity.add((nr, nc))
                                q_bfs.append((nr, nc))

            if num_components > 1:
                h_connectivity = num_components - 1

        h_value = h_unassigned + h_connectivity

        if DEBUG_MODE:
            print(
                f"Heuristic for state {node.state.id}: unassigned={h_unassigned}, connectivity={h_connectivity}, total={h_value}",
                file=sys.stderr,
            )
        return h_value


def solve_nuruomino():
    DEBUG_MODE_COMPARISON = True
    expected_output_file = "test06.out"
    
    try:
        generate_all_tetromino_variants()
        if DEBUG_MODE:
            print("Tetromino variants generated.", file=sys.stderr)

        board = (
            Board.parse_instance()
        )  # This calls generate_all_tetromino_variants internally now
        if DEBUG_MODE:
            print(
                f"Input parsed: {board.N}x{board.N} grid, {len(board.regions_map)} regions.",
                file=sys.stderr,
            )
        problem = Nuruomino(board)
        if DEBUG_MODE:
            print("Nuruomino problem created.", file=sys.stderr)
            print("\nSolving with best-first graph search...", file=sys.stderr)

        # Pass the heuristic function (h) to best_first_graph_search
        goal_node = best_first_graph_search(
            problem, lambda n: problem.h(n), display=False
        )

        if goal_node:
            if DEBUG_MODE:
                print("\nSolution found!", file=sys.stderr)
            solution_str = goal_node.state.board.print_board()
            print(solution_str)

            # --- Comparação com ficheiro esperado ---
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
                    print(f"[WARN] Não foi possível comparar com {expected_output_file}: {e}")

        else:
            if DEBUG_MODE:
                print("\nNo solution found.", file=sys.stderr)
            print("No solution.")

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)

    if DEBUG_MODE:
        print("\n--- End of Execution ---", file=sys.stderr)


if __name__ == "__main__":
    solve_nuruomino()