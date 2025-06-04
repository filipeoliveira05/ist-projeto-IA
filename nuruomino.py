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
DEBUG_MODE_SIMPLE = False  # Simplified debug mode for less verbose output

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

    for name, (info) in TETROMINO_BASE_SHAPES.items():
        base_shape = info["shape"]
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


class Board:
    """Represents the Nuruomino game board, its regions, and current assignments."""

    def __init__(self, N, initial_grid_ids, regions_map, assignments=None):
        self.N = N
        self.initial_grid_ids = initial_grid_ids
        self.regions_map = regions_map
        self.assignments = assignments or {}
        self.solution_grid = [row[:] for row in initial_grid_ids]
        self.forbidden_cells = set()  # New attribute to track forbidden cells
        for region_id, data in self.assignments.items():
            for r, c in data["abs_cells"]:
                if 0 <= r < N and 0 <= c < N:
                    self.solution_grid[r][c] = data["type"]

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

    def get_filled_cells(self):
        """Returns a set of all currently filled cells on the board."""
        filled_cells = set()
        for data in self.assignments.values():
            filled_cells.update(data["abs_cells"])
        return filled_cells

    def mark_forbidden_2x2(self, placed_cells):
        """
        Identifies and marks cells that would complete a 2x2 block if filled.
        Adds these cells to the self.forbidden_cells set, using the _check_if_creates_2x2_block logic.

        Args:
            placed_cells: A frozenset of the cells that were just placed.
        """
        n = self.N
        filled_cells = self.get_filled_cells()

        forbidden_candidates = set()

        # Check empty neighbors of placed cells that would complete a 2x2
        for r_placed, c_placed in placed_cells:
            for dr, dc in [
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
            ]:  # Check orthogonal neighbors
                r_neighbor, c_neighbor = r_placed + dr, c_placed + dc
                if (
                    0 <= r_neighbor < n
                    and 0 <= c_neighbor < n
                    and (r_neighbor, c_neighbor) not in filled_cells
                ):
                    # Temporarily consider this neighbor as filled and check for 2x2 completion
                    temp_filled = filled_cells | {(r_neighbor, c_neighbor)}
                    if _check_if_creates_2x2_block(
                        n, filled_cells, {(r_neighbor, c_neighbor)}
                    ):
                        forbidden_candidates.add((r_neighbor, c_neighbor))

        # Add the forbidden candidates to the set of forbidden cells
        for r_forbid, c_forbid in forbidden_candidates:
            if (r_forbid, c_forbid) not in self.get_filled_cells():
                self.forbidden_cells.add((r_forbid, c_forbid))
        if DEBUG_MODE:
            print(f"MarkForbidden - Placed cells: {placed_cells}")
            print(f"MarkForbidden - Filled cells: {self.get_filled_cells()}")
            print(f"MarkForbidden - Forbidden candidates: {forbidden_candidates}")
            print(
                f"MarkForbidden - Forbidden cells after check: {self.forbidden_cells}"
            )

    def print_board(self):
        """Imprime a grelha (solution_grid) mostrando 'X' para células proibidas."""
        output = ""
        for r in range(self.N):
            row_str = []
            for c in range(self.N):
                # if (r, c) in self.forbidden_cells:
                #     row_str.append("X")

                row_str.append(self.solution_grid[r][c])
            output += "\t".join(row_str) + "\n"
        return output.strip()

    def recalculate_valid_placements(self, region_id):
        """Recalculates the valid tetromino placements for a given region,
        considering the currently filled and forbidden cells."""
        if region_id not in self.regions_map:
            return []

        region_cells = self.regions_map[region_id]["cells"]
        valid_placements = set()
        occupied_cells = self.get_filled_cells() | self.forbidden_cells

        for tet_type in ALLOWED_TETROMINO_TYPES:
            for variant_coords_list in TETROMINO_VARIANTS[tet_type]:
                variant_coords_frozen = frozenset(variant_coords_list)

                for vr_anchor, vc_anchor in variant_coords_frozen:
                    for r_region_cell, c_region_cell in region_cells:
                        offset_r = r_region_cell - vr_anchor
                        offset_c = c_region_cell - vc_anchor

                        current_placement_abs_cells = frozenset(
                            [
                                (vr + offset_r, vc + offset_c)
                                for vr, vc in variant_coords_frozen
                            ]
                        )

                        # Check if the placement is within the region's original bounds
                        if current_placement_abs_cells.issubset(region_cells):
                            # Check if any cell in the placement overlaps with already occupied or forbidden cells
                            if not (current_placement_abs_cells & occupied_cells):
                                if len(current_placement_abs_cells) == 4:
                                    valid_placements.add(
                                        (tet_type, current_placement_abs_cells)
                                    )

        self.regions_map[region_id]["valid_placements"] = list(valid_placements)
        if DEBUG_MODE:
            print(
                f"Region {region_id}: Recalculated {len(valid_placements)} valid placements.",
                file=sys.stderr,
            )
        return list(valid_placements)


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
        Corrected adjacency check: checks adjacency between the placed piece and existing pieces of the same type.
        """
        board = state.board
        current_filled_cells = board.get_filled_cells()

        if _check_if_creates_2x2_block(self.N, current_filled_cells, frozenset()):
            if DEBUG_MODE:
                print(
                    f"PRUNED STATE (2x2 violation): State {state.id}",
                    file=sys.stderr,
                )
            return []

        unassigned_regions_info = []
        for rid in self.all_region_ids:
            if rid not in board.assignments:
                num_valid_placements = len(board.regions_map[rid]["valid_placements"])
                unassigned_regions_info.append((num_valid_placements, rid))

        if not unassigned_regions_info:
            return []

        unassigned_regions_info.sort()
        target_region_id = unassigned_regions_info[0][1]
        if DEBUG_MODE:
            print(
                f"\n--- Generating actions for State {state.id}, Target Region: {target_region_id} ---",
                file=sys.stderr,
            )
            print(f"Current Assignments: {board.assignments}", file=sys.stderr)
            print(
                f"Valid Placements for Region {target_region_id}: {board.regions_map[target_region_id]['valid_placements']}",
                file=sys.stderr,
            )

        actions = []
        for tet_type, abs_cells_candidate in board.regions_map[target_region_id][
            "valid_placements"
        ]:
            if DEBUG_MODE:
                print(
                    f"\nTrying placement: {tet_type} at {sorted(list(abs_cells_candidate))}",
                    file=sys.stderr,
                )

            # Check for orthogonal adjacency with already placed pieces of the same type
            violates_adjacency = False
            for r_candidate, c_candidate in abs_cells_candidate:
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = r_candidate + dr, c_candidate + dc
                    for assigned_rid, assigned_data in board.assignments.items():
                        if (
                            assigned_rid != target_region_id
                            and assigned_data["type"] == tet_type
                            and (nr, nc) in assigned_data["abs_cells"]
                        ):
                            violates_adjacency = True
                            if DEBUG_MODE:
                                print(
                                    f"    Orthogonal Adjacency Violation with existing {tet_type} at ({nr},{nc})",
                                    file=sys.stderr,
                                )
                            break  # Break inner loop (neighbor check)
                    if violates_adjacency:
                        break  # Break outer loop (candidate cell check)
                if violates_adjacency:
                    break  # Break the loop over candidate cells

            if violates_adjacency:
                continue

            # 2x2 Block Check
            creates_2x2 = _check_if_creates_2x2_block(
                self.N, current_filled_cells, abs_cells_candidate
            )
            if creates_2x2:
                if DEBUG_MODE:
                    print("    Creates a 2x2 block", file=sys.stderr)
                continue

            actions.append((target_region_id, tet_type, abs_cells_candidate))
            if DEBUG_MODE:
                print(
                    f"    Action Generated: {tet_type} at {sorted(list(abs_cells_candidate))}",
                    file=sys.stderr,
                )

        if DEBUG_MODE:
            print(
                f"Generated {len(actions)} actions for region {target_region_id} in state {state.id}",
                file=sys.stderr,
            )

        if DEBUG_MODE_SIMPLE:
            print(board.print_board(), file=sys.stderr)
            print("\n")

        return actions

    def result(self, state, action):
        """Returns the new state after applying an action."""
        region_id, tet_type, abs_cells = action
        new_assignments = dict(state.board.assignments)
        new_assignments[region_id] = {"type": tet_type, "abs_cells": abs_cells}

        # Create a new Board instance with the updated assignments
        new_board = Board(
            state.board.N,
            state.board.initial_grid_ids,  # Pass initial_grid_ids to new Board
            state.board.regions_map,  # regions_map is immutable, can be shared
            new_assignments,
        )
        new_board.forbidden_cells = set(state.board.forbidden_cells)
        new_board.mark_forbidden_2x2(abs_cells)
        return NuruominoState(new_board)

    def goal_test(self, state):
        """Checks if the current state is a goal state."""
        board = state.board

        # 1. All regions must be assigned
        if len(board.assignments) != len(self.all_region_ids):
            if DEBUG_MODE:
                print(
                    f"Goal test failed: {len(board.assignments)} regions assigned, expected {len(self.all_region_ids)}.",
                    file=sys.stderr,
                )

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
        board = node.state.board
        h_value = 0
        unassigned_regions = [
            rid for rid in self.all_region_ids if rid not in board.assignments
        ]
        num_assigned = len(board.assignments)
        total_regions = len(self.all_region_ids)
        num_forbidden_cells = len(board.forbidden_cells)

        if not unassigned_regions:
            return 0

        min_valid_placements = float("inf")
        for rid in unassigned_regions:
            num_valid = len(board.regions_map[rid]["valid_placements"])
            min_valid_placements = min(min_valid_placements, num_valid)
            if DEBUG_MODE:
                print(
                    f"Heuristic Debug - Region: {rid}, Valid Placements: {num_valid}",
                    file=sys.stderr,
                )

        # Component 1: Prioritize constrained regions (MRV heuristic)
        constraint_priority = 0
        if min_valid_placements > 0 and min_valid_placements != float("inf"):
            constraint_priority = min_valid_placements
        elif min_valid_placements == 0:
            return float("inf")  # Dead end

        # Component 2: Reward progress (more assigned pieces)
        progress_reward = total_regions - num_assigned

        # Component 3: Normalized forbidden cell count
        normalized_forbidden = 0
        if num_assigned > 0:
            normalized_forbidden = num_forbidden_cells / num_assigned
        elif num_forbidden_cells > 0:
            normalized_forbidden = (
                num_forbidden_cells  # If no pieces, just use the count
            )

        # We want to prioritize states with a *higher* density of forbidden cells,
        # so we use a positive weight.
        forbidden_density_priority = -normalized_forbidden * 1.0

        # Piece type prioritization with different weights
        piece_type_bonus = 0
        piece_weights = {
            "I": 0.4,
            "T": 0.3,
            "S": 0.2,
            "L": 0.1,
        }  # Higher weight = higher priority

        for region_id, assignment in board.assignments.items():
            piece_type = assignment["type"]
            piece_type_bonus += piece_weights.get(piece_type, 0)

        # We want to prioritize states with a higher weighted sum of placed pieces,
        # so we subtract this bonus (because best-first search minimizes the heuristic).
        piece_priority_heuristic = -piece_type_bonus * 1.0  # Adjust the overall weight

        # Combine the components with adjusted weights
        weight_constraint = 0.1
        weight_progress = 1
        weight_forbidden_impact = 0.5  # Increase weight for forbidden impact
        weight_piece_priority = 0.1  # Adjusted weight for piece type priority

        h_value = (
            (weight_constraint * constraint_priority)
            + (weight_progress * progress_reward)
            + (weight_forbidden_impact * forbidden_density_priority)
            + (weight_piece_priority * piece_priority_heuristic)
        )

        if DEBUG_MODE:
            print(
                f"Heuristic for state {node.state.id}: min_valid={min_valid_placements}, assigned={num_assigned}, forbidden={num_forbidden_cells}, forbidden_impact={forbidden_impact:.2f}, h={h_value}",
                file=sys.stderr,
            )

        return h_value


def solve_nuruomino():
    DEBUG_MODE_COMPARISON = False
    expected_output_file = "test15.out"
    # expected_output_file = "../132/sample-nuruominoboards/test-03.out"

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


if __name__ == "__main__":
    solve_nuruomino()
