# nuruomino.py: Projeto de Inteligência Artificial 2024/2025
# Grupo 132:
# 110633 Filipe Oliveira
# 110720 Francisco Andrade

import sys
import heapq
from collections import deque
import copy

DEBUG_MODE = False

TETROMINO_BASE_SHAPES = {
    "L": frozenset([(0, 0), (1, 0), (2, 0), (2, 1)]),
    "I": frozenset([(0, 0), (1, 0), (2, 0), (3, 0)]),
    "T": frozenset([(0, 0), (0, 1), (0, 2), (1, 1)]),
    "S": frozenset([(0, 1), (1, 1), (1, 0), (2, 0)]),
}
ALLOWED_TETROMINO_TYPES = TETROMINO_BASE_SHAPES.keys()
TETROMINO_VARIANTS = {}


def _normalize_shape(shape_cells):
    """
    Normalizes a tetromino shape by shifting its cells so that the top-leftmost cell
    (minimum row, then minimum column) is is at (0,0).
    Ensures consistent representation for shape comparison regardless of absolute position.
    """
    if not shape_cells:
        return frozenset()
    min_r = min(r for r, c in shape_cells)
    min_c = min(c for r, c in shape_cells)
    return frozenset([(r - min_r, c - min_c) for r, c in shape_cells])


def _rotate_shape_90_clockwise(shape_cells):
    """Rotates a shape 90 degrees clockwise."""
    return _normalize_shape(frozenset([(c, -r) for r, c in shape_cells]))


def _reflect_shape_vertical_axis(shape_cells):
    """Reflects a shape across the vertical axis."""
    return _normalize_shape(frozenset([(r, -c) for r, c in shape_cells]))


def generate_all_tetromino_variants():
    """Generates all unique rotations and reflections for each tetromino type."""
    global TETROMINO_VARIANTS
    if TETROMINO_VARIANTS:
        return

    for tet_type, base_shape in TETROMINO_BASE_SHAPES.items():
        variants = set()
        current_shape = base_shape

        for _ in range(4):
            variants.add(current_shape)
            current_shape = _rotate_shape_90_clockwise(current_shape)

        current_shape_reflected = _reflect_shape_vertical_axis(base_shape)
        for _ in range(4):
            variants.add(current_shape_reflected)
            current_shape_reflected = _rotate_shape_90_clockwise(
                current_shape_reflected
            )

        TETROMINO_VARIANTS[tet_type] = sorted(
            list(variants),
            key=lambda s: (min(x[0] for x in s), min(x[1] for x in s), frozenset(s)),
        )

    if DEBUG_MODE:
        total_variants = sum(len(v) for v in TETROMINO_VARIANTS.values())
        print(f"Generated {total_variants} total tetromino variants.", file=sys.stderr)


generate_all_tetromino_variants()


# --- Helper functions for constraint checks (static, take state as args) ---


def _check_if_creates_2x2_block(N, existing_filled_cells, new_cells_to_add):
    """
    Checks if adding `new_cells_to_add` to `existing_filled_cells` creates any 2x2 block.
    Optimized to only check 2x2 blocks where a new cell is one of the four corners.
    """
    all_filled = existing_filled_cells | new_cells_to_add

    for r_new, c_new in new_cells_to_add:
        for dr, dc in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
            r_tl, c_tl = r_new + dr, c_new + dc

            if not (0 <= r_tl < N - 1 and 0 <= c_tl < N - 1):
                continue

            block_cells = frozenset(
                [(r_tl, c_tl), (r_tl + 1, c_tl), (r_tl, c_tl + 1), (r_tl + 1, c_tl + 1)]
            )

            if block_cells.issubset(all_filled):
                return True
    return False


def _mark_forbidden_2x2_static(
    N, filled_cells, current_forbidden_cells, new_piece_abs_cells
):
    """
    Calculates *new* forbidden cells based on newly filled cells and existing ones.
    Cells become forbidden if they would complete a 2x2 block with 3 already filled cells.
    Returns a new frozenset of forbidden cells.
    """
    temp_forbidden_set = set(current_forbidden_cells)

    temp_forbidden_set -= new_piece_abs_cells

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
            if 0 <= nr < N and 0 <= nc < N:
                cells_to_scan.add((nr, nc))

    for r_target, c_target in cells_to_scan:
        for dr_offset, dc_offset in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
            r_top_left, c_top_left = r_target + dr_offset, c_target + dc_offset

            if 0 <= r_top_left < N - 1 and 0 <= c_top_left < N - 1:
                block_cells = frozenset(
                    [
                        (r_top_left, c_top_left),
                        (r_top_left + 1, c_top_left),
                        (r_top_left, c_top_left + 1),
                        (r_top_left + 1, c_top_left + 1),
                    ]
                )

                filled_in_block = block_cells.intersection(filled_cells)

                if len(filled_in_block) == 3:
                    empty_cells_in_block = block_cells - filled_in_block
                    forbidden_cell = next(iter(empty_cells_in_block))

                    if forbidden_cell not in filled_cells:
                        temp_forbidden_set.add(forbidden_cell)

    return frozenset(temp_forbidden_set)


# --- Classes ---
class NuruominoState:
    """Represents a state in the Nuruomino problem."""

    # Class-level counter for unique IDs
    state_id_counter = 0

    def __init__(self, board):
        self.board = board  # The actual Board object
        self.id = NuruominoState.state_id_counter  # Assign unique ID
        NuruominoState.state_id_counter += 1  # Increment for next instance

    def __lt__(self, other):
        """Este método é utilizado em caso de empate na gestão da lista
        de abertos nas procuras informadas."""
        return self.id < other.id


class Board:
    """
    Represents the Nuruomino game board, its regions, and current assignments.
    This class is designed to be IMMUTABLE. When a "change" occurs, a new Board object is created.
    """

    def __init__(
        self,
        N,
        initial_grid_ids,
        regions_map_for_state,
        cell_to_region_id_map,
        assignments,
        filled_cells,
        forbidden_cells,
    ):
        self.N = N
        self.initial_grid_ids = initial_grid_ids
        self.cell_to_region_id_map = cell_to_region_id_map

        self.regions_map = regions_map_for_state

        self.assignments = dict(assignments)

        self._filled_cells = filled_cells
        self._forbidden_cells = forbidden_cells

        self.solution_grid = None

    def get_filled_cells(self):
        return self._filled_cells

    def get_forbidden_cells(self):
        return self._forbidden_cells

    def __hash__(self):
        # Correctly hash the assignments dictionary:
        hashable_assignments = frozenset(
            (rid, frozenset(data.items())) for rid, data in self.assignments.items()
        )

        return hash(
            (
                hashable_assignments,
                self._filled_cells,
                self._forbidden_cells,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, Board):
            return NotImplemented
        return (
            self.assignments == other.assignments
            and self._filled_cells == other._filled_cells
            and self._forbidden_cells == other._forbidden_cells
        )

    def print_board(self):
        """Generates a string representation of the board for output."""
        temp_solution_grid = [list(row) for row in self.initial_grid_ids]
        for region_id, data in self.assignments.items():
            for r, c in data["abs_cells"]:
                temp_solution_grid[r][c] = data["type"]

        output = ""
        for r in range(self.N):
            row_str = [str(temp_solution_grid[r][c]) for c in range(self.N)]
            output += "\t".join(row_str) + "\n"
        return output.strip()

    @staticmethod
    def parse_instance():
        """Parses the Nuruomino puzzle input from stdin and initializes the first Board state."""
        lines = [line.strip().split() for line in sys.stdin if line.strip()]
        if not lines:
            raise ValueError("Empty input")

        N = len(lines)
        for i, row in enumerate(lines):
            if len(row) != N:
                raise ValueError(
                    f"Invalid input: row {i+1} has {len(row)} columns, expected {N}"
                )

        initial_grid_ids = tuple(tuple(row) for row in lines)

        regions_map = {}
        cell_to_region_id_map = {}
        for r in range(N):
            for c in range(N):
                rid = initial_grid_ids[r][c]
                if rid not in regions_map:
                    regions_map[rid] = {"cells": frozenset(), "valid_placements": []}
                regions_map[rid]["cells"] = regions_map[rid]["cells"] | frozenset(
                    [(r, c)]
                )
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

                            if not all(
                                0 <= r < N and 0 <= c < N
                                for r, c in current_placement_abs_cells
                            ):
                                continue

                            valid_placements.add(
                                (tet_type, current_placement_abs_cells)
                            )

            data["valid_placements"] = sorted(
                list(valid_placements), key=lambda x: (x[0], sorted(list(x[1])))
            )

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

        initial_board = Board(
            N,
            initial_grid_ids,
            regions_map,
            cell_to_region_id_map,
            frozenset(),
            frozenset(),
            frozenset(),
        )
        return NuruominoState(initial_board)


class Problem:
    """The abstract class for a formal problem."""

    def __init__(self, initial, goal=None):
        """The initial state (Board object) and a goal state, if any."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given state.
        The result would typically be a list, set, or iterator of actions."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given action in the given state.
        The action must be one of self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal state."""
        raise NotImplementedError

    def step_cost(self, state, action):
        """Return the cost of a solution path that arrives at state2 from state1
        via action, assuming cost c to get to state1. The default is 1, for
        each step in the path."""
        return 1

    def h(self, node):
        """Return the value of the heuristic function at Node."""
        return 0


class NuruominoProblem(Problem):
    def __init__(self, initial_nuruomino_state):
        super().__init__(initial_nuruomino_state)

        initial_board = initial_nuruomino_state.board

        self.N = initial_board.N
        self.all_region_ids = sorted(initial_board.regions_map.keys())
        self.initial_grid_ids = initial_board.initial_grid_ids
        self.cell_to_region_id_map = initial_board.cell_to_region_id_map

        self.initial_regions_map_template = {
            rid: {
                "cells": data["cells"],
                "valid_placements": frozenset(data["valid_placements"]),
            }
            for rid, data in initial_board.regions_map.items()
        }

        self.region_adjacencies = self._calculate_region_adjacencies(
            initial_board.initial_grid_ids
        )
        if DEBUG_MODE:
            print(
                f"Region adjacencies precomputed: {self.region_adjacencies}",
                file=sys.stderr,
            )

        self.precomputed_future_region_cells_by_id = {
            rid: data["cells"]
            for rid, data in self.initial_regions_map_template.items()
        }

    def _calculate_region_adjacencies(self, initial_grid_ids):
        """
        Precomputes a map of region_id -> frozenset of adjacent region_ids.
        """
        adjacencies = {rid: set() for rid in self.all_region_ids}

        for r in range(self.N):
            for c in range(self.N):
                current_rid = initial_grid_ids[r][c]
                for dr, dc in [
                    (0, 1),
                    (0, -1),
                    (1, 0),
                    (-1, 0),
                ]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.N and 0 <= nc < self.N:
                        neighbor_rid = initial_grid_ids[nr][nc]
                        if neighbor_rid != current_rid:
                            adjacencies[current_rid].add(neighbor_rid)
                            adjacencies[neighbor_rid].add(current_rid)

        return {rid: frozenset(adj_rids) for rid, adj_rids in adjacencies.items()}

    def _check_future_connectivity(self, board_state):
        """
        Heuristic check: Determines if all currently filled cells PLUS the cells of ALL unassigned regions
        can form a single connected component. If not, it's impossible to form a connected final board.
        """
        board_state_to_use = board_state
        all_potential_filled_cells = set(board_state_to_use.get_filled_cells())

        for rid in self.all_region_ids:
            if rid not in board_state_to_use.assignments:
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
                    f"  Connectivity PRUNED: Visited {len(visited)} of {len(all_potential_filled_cells)} total potential cells. NOT CONNECTED.",
                    file=sys.stderr,
                )
            return False
        return True

    def actions(self, nuruomino_state):
        """
        Generates possible actions from the current board state.
        Applies Most Constrained Variable (MRV) and Degree Heuristic for variable selection,
        and Least Constraining Value (LCV) for value ordering.
        """
        current_board_state = nuruomino_state.board

        if not self._check_future_connectivity(current_board_state):
            if DEBUG_MODE:
                print(
                    f"PRUNED STATE (Actions): Board connectivity impossible.",
                    file=sys.stderr,
                )
            return []

        unassigned_regions_info = []
        for rid in self.all_region_ids:
            if rid not in current_board_state.assignments:
                num_valid_placements = len(
                    current_board_state.regions_map[rid]["valid_placements"]
                )

                if num_valid_placements == 0:
                    if DEBUG_MODE:
                        print(
                            f"  Actions Prune: Region {rid} has 0 valid placements (empty domain).",
                            file=sys.stderr,
                        )
                    return []

                degree = 0
                for adj_rid in self.region_adjacencies.get(rid, frozenset()):
                    if adj_rid not in current_board_state.assignments:
                        degree += 1
                unassigned_regions_info.append((num_valid_placements, -degree, rid))

        if not unassigned_regions_info:
            return []

        unassigned_regions_info.sort(key=lambda x: (x[0], x[1], x[2]))
        target_region_id = unassigned_regions_info[0][2]

        if DEBUG_MODE:
            print(
                f"\n--- Generating actions for State, Target Region (MRV+Deg): {target_region_id} ---",
                file=sys.stderr,
            )
            print(
                f"Current Assignments: {len(current_board_state.assignments)} regions assigned.",
                file=sys.stderr,
            )
            print(
                f"Valid Placements for Region {target_region_id}: {len(current_board_state.regions_map[target_region_id]['valid_placements'])} options.",
                file=sys.stderr,
            )

        scored_placements = []
        current_filled = current_board_state.get_filled_cells()
        current_forbidden = current_board_state.get_forbidden_cells()
        current_occupied_or_forbidden = current_filled | current_forbidden

        for tet_type, abs_cells_candidate in current_board_state.regions_map[
            target_region_id
        ]["valid_placements"]:
            constraining_score = 0

            hypothetical_filled = current_filled | abs_cells_candidate
            hypothetical_forbidden_set = _mark_forbidden_2x2_static(
                self.N, hypothetical_filled, current_forbidden, abs_cells_candidate
            )
            hypothetical_occupied_or_forbidden = (
                hypothetical_filled | hypothetical_forbidden_set
            )

            for neighbor_rid in self.region_adjacencies.get(
                target_region_id, frozenset()
            ):
                if neighbor_rid not in current_board_state.assignments:
                    neighbor_domain = current_board_state.regions_map[neighbor_rid][
                        "valid_placements"
                    ]

                    for n_tet_type, n_abs_cells_candidate in neighbor_domain:
                        if not n_abs_cells_candidate.isdisjoint(
                            hypothetical_occupied_or_forbidden
                        ):
                            constraining_score += 1
                            continue

                        if _check_if_creates_2x2_block(
                            self.N, hypothetical_filled, n_abs_cells_candidate
                        ):
                            constraining_score += 1
                            continue

                        violates_neighbor_adjacency = False
                        if n_tet_type == tet_type:
                            for r_n_cand, c_n_cand in n_abs_cells_candidate:
                                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                    nr, nc = r_n_cand + dr, c_n_cand + dc
                                    if (nr, nc) in abs_cells_candidate:
                                        violates_neighbor_adjacency = True
                                        break
                                if violates_neighbor_adjacency:
                                    break
                        if violates_neighbor_adjacency:
                            constraining_score += 1
                            continue

            scored_placements.append(
                (constraining_score, tet_type, abs_cells_candidate)
            )

        scored_placements.sort(key=lambda x: (x[0], x[1], sorted(list(x[2]))))

        return [
            (target_region_id, tet_type, abs_cells_candidate)
            for score, tet_type, abs_cells_candidate in scored_placements
        ]

    def result(self, nuruomino_state, action):
        """
        Returns the new NuruominoState after applying an action.
        Implements Forward Checking (FC) and Arc Consistency (AC-3 style propagation).
        """
        current_board_state = nuruomino_state.board
        region_id, tet_type, abs_cells = action

        new_assignments = dict(current_board_state.assignments)
        new_assignments[region_id] = {"type": tet_type, "abs_cells": abs_cells}

        new_filled_cells = current_board_state.get_filled_cells() | abs_cells

        new_forbidden_cells = _mark_forbidden_2x2_static(
            self.N,
            new_filled_cells,
            current_board_state.get_forbidden_cells(),
            abs_cells,
        )

        new_regions_map_for_state = dict(current_board_state.regions_map)

        new_regions_map_for_state[region_id] = {
            "cells": new_regions_map_for_state[region_id]["cells"],
            "valid_placements": [],
        }

        propagation_queue = deque()
        for neighbor_rid in self.region_adjacencies.get(region_id, frozenset()):
            if (
                neighbor_rid not in new_assignments
                and neighbor_rid in new_regions_map_for_state
            ):
                propagation_queue.append(neighbor_rid)

        while propagation_queue:
            rid_to_check = propagation_queue.popleft()

            if (
                new_regions_map_for_state[rid_to_check]["valid_placements"]
                is current_board_state.regions_map[rid_to_check]["valid_placements"]
            ):
                new_regions_map_for_state[rid_to_check] = {
                    "cells": new_regions_map_for_state[rid_to_check]["cells"],
                    "valid_placements": list(
                        current_board_state.regions_map[rid_to_check][
                            "valid_placements"
                        ]
                    ),
                }

            old_domain = new_regions_map_for_state[rid_to_check]["valid_placements"]
            filtered_placements = []

            current_occupied_or_forbidden = new_filled_cells | new_forbidden_cells

            for tet_type_cand, abs_cells_cand in old_domain:
                if not abs_cells_cand.isdisjoint(current_occupied_or_forbidden):
                    continue

                if _check_if_creates_2x2_block(
                    self.N, new_filled_cells, abs_cells_cand
                ):
                    continue

                violates_adjacency = False
                for r_cand, c_cand in abs_cells_cand:
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r_cand + dr, c_cand + dc
                        if (nr, nc) in new_filled_cells:
                            neighbor_region_id = self.cell_to_region_id_map.get(
                                (nr, nc)
                            )
                            if (
                                neighbor_region_id
                                and neighbor_region_id in new_assignments
                                and neighbor_region_id != rid_to_check
                            ):
                                assigned_neighbor_piece_type = new_assignments[
                                    neighbor_region_id
                                ]["type"]
                                if assigned_neighbor_piece_type == tet_type_cand:
                                    violates_adjacency = True
                                    break
                    if violates_adjacency:
                        break

                if violates_adjacency:
                    continue

                filtered_placements.append((tet_type_cand, abs_cells_cand))

            old_domain_size = len(old_domain)
            new_regions_map_for_state[rid_to_check][
                "valid_placements"
            ] = filtered_placements

            if not filtered_placements:
                if DEBUG_MODE:
                    print(
                        f"  FC in result: Region {rid_to_check}'s domain became empty. Returning dead-end state.",
                        file=sys.stderr,
                    )
                invalid_board = Board(
                    self.N,
                    self.initial_grid_ids,
                    new_regions_map_for_state,
                    self.cell_to_region_id_map,
                    frozenset(),
                    frozenset(),
                    frozenset(),
                )
                return NuruominoState(invalid_board)

            if len(filtered_placements) < old_domain_size:
                for nn_rid in self.region_adjacencies.get(rid_to_check, frozenset()):
                    if (
                        nn_rid not in new_assignments
                        and nn_rid in new_regions_map_for_state
                        and nn_rid not in propagation_queue
                    ):
                        propagation_queue.append(nn_rid)

        next_board_state = Board(
            self.N,
            self.initial_grid_ids,
            new_regions_map_for_state,
            self.cell_to_region_id_map,
            new_assignments,
            new_filled_cells,
            new_forbidden_cells,
        )
        return NuruominoState(next_board_state)

    def goal_test(self, nuruomino_state):
        """Checks if the current board state is a goal state."""
        current_board_state = nuruomino_state.board

        if len(current_board_state.assignments) != len(self.all_region_ids):
            return False

        if _check_if_creates_2x2_block(
            self.N, current_board_state.get_filled_cells(), frozenset()
        ):
            if DEBUG_MODE:
                print(
                    "Goal test failed: 2x2 block found in final state.", file=sys.stderr
                )
            return False

        if not self._check_future_connectivity(current_board_state):
            if DEBUG_MODE:
                print(
                    "Goal test failed: final shaded cells not connected.",
                    file=sys.stderr,
                )
            return False

        if DEBUG_MODE:
            print(f"Goal test PASSED!", file=sys.stderr)
        return True

    def h(self, node):
        board = node.state.board
        h_value = 0
        unassigned_regions = [
            rid for rid in self.all_region_ids if rid not in board.assignments
        ]
        num_assigned = len(board.assignments)
        total_regions = len(self.all_region_ids)
        num_forbidden_cells = len(board.get_forbidden_cells())

        if len(board.assignments) == 0 and total_regions > 0:
            return float("inf")

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
            if num_valid == 0:
                if DEBUG_MODE:
                    print(
                        f"  Heuristic (Empty Domain Prune): Region {rid} has 0 valid placements. Returning inf.",
                        file=sys.stderr,
                    )
                return float("inf")

        constraint_cost = min_valid_placements
        progress_cost = total_regions - num_assigned
        forbidden_cost = num_forbidden_cells

        piece_type_bonus = 0
        piece_weights = {
            "I": 0.4,
            "T": 0.3,
            "S": 0.2,
            "L": 0.1,
        }

        for assignment_data in board.assignments.values():
            piece_type = assignment_data["type"]
            piece_type_bonus += piece_weights.get(piece_type, 0)

        piece_type_contribution = -piece_type_bonus

        weight_constraint = 100
        weight_progress = 1000
        weight_forbidden_impact = 50
        weight_piece_type_preference = 10

        h_value = (
            (weight_constraint * constraint_cost)
            + (weight_progress * progress_cost)
            + (weight_forbidden_impact * forbidden_cost)
            + (weight_piece_type_preference * piece_type_contribution)
        )

        if not self._check_future_connectivity(board):
            if DEBUG_MODE:
                print(
                    f"  Heuristic (Connectivity Prune): Board cannot form a single connected component. Returning inf.",
                    file=sys.stderr,
                )
            return float("inf")

        h_value = max(0, h_value)

        if DEBUG_MODE:
            print(
                f"Heuristic Debug - State ID: {node.state.id}, "
                f"Min Valid Placements: {min_valid_placements}, "
                f"Unassigned Regions: {len(unassigned_regions)}, "
                f"Forbidden Cells: {num_forbidden_cells}, "
                f"Piece Type Bonus: {piece_type_bonus:.2f}, "
                f"Calculated H: {h_value:.2f}",
                file=sys.stderr,
            )

        return h_value

    def step_cost(self, state, action):
        """Cost of each step is 1."""
        return 1


class Node:
    """A node in a search tree."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

        self.f = None

    def __repr__(self):
        if isinstance(self.state, NuruominoState):
            assigned_count = len(self.state.board.assignments)
            return f"<Node ID={self.state.id} assignments={assigned_count} depth={self.depth} cost={self.path_cost}>"
        return f"<Node state={self.state} depth={self.depth} cost={self.path_cost}>"

    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        if self.path_cost != other.path_cost:
            return self.path_cost < other.path_cost
        if self.depth != other.depth:
            return self.depth > other.depth
        return self.state.id < other.state.id

    def expand(self, problem):
        """List the nodes reachable from this node."""
        return [
            self.child_node(problem, action) for action in problem.actions(self.state)
        ]

    def child_node(self, problem, action):
        """Create a new node by applying an action."""
        next_state = problem.result(self.state, action)
        new_cost = self.path_cost + problem.step_cost(self.state, action)
        return Node(next_state, self, action, new_cost)

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))


def memoize(fn, slot=None, maxsize=32):
    """Memoize fn: store results for previous calls."""
    if slot:

        def memoized_fn(obj, *args):
            if hasattr(obj, slot) and getattr(obj, slot) is not None:
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val

        return memoized_fn
    else:
        cache = {}

        def memoized_fn(*args):
            if args not in cache:
                cache[args] = fn(*args)
            return cache[args]

        return memoized_fn


class PriorityQueue:
    """A Queue in which the item with the lowest priority (lowest f-score) is retrieved first."""

    def __init__(self, order="min", f=lambda x: x):
        self.heap = []
        self.f = f
        self.order = 1 if order == "min" else -1
        self.entry_finder = {}
        self.counter = 0

    def append(self, item):
        priority = self.f(item) * self.order
        state = item.state

        if state in self.entry_finder:
            old_priority, old_count, old_item = self.entry_finder[state]
            if priority < old_priority:
                self.entry_finder[state][-1] = None
                count = self.counter
                self.counter += 1
                entry = [priority, count, item]
                self.entry_finder[state] = entry
                heapq.heappush(self.heap, entry)
        else:
            count = self.counter
            self.counter += 1
            entry = [priority, count, item]
            self.entry_finder[state] = entry
            heapq.heappush(self.heap, entry)

    def pop(self):
        while self.heap:
            priority, count, item = heapq.heappop(self.heap)
            if (
                item is not None
                and item.state in self.entry_finder
                and self.entry_finder[item.state][1] == count
            ):
                del self.entry_finder[item.state]
                return item
        raise KeyError("pop from an empty priority queue")

    def __len__(self):
        return len(self.entry_finder)

    def __contains__(self, key):
        return key in self.entry_finder

    def __getitem__(self, key):
        if key in self.entry_finder:
            entry = self.entry_finder[key]
            return entry[0] * self.order
        raise KeyError(key)


def astar_graph_search(problem, h=None, display=False):
    """
    A* Graph Search algorithm: optimized Best-First Search for optimal paths.
    Uses a priority queue and keeps track of the lowest cost found to each state
    to avoid re-exploring worse paths.
    """
    h = h or (lambda node: 0)
    f_cost_calculator = memoize(lambda node: node.path_cost + h(node), "f")

    node = Node(problem.initial, path_cost=0)
    node.f = f_cost_calculator(node)

    frontier = PriorityQueue("min", f_cost_calculator)
    frontier.append(node)

    explored_g_values = {node.state: node.path_cost}

    nodes_expanded = 0

    while frontier:
        node = frontier.pop()

        if problem.goal_test(node.state):
            if display:
                print(f"Nodes expanded: {nodes_expanded}", file=sys.stderr)
                print(f"Paths remaining in frontier: {len(frontier)}", file=sys.stderr)
                print(
                    f"Unique states explored (min g-value): {len(explored_g_values)}",
                    file=sys.stderr,
                )
            return node

        nodes_expanded += 1

        for child in node.expand(problem):
            s = child.state
            g_s = child.path_cost

            if g_s < explored_g_values.get(s, float("inf")):
                explored_g_values[s] = g_s
                child.f = f_cost_calculator(child)
                frontier.append(child)

    return None


def solve_nuruomino_astar():
    global DEBUG_MODE_CPROFILE
    DEBUG_MODE_CPROFILE = False

    try:
        initial_nuruomino_state = Board.parse_instance()
        if DEBUG_MODE:
            print(
                f"Input parsed: {initial_nuruomino_state.board.N}x{initial_nuruomino_state.board.N} grid, {len(initial_nuruomino_state.board.regions_map)} regions.",
                file=sys.stderr,
            )

        problem = NuruominoProblem(initial_nuruomino_state)
        if DEBUG_MODE:
            print("Nuruomino Problem for A* created.", file=sys.stderr)
            print("\nSolving with A* graph search...", file=sys.stderr)

        goal_node = astar_graph_search(problem, h=problem.h, display=DEBUG_MODE)

        if goal_node:
            if DEBUG_MODE:
                print("\nSolution found!", file=sys.stderr)
            solution_str = goal_node.state.board.print_board()
            print(solution_str)

        else:
            if DEBUG_MODE:
                print("\nNo solution found.", file=sys.stderr)
            print("No solution.")

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    if DEBUG_MODE:
        print("\n--- End of Execution ---", file=sys.stderr)


if __name__ == "__main__":
    run_profiling = False
    if len(sys.argv) > 1 and sys.argv[1] == "--profile":
        run_profiling = True
        DEBUG_MODE = False

    if run_profiling:
        import cProfile
        import pstats

        with cProfile.Profile() as pr:
            solve_nuruomino_astar()
        stats = pstats.Stats(pr)
        stats.sort_stats("cumulative").print_stats(30)
    else:
        solve_nuruomino_astar()
