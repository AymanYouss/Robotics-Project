"""
Symbolic Control for Unicycle Robot

Uses discrete abstraction and temporal logic specifications (LTL)
to generate control strategies with formal guarantees.

THEORY:
=======
Instead of continuous trajectory tracking, symbolic control works with:
1. Discrete state space (grid cells)
2. Transition system (which cells can reach which cells)
3. Temporal logic specifications (high-level goals)
4. Controller synthesis (automatic generation of control strategy)

This provides FORMAL GUARANTEES that specifications are satisfied.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx

class GridAbstraction:
    """
    Discretize continuous state space into grid cells
    
    The continuous space [0,10] × [0,10] is divided into an n×n grid.
    Each cell represents a region of space where the robot can be.
    
    For heading (θ), we discretize into k directions: N, NE, E, SE, S, SW, W, NW
    """
    
    def __init__(self, world_size=10.0, grid_size=20, num_headings=8):
        """
        Initialize grid abstraction
        
        Args:
            world_size: Size of square world (meters)
            grid_size: Number of cells per dimension (e.g., 20 → 20×20 = 400 cells)
            num_headings: Number of discrete heading directions (usually 4 or 8)
        """
        self.world_size = world_size
        self.grid_size = grid_size
        self.num_headings = num_headings
        
        # Cell dimensions
        self.cell_width = world_size / grid_size
        self.cell_height = world_size / grid_size
        
        # Discrete headings (evenly spaced around circle)
        self.headings = [2 * np.pi * i / num_headings for i in range(num_headings)]
        self.heading_names = self._get_heading_names()
        
        # Grid of cells
        self.cells = {}
        self._create_grid()
        
        # Obstacles (cells marked as blocked)
        self.obstacle_cells = set()
        
        print(f"Grid Abstraction created:")
        print(f"  World size: {world_size}×{world_size} m")
        print(f"  Grid: {grid_size}×{grid_size} = {grid_size*grid_size} cells")
        print(f"  Cell size: {self.cell_width:.2f}×{self.cell_height:.2f} m")
        print(f"  Headings: {num_headings} directions")
        print(f"  Total states: {grid_size*grid_size*num_headings}")
    
    def _get_heading_names(self):
        """Get compass names for headings"""
        if self.num_headings == 4:
            return ['E', 'N', 'W', 'S']
        elif self.num_headings == 8:
            return ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
        else:
            return [f'{i}' for i in range(self.num_headings)]
    
    def _create_grid(self):
        """Create grid of cells"""
        cell_id = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Cell center
                x_center = (i + 0.5) * self.cell_width
                y_center = (j + 0.5) * self.cell_height
                
                # Cell boundaries
                x_min = i * self.cell_width
                x_max = (i + 1) * self.cell_width
                y_min = j * self.cell_height
                y_max = (j + 1) * self.cell_height
                
                self.cells[(i, j)] = {
                    'id': cell_id,
                    'grid_pos': (i, j),
                    'center': (x_center, y_center),
                    'bounds': (x_min, x_max, y_min, y_max),
                    'is_obstacle': False
                }
                cell_id += 1
    
    def continuous_to_cell(self, x, y):
        """
        Convert continuous position to cell indices
        
        Args:
            x, y: Continuous position in [0, world_size]
        
        Returns:
            (i, j): Grid cell indices
        """
        i = int(np.clip(x / self.cell_width, 0, self.grid_size - 1))
        j = int(np.clip(y / self.cell_height, 0, self.grid_size - 1))
        return (i, j)
    
    def cell_to_continuous(self, i, j):
        """
        Get center of cell in continuous coordinates
        
        Args:
            i, j: Grid cell indices
        
        Returns:
            (x, y): Center position
        """
        return self.cells[(i, j)]['center']
    
    def continuous_heading_to_discrete(self, theta):
        """
        Convert continuous heading to discrete heading index
        
        Args:
            theta: Heading in radians [-π, π]
        
        Returns:
            heading_idx: Index of closest discrete heading
        """
        # Normalize to [0, 2π)
        theta_normalized = theta % (2 * np.pi)
        
        # Find closest discrete heading
        distances = [abs(theta_normalized - h) for h in self.headings]
        # Also check wrapped distance
        distances_wrapped = [abs(theta_normalized - h - 2*np.pi) for h in self.headings]
        
        min_dist = min(min(distances), min(distances_wrapped))
        if min_dist in distances:
            heading_idx = distances.index(min_dist)
        else:
            heading_idx = distances_wrapped.index(min_dist)
        
        return heading_idx
    
    def discrete_heading_to_continuous(self, heading_idx):
        """
        Convert discrete heading index to continuous angle
        
        Args:
            heading_idx: Discrete heading index
        
        Returns:
            theta: Heading in radians
        """
        return self.headings[heading_idx]
    
    def mark_obstacle_region(self, center, radius):
        """
        Mark cells as obstacles based on circular region
        
        Args:
            center: (x, y) center of obstacle
            radius: Radius of obstacle
        """
        cx, cy = center
        
        for (i, j), cell in self.cells.items():
            cell_center = cell['center']
            distance = np.sqrt((cell_center[0] - cx)**2 + (cell_center[1] - cy)**2)
            
            if distance < radius:
                cell['is_obstacle'] = True
                self.obstacle_cells.add((i, j))
        
        print(f"Marked {len(self.obstacle_cells)} cells as obstacles")
    
    def get_neighbors(self, i, j, heading_idx):
        """
        Get neighboring cells reachable from current cell with given heading
        
        For a unicycle, we can:
        - Move forward (in direction of heading)
        - Turn left or right (change heading)
        - Stay in place
        
        Args:
            i, j: Current cell
            heading_idx: Current heading
        
        Returns:
            List of (next_i, next_j, next_heading_idx, action_name)
        """
        neighbors = []
        
        # Action 1: Move forward
        theta = self.headings[heading_idx]
        dx = np.cos(theta)
        dy = np.sin(theta)
        
        # Move approximately one cell in heading direction
        next_i = i + int(np.round(dx))
        next_j = j + int(np.round(dy))
        
        # Check bounds
        if 0 <= next_i < self.grid_size and 0 <= next_j < self.grid_size:
            if not self.cells[(next_i, next_j)]['is_obstacle']:
                neighbors.append((next_i, next_j, heading_idx, 'forward'))
        
        # Action 2: Turn left (counterclockwise)
        next_heading_left = (heading_idx + 1) % self.num_headings
        neighbors.append((i, j, next_heading_left, 'turn_left'))
        
        # Action 3: Turn right (clockwise)
        next_heading_right = (heading_idx - 1) % self.num_headings
        neighbors.append((i, j, next_heading_right, 'turn_right'))
        
        # Action 4: Stay (for safety/waiting)
        neighbors.append((i, j, heading_idx, 'stay'))
        
        return neighbors
    
    def visualize_grid(self, highlight_cells=None, goal_cells=None, path=None):
        """
        Visualize the grid with obstacles
        
        Args:
            highlight_cells: List of (i,j) cells to highlight
            goal_cells: List of (i,j) goal cells
            path: List of (i,j) cells forming a path
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw grid
        for (i, j), cell in self.cells.items():
            x_min, x_max, y_min, y_max = cell['bounds']
            
            # Color based on type
            if cell['is_obstacle']:
                color = 'red'
                alpha = 0.5
            elif highlight_cells and (i, j) in highlight_cells:
                color = 'yellow'
                alpha = 0.6
            elif goal_cells and (i, j) in goal_cells:
                color = 'green'
                alpha = 0.6
            else:
                color = 'white'
                alpha = 0.1
            
            rect = Rectangle((x_min, y_min), self.cell_width, self.cell_height,
                           linewidth=0.5, edgecolor='gray', facecolor=color, alpha=alpha)
            ax.add_patch(rect)
        
        # Draw path if provided
        if path:
            path_x = [self.cells[cell]['center'][0] for cell in path]
            path_y = [self.cells[cell]['center'][1] for cell in path]
            ax.plot(path_x, path_y, 'b-', linewidth=3, marker='o', markersize=5, label='Path')
        
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Grid Abstraction')
        ax.grid(True, alpha=0.3)
        if path:
            ax.legend()
        
        plt.tight_layout()
        plt.show()


class TransitionSystem:
    """
    Transition system representing reachable states and transitions
    
    This is a directed graph where:
    - Nodes = discrete states (cell, heading)
    - Edges = possible transitions (actions)
    - Labels = atomic propositions true in each state
    """
    
    def __init__(self, grid_abstraction):
        """
        Initialize transition system from grid abstraction
        
        Args:
            grid_abstraction: GridAbstraction object
        """
        self.grid = grid_abstraction
        self.graph = nx.DiGraph()
        
        # Atomic propositions (regions of interest)
        self.regions = {}  # name -> set of (i,j) cells
        
        print("\nBuilding transition system...")
        self._build_graph()
        print(f"Transition system built:")
        print(f"  States: {self.graph.number_of_nodes()}")
        print(f"  Transitions: {self.graph.number_of_edges()}")
    
    def _build_graph(self):
        """Build the transition graph"""
        # Create all states
        for (i, j), cell in self.grid.cells.items():
            if cell['is_obstacle']:
                continue  # Skip obstacle cells
            
            for h_idx in range(self.grid.num_headings):
                state = (i, j, h_idx)
                self.graph.add_node(state, cell=(i,j), heading=h_idx)
        
        # Add transitions
        for state in self.graph.nodes():
            i, j, h_idx = state
            neighbors = self.grid.get_neighbors(i, j, h_idx)
            
            for next_i, next_j, next_h, action in neighbors:
                next_state = (next_i, next_j, next_h)
                if next_state in self.graph.nodes():
                    self.graph.add_edge(state, next_state, action=action)
    
    def add_region(self, name, cells):
        """
        Add a region (atomic proposition)
        
        Args:
            name: Name of region (e.g., "goal", "safe", "checkpoint_A")
            cells: List of (i, j) cells in this region
        """
        self.regions[name] = set(cells)
        print(f"Region '{name}' added with {len(cells)} cells")
    
    def add_region_from_continuous(self, name, center, radius):
        """
        Add a circular region from continuous coordinates
        
        Args:
            name: Region name
            center: (x, y) center in continuous coordinates
            radius: Radius in meters
        """
        cx, cy = center
        cells = []
        
        for (i, j), cell in self.grid.cells.items():
            if cell['is_obstacle']:
                continue
            cell_center = cell['center']
            distance = np.sqrt((cell_center[0] - cx)**2 + (cell_center[1] - cy)**2)
            if distance <= radius:
                cells.append((i, j))
        
        self.add_region(name, cells)
    
    def state_satisfies(self, state, proposition):
        """
        Check if a state satisfies an atomic proposition
        
        Args:
            state: (i, j, heading_idx)
            proposition: Region name
        
        Returns:
            True if state is in the region
        """
        i, j, h_idx = state
        if proposition not in self.regions:
            return False
        return (i, j) in self.regions[proposition]
    
    def get_states_in_region(self, region_name):
        """
        Get all states in a region (with any heading)
        
        Args:
            region_name: Name of region
        
        Returns:
            List of states
        """
        if region_name not in self.regions:
            return []
        
        states = []
        for state in self.graph.nodes():
            if self.state_satisfies(state, region_name):
                states.append(state)
        return states


class LTLSpecification:
    """
    LTL (Linear Temporal Logic) specification
    
    Common specifications:
    - Safety: □(¬obstacle) = "always avoid obstacles"
    - Reachability: ◇(goal) = "eventually reach goal"
    - Sequencing: ◇(A ∧ ◇B) = "visit A then B"
    - Surveillance: □◇(checkpoint) = "infinitely often visit checkpoint"
    """
    
    def __init__(self, spec_type, regions, description=""):
        """
        Initialize LTL specification
        
        Args:
            spec_type: Type of specification
                - 'reach': Reachability ◇(goal)
                - 'reach_avoid': ◇(goal) ∧ □(¬obstacle)
                - 'sequence': ◇(A ∧ ◇B)
                - 'patrol': □◇(A) ∧ □◇(B)
            regions: Dictionary with region names needed for spec
            description: Human-readable description
        """
        self.spec_type = spec_type
        self.regions = regions
        self.description = description
        
        print(f"\nLTL Specification: {spec_type}")
        print(f"Description: {description}")
        print(f"Regions: {list(regions.keys())}")
    
    def __str__(self):
        return f"LTL[{self.spec_type}]: {self.description}"


class SymbolicController:
    """
    Symbolic controller that synthesizes control strategy from LTL specification
    
    APPROACH:
    1. Build transition system from grid
    2. Define LTL specification
    3. Solve graph search problem to find satisfying strategy
    4. Execute strategy step-by-step
    """
    
    def __init__(self, transition_system, specification):
        """
        Initialize symbolic controller
        
        Args:
            transition_system: TransitionSystem object
            specification: LTLSpecification object
        """
        self.ts = transition_system
        self.spec = specification
        self.strategy = None
        
        print(f"\nSymbolic Controller initialized")
        print(f"Specification: {self.spec}")
    
    def synthesize_reach_avoid(self, goal_region, avoid_regions=None):
        """
        Synthesize controller for reach-avoid specification:
        ◇(goal) ∧ □(¬avoid)
        
        "Eventually reach goal while always avoiding obstacles"
        
        This is solved using backward reachability analysis.
        
        Args:
            goal_region: Name of goal region
            avoid_regions: List of region names to avoid
        
        Returns:
            strategy: Dictionary mapping states to actions
        """
        print("\n" + "="*60)
        print("SYNTHESIZING REACH-AVOID CONTROLLER")
        print("="*60)
        
        # Get goal states
        goal_states = set(self.ts.get_states_in_region(goal_region))
        print(f"Goal states: {len(goal_states)}")
        
        # Get states to avoid
        avoid_states = set()
        if avoid_regions:
            for region in avoid_regions:
                avoid_states.update(self.ts.get_states_in_region(region))
                print(f"  DEBUG: Region '{region}' has {len(self.ts.regions[region])} cells, {len([s for s in avoid_states if s in self.ts.graph.nodes()])} states")

        print(f"States to avoid: {len(avoid_states)}")
        
        # Backward reachability from goal
        strategy = {}
        reachable = goal_states.copy()
        frontier = list(goal_states)
        
        # Mark goal states (no action needed, already there)
        for state in goal_states:
            strategy[state] = 'goal_reached'
        
        iteration = 0
        while frontier:
            iteration += 1
            current = frontier.pop(0)
            
            # Look at predecessors (states that can reach current)
            for pred in self.ts.graph.predecessors(current):
                # Skip if already processed or should be avoided
                if pred in reachable or pred in avoid_states:
                    continue
                
                # This predecessor can reach the goal
                reachable.add(pred)
                frontier.append(pred)
                
                # Store action to take from pred to current
                action = self.ts.graph[pred][current]['action']
                strategy[pred] = (current, action)
            
            if iteration % 100 == 0:
                print(f"  Iteration {iteration}: Reachable states = {len(reachable)}")
        
        print(f"\nSynthesis complete!")
        print(f"  Total iterations: {iteration}")
        print(f"  Reachable states: {len(reachable)}")
        print(f"  Strategy covers: {len(strategy)} states")
        
        self.strategy = strategy
        return strategy
    
    def synthesize_sequence(self, waypoint_regions):
        """
        Synthesize controller for sequencing specification:
        ◇(A ∧ ◇(B ∧ ◇(C ∧ ...)))
        
        "Visit waypoints in sequence: A then B then C ..."
        
        Args:
            waypoint_regions: List of region names in order
        
        Returns:
            strategy: Dictionary mapping (state, progress) to actions
        """
        print("\n" + "="*60)
        print("SYNTHESIZING SEQUENCING CONTROLLER")
        print("="*60)
        print(f"Waypoints: {' → '.join(waypoint_regions)}")
        
        # We'll build a product automaton:
        # State = (grid_state, progress)
        # progress = which waypoint we've reached so far
        
        strategies = []
        
        # Work backwards from last waypoint
        for i in range(len(waypoint_regions) - 1, -1, -1):
            current_wp = waypoint_regions[i]
            
            if i == len(waypoint_regions) - 1:
                # Last waypoint: just reach it
                print(f"\nWaypoint {i+1}/{len(waypoint_regions)}: Reach '{current_wp}'")
                strategy_i = self.synthesize_reach_avoid(current_wp)
            else:
                # Intermediate waypoint: reach it, then continue
                next_wp = waypoint_regions[i + 1]
                print(f"\nWaypoint {i+1}/{len(waypoint_regions)}: Reach '{current_wp}' then '{next_wp}'")
                
                # First reach current waypoint
                strategy_i = self.synthesize_reach_avoid(current_wp)
                
                # Then switch to strategy for next waypoint
                # (This is simplified; full implementation would use product automaton)
            
            strategies.append(strategy_i)
        
        # Combine strategies
        # For simplicity, we'll use a hierarchical approach
        self.strategy = strategies[0]  # Start with first waypoint strategy
        self.waypoint_strategies = strategies
        self.waypoint_regions = waypoint_regions
        self.current_waypoint_idx = 0
        
        print(f"\nSequencing controller synthesized!")
        return self.strategy
    
    def get_action(self, current_state):
        """
        Get control action for current state
        
        Args:
            current_state: (i, j, heading_idx)
        
        Returns:
            action: Action to take ('forward', 'turn_left', 'turn_right', 'stay')
                   or None if no action available
        """
        if self.strategy is None:
            raise ValueError("No strategy synthesized! Call synthesize_* first.")
        
        if current_state not in self.strategy:
            print(f"Warning: State {current_state} not in strategy!")
            return None
        
        strategy_value = self.strategy[current_state]
        
        if strategy_value == 'goal_reached':
            return 'stay'  # Already at goal
        
        next_state, action = strategy_value
        return action
    
    def execute_strategy(self, start_state, max_steps=100):
        """
        Execute the synthesized strategy from start state
        
        Args:
            start_state: (i, j, heading_idx) starting state
            max_steps: Maximum number of steps
        
        Returns:
            path: List of states visited
            actions: List of actions taken
        """
        if self.strategy is None:
            raise ValueError("No strategy synthesized!")
        
        path = [start_state]
        actions = []
        current_state = start_state
        
        print("\n" + "="*60)
        print("EXECUTING STRATEGY")
        print("="*60)
        
        for step in range(max_steps):
            action = self.get_action(current_state)
            
            if action is None:
                print(f"Step {step}: No action available (outside winning region)")
                break
            
            if action == 'stay':
                print(f"Step {step}: Goal reached!")
                break
            
            actions.append(action)
            
            # Find next state by following this action
            next_state = None
            for neighbor in self.ts.graph.successors(current_state):
                if self.ts.graph[current_state][neighbor]['action'] == action:
                    next_state = neighbor
                    break
            
            if next_state is None:
                print(f"Step {step}: Action '{action}' not feasible from {current_state}")
                break
            
            i, j, h_idx = next_state
            heading_name = self.ts.grid.heading_names[h_idx]
            cell_center = self.ts.grid.cell_to_continuous(i, j)
            
            if step % 5 == 0 or step < 5:
                print(f"Step {step}: cell ({i},{j}) at ({cell_center[0]:.1f},{cell_center[1]:.1f}), "
                      f"heading {heading_name}, action='{action}'")
            
            path.append(next_state)
            current_state = next_state
        
        print(f"\nExecution complete: {len(path)} states visited")
        return path, actions
    
    def visualize_strategy(self, start_state=None, goal_region=None):
        """
        Visualize the synthesized strategy
        
        Args:
            start_state: (i, j, heading_idx) starting state
            goal_region: Name of goal region to highlight
        """
        if self.strategy is None:
            print("No strategy to visualize!")
            return
        
        # Get path if start state provided
        path_cells = None
        if start_state:
            path, actions = self.execute_strategy(start_state, max_steps=100)
            path_cells = [(s[0], s[1]) for s in path]
        
        # Get goal cells
        goal_cells = None
        if goal_region and goal_region in self.ts.regions:
            goal_cells = list(self.ts.regions[goal_region])
        
        # Get winning region (states in strategy)
        winning_cells = set()
        for state in self.strategy.keys():
            if state in self.ts.graph.nodes():
                i, j, h = state
                winning_cells.add((i, j))
        
        self.ts.grid.visualize_grid(
            highlight_cells=winning_cells,
            goal_cells=goal_cells,
            path=path_cells
        )

    def get_states_in_region(self, region_name):
        """
        Get all states in a region (with any heading)
        
        Args:
            region_name: Name of region
        
        Returns:
            List of states (i, j, heading_idx)
        """
        if region_name not in self.regions:
            return []
        
        states = []
        for state in self.graph.nodes():
            i, j, h_idx = state
            if (i, j) in self.regions[region_name]:
                states.append(state)
        return states


# Test symbolic controller
if __name__ == "__main__":
    print("="*70)
    print("TESTING SYMBOLIC CONTROLLER")
    print("="*70)
    
    # Step 1: Create grid
    print("\n1. Creating grid abstraction...")
    grid = GridAbstraction(world_size=10.0, grid_size=20, num_headings=8)
    
    # Add obstacles
    grid.mark_obstacle_region(center=(5.0, 5.0), radius=0.8)
    grid.mark_obstacle_region(center=(8.0, 7.0), radius=0.8)
    grid.mark_obstacle_region(center=(1.0, 2.0), radius=0.8)
    grid.mark_obstacle_region(center=(3.0, 1.0), radius=0.8)
    grid.mark_obstacle_region(center=(7.0, 4.0), radius=0.8)
    
    # Step 2: Build transition system
    print("\n2. Building transition system...")
    ts = TransitionSystem(grid)
    
    # Step 3: Define regions
    print("\n3. Defining regions...")
    ts.add_region_from_continuous('start', center=(1.0, 1.0), radius=0.5)
    ts.add_region_from_continuous('goal', center=(9.0, 9.0), radius=0.5)
    ts.add_region_from_continuous('obstacle', center=(5.0, 5.0), radius=0.8)
    ts.add_region_from_continuous('obstacle', center=(8.0, 7.0), radius=0.8)
    ts.add_region_from_continuous('obstacle', center=(1.0, 2.0), radius=0.8)
    ts.add_region_from_continuous('obstacle', center=(3.0, 1.0), radius=0.8)
    ts.add_region_from_continuous('obstacle', center=(7.0, 4.0), radius=0.8)
    
    # Step 4: Create LTL specification
    print("\n4. Creating LTL specification...")
    spec = LTLSpecification(
        spec_type='reach_avoid',
        regions={'goal': 'goal', 'avoid': 'obstacle'},
        description="Reach goal while avoiding obstacle: ◇(goal) ∧ □(¬obstacle)"
    )
    
    # Step 5: Create controller
    print("\n5. Creating symbolic controller...")
    controller = SymbolicController(ts, spec)
    
    # Step 6: Synthesize strategy
    print("\n6. Synthesizing control strategy...")
    strategy = controller.synthesize_reach_avoid(goal_region='goal', avoid_regions=['obstacle'])
    
    # Step 7: Execute from start
    print("\n7. Executing strategy...")
    start_states = ts.get_states_in_region('start')
    if start_states:
        start_state = start_states[0]
        path, actions = controller.execute_strategy(start_state, max_steps=50)
        
        print(f"\nPath length: {len(path)} states")
        print(f"Actions taken: {len(actions)}")
    
    # Step 8: Visualize
    print("\n8. Visualizing strategy...")
    controller.visualize_strategy(start_state=start_states[0] if start_states else None, 
                                  goal_region='goal')
    
    print("\n" + "="*70)
    print("SYMBOLIC CONTROLLER TEST COMPLETE!")
    print("="*70)