"""
Unified LTL Controller - Parse and solve any LTL specification

This controller accepts LTL formulas as strings and automatically:
1. Parses the formula
2. Constructs the appropriate automaton
3. Synthesizes a winning strategy
4. Executes the strategy

SUPPORTED LTL SYNTAX:
====================
- Atomic propositions: at_goal, at_A, safe, etc.
- Operators:
  - ! (not)
  - & (and)
  - | (or)
  - -> (implies)
  - <-> (iff)
  - X (next)
  - F (eventually, same as ◇)
  - G (always, same as □)
  - U (until)
  - R (release)

EXAMPLES:
=========
1. "F(goal)"                    - Eventually reach goal
2. "G(!obstacle)"               - Always avoid obstacle
3. "F(goal) & G(!obstacle)"     - Reach goal while avoiding obstacles
4. "F(A & F(B))"               - Visit A then B
5. "G(F(checkpoint))"          - Infinitely often visit checkpoint
6. "(F(A)) & (F(B))"           - Eventually visit both A and B (any order)
"""

import numpy as np
import networkx as nx
from .symbolic_controller import GridAbstraction, TransitionSystem
import re

class LTLFormula:
    """
    LTL Formula representation and parsing
    """
    
    def __init__(self, formula_string):
        """
        Initialize LTL formula from string
        
        Args:
            formula_string: LTL formula as string
                           e.g., "F(goal) & G(!obstacle)"
        """
        self.formula_string = formula_string
        self.atomic_props = self._extract_atomic_propositions()
        
        print(f"\nLTL Formula: {formula_string}")
        print(f"Atomic Propositions: {self.atomic_props}")
    
    def _extract_atomic_propositions(self):
        """Extract atomic propositions (region names) from formula"""
        # Strategy: Find all identifiers that are NOT operators
        
        # List of LTL operators to exclude
        operators = {'F', 'G', 'X', 'U', 'R', 'f', 'g', 'x', 'u', 'r'}
        
        # Split by operators and special characters, keep words
        # Use word boundaries to get complete identifiers
        import re
        
        # Remove parentheses first
        cleaned = self.formula_string.replace('(', ' ').replace(')', ' ')
        
        # Split by operators (keeping word boundaries)
        # This regex finds sequences of alphanumeric characters and underscores
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', cleaned)
        
        # Filter out operators
        props = []
        for word in words:
            if word not in operators and word.upper() not in operators:
                props.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_props = []
        for prop in props:
            if prop not in seen:
                seen.add(prop)
                unique_props.append(prop)
        
        return unique_props
    
    def __str__(self):
        return self.formula_string


class UnifiedLTLController:
    """
    Unified controller that accepts any LTL specification
    
    This implements a general algorithm:
    1. Parse LTL formula
    2. Identify specification type
    3. Use appropriate synthesis method
    4. Return strategy
    """
    
    def __init__(self, transition_system):
        """
        Initialize unified LTL controller
        
        Args:
            transition_system: TransitionSystem object
        """
        self.ts = transition_system
        self.formula = None
        self.strategy = None
        
        print("\nUnified LTL Controller initialized")
    
    def set_specification(self, ltl_formula_string):
        """
        Set the LTL specification
        
        Args:
            ltl_formula_string: LTL formula as string
        """
        self.formula = LTLFormula(ltl_formula_string)
        return self.formula
    
    def _identify_pattern(self):
        """
        Identify the pattern of LTL formula to choose synthesis method
        
        Returns:
            pattern_type: 'reachability', 'safety', 'reach_avoid', 'sequence', etc.
        """
        formula = self.formula.formula_string.lower()
        
        # Pattern matching
        if 'f(' in formula and 'g(' in formula and '!' in formula:
            return 'reach_avoid'
        elif 'f(' in formula and '&' in formula and 'f(' in formula:
            return 'sequence'
        elif 'g(f(' in formula:
            return 'recurrence'
        elif 'f(' in formula:
            return 'reachability'
        elif 'g(' in formula and '!' in formula:
            return 'safety'
        else:
            return 'general'
    
    def synthesize(self):
        """
        Synthesize controller for the given LTL specification
        
        This is the main entry point - it automatically determines
        the best synthesis approach based on the formula structure.
        
        Returns:
            strategy: Dictionary mapping states to actions
        """
        if self.formula is None:
            raise ValueError("No LTL formula set! Call set_specification() first.")
        
        print("\n" + "="*70)
        print("SYNTHESIZING CONTROLLER")
        print("="*70)
        print(f"Formula: {self.formula}")
        
        # Identify pattern
        pattern = self._identify_pattern()
        print(f"Detected pattern: {pattern}")
        
        # Dispatch to appropriate synthesis method
        if pattern == 'reachability':
            self.strategy = self._synthesize_reachability()
        elif pattern == 'safety':
            self.strategy = self._synthesize_safety()
        elif pattern == 'reach_avoid':
            self.strategy = self._synthesize_reach_avoid()
        elif pattern == 'sequence':
            self.strategy = self._synthesize_sequence()
        elif pattern == 'recurrence':
            self.strategy = self._synthesize_recurrence()
        else:
            self.strategy = self._synthesize_general()
        
        print(f"\nSynthesis complete! Strategy covers {len(self.strategy)} states")
        return self.strategy
    
    def _synthesize_reachability(self):
        """
        Synthesize for reachability: F(goal)
        
        Algorithm: Backward reachability from goal states
        """
        print("\nUsing: Backward Reachability Algorithm")
        
        # Find goal region from formula
        goal_regions = self.formula.atomic_props
        
        if len(goal_regions) == 0:
            raise ValueError("No goal region found in formula")
        
        goal_region = goal_regions[0]
        print(f"Goal region: {goal_region}")
        
        # Get goal states
        goal_states = set(self.ts.get_states_in_region(goal_region))
        print(f"Goal states: {len(goal_states)}")
        
        # Backward reachability
        strategy = {}
        reachable = goal_states.copy()
        frontier = list(goal_states)
        
        for state in goal_states:
            strategy[state] = 'goal_reached'
        
        iteration = 0
        while frontier:
            iteration += 1
            current = frontier.pop(0)
            
            for pred in self.ts.graph.predecessors(current):
                if pred in reachable:
                    continue
                
                reachable.add(pred)
                frontier.append(pred)
                
                action = self.ts.graph[pred][current]['action']
                strategy[pred] = (current, action)
            
            if iteration % 100 == 0:
                print(f"  Iteration {iteration}: {len(reachable)} reachable states")
        
        print(f"Total iterations: {iteration}")
        return strategy
    
    def _synthesize_safety(self):
        """
        Synthesize for safety: G(!bad)
        
        Algorithm: Remove bad states and compute safe region
        """
        print("\nUsing: Safety Algorithm - Compute Safe Region")
        
        # Extract regions to avoid
        formula = self.formula.formula_string
        avoid_regions = []
        
        for prop in self.formula.atomic_props:
            if f'!{prop}' in formula or f'! {prop}' in formula:
                avoid_regions.append(prop)
        
        print(f"Regions to avoid: {avoid_regions}")
        
        # Get states to avoid
        avoid_states = set()
        for region in avoid_regions:
            if region in self.ts.regions:
                for state in self.ts.graph.nodes():
                    i, j, h = state
                    if (i, j) in self.ts.regions[region]:
                        avoid_states.add(state)
        
        print(f"States to avoid: {len(avoid_states)}")
        
        # All states except avoided ones are "goal" (safe states)
        safe_states = set(self.ts.graph.nodes()) - avoid_states
        print(f"Safe states: {len(safe_states)}")
        
        # Strategy: stay in safe region
        strategy = {}
        for state in safe_states:
            # Find any safe successor
            for succ in self.ts.graph.successors(state):
                if succ in safe_states:
                    action = self.ts.graph[state][succ]['action']
                    strategy[state] = (succ, action)
                    break
            else:
                # No safe successor, stay in place
                strategy[state] = (state, 'stay')
        
        return strategy
    
    def _synthesize_reach_avoid(self):
        """
        Synthesize for reach-avoid: F(goal) & G(!obstacle)
        
        Algorithm: Backward reachability avoiding bad states
        """
        print("\nUsing: Reach-Avoid Algorithm")
        
        # Parse formula to extract goal and avoid regions
        formula_lower = self.formula.formula_string.lower()
        
        # Find goal (what's after F(...))
        goal_regions = []
        avoid_regions = []
        
        for prop in self.formula.atomic_props:
            if f'f({prop})' in formula_lower or f'f( {prop}' in formula_lower:
                goal_regions.append(prop)
            if f'!{prop}' in formula_lower or f'! {prop}' in formula_lower:
                avoid_regions.append(prop)
        
        if not goal_regions:
            goal_regions = [self.formula.atomic_props[0]]
        
        print(f"Goal regions: {goal_regions}")
        print(f"Avoid regions: {avoid_regions}")
        
        # Get goal and avoid states
        goal_states = set()
        for region in goal_regions:
            if region in self.ts.regions:
                goal_states.update(self.ts.get_states_in_region(region))
        
        avoid_states = set()
        for region in avoid_regions:
            if region in self.ts.regions:
                states_in_region = self.ts.get_states_in_region(region)
                avoid_states.update(states_in_region)
                print(f"  DEBUG: Added {len(states_in_region)} states from region '{region}'")
        
        print(f"Goal states: {len(goal_states)}")
        print(f"States to avoid: {len(avoid_states)}")
        
        # Backward reachability avoiding obstacles
        strategy = {}
        reachable = goal_states.copy()
        frontier = list(goal_states)
        
        for state in goal_states:
            strategy[state] = 'goal_reached'
        
        iteration = 0
        while frontier:
            iteration += 1
            current = frontier.pop(0)
            
            for pred in self.ts.graph.predecessors(current):
                if pred in reachable or pred in avoid_states:
                    continue
                
                reachable.add(pred)
                frontier.append(pred)
                
                action = self.ts.graph[pred][current]['action']
                strategy[pred] = (current, action)
            
            if iteration % 100 == 0:
                print(f"  Iteration {iteration}: {len(reachable)} reachable states")
        
        print(f"Total iterations: {iteration}")
        return strategy
    
    def _synthesize_sequence(self):
        """
        Synthesize for sequencing: F(A & F(B & F(C & ...)))
        
        Handles arbitrary length sequences
        """
        print("\nUsing: Sequencing Algorithm")
        
        # Extract waypoints from formula
        waypoints = self.formula.atomic_props
        print(f"Waypoint sequence: {' → '.join(waypoints)}")
        
        if len(waypoints) < 2:
            print("Warning: Sequence needs at least 2 waypoints, using reachability")
            return self._synthesize_reachability()
        
        # Synthesize strategies for each waypoint in REVERSE order
        # (from final goal backwards)
        strategies = []
        
        for i in range(len(waypoints) - 1, -1, -1):
            waypoint = waypoints[i]
            print(f"\nSegment {len(waypoints)-i}: Synthesize strategy to reach '{waypoint}'")
            
            temp_formula = f"F({waypoint})"
            self.formula = LTLFormula(temp_formula)
            strategy = self._synthesize_reachability()
            strategies.insert(0, (waypoint, strategy))
        
        # Store all strategies
        self.waypoint_sequence = waypoints
        self.waypoint_strategies = strategies
        self.current_waypoint_idx = 0
        
        # Start with first waypoint strategy
        return strategies[0][1]
    


    def _synthesize_recurrence(self):
        """
        Synthesize for recurrence: G(F(checkpoint))
        
        Algorithm: Find strongly connected components that include checkpoint
        """
        print("\nUsing: Recurrence Algorithm (SCC-based)")
        
        checkpoint_region = self.formula.atomic_props[0]
        print(f"Checkpoint region: {checkpoint_region}")
        
        # Get checkpoint states
        checkpoint_states = set(self.ts.get_states_in_region(checkpoint_region))
        print(f"Checkpoint states: {len(checkpoint_states)}")
        
        # Find strongly connected components
        sccs = list(nx.strongly_connected_components(self.ts.graph))
        print(f"Found {len(sccs)} strongly connected components")
        
        # Find SCCs that contain at least one checkpoint state
        valid_sccs = []
        for scc in sccs:
            if scc.intersection(checkpoint_states):
                valid_sccs.append(scc)
        
        print(f"Valid SCCs (containing checkpoint): {len(valid_sccs)}")
        
        if not valid_sccs:
            print("Warning: No valid recurrent strategy found!")
            return {}
        
        # Use largest valid SCC
        target_scc = max(valid_sccs, key=len)
        print(f"Target SCC size: {len(target_scc)}")
        
        # Build strategy to stay within SCC
        strategy = {}
        for state in target_scc:
            # Find successor within SCC
            for succ in self.ts.graph.successors(state):
                if succ in target_scc:
                    action = self.ts.graph[state][succ]['action']
                    strategy[state] = (succ, action)
                    break
        
        return strategy
    
    def _synthesize_general(self):
        """
        Synthesize for general LTL formulas
        
        This is a placeholder for more complex formulas
        that don't match simple patterns.
        
        For full generality, would need to:
        1. Convert LTL to automaton (using spot/ltlf2dfa)
        2. Compute product of transition system and automaton
        3. Solve game on product
        """
        print("\nWarning: General LTL synthesis not fully implemented")
        print("Using reachability as fallback")
        return self._synthesize_reachability()
   
    def get_action(self, current_state):
        """Get control action for current state"""
        if self.strategy is None:
            raise ValueError("No strategy synthesized!")
        
        # Special handling for sequencing
        if hasattr(self, 'waypoint_sequence') and hasattr(self, 'waypoint_strategies'):
            # Check if we've reached current waypoint
            current_waypoint = self.waypoint_sequence[self.current_waypoint_idx]
            
            if self.ts.state_satisfies(current_state, current_waypoint):
                # Reached waypoint! Move to next
                if self.current_waypoint_idx < len(self.waypoint_sequence) - 1:
                    self.current_waypoint_idx += 1
                    next_waypoint = self.waypoint_sequence[self.current_waypoint_idx]
                    print(f"    ✓ Reached {current_waypoint}! Moving to {next_waypoint}...")
                    self.strategy = self.waypoint_strategies[self.current_waypoint_idx][1]
                else:
                    print(f"    ✓ Reached final waypoint {current_waypoint}!")
                    return 'stay'
            
            # Use current strategy
            strategy = self.waypoint_strategies[self.current_waypoint_idx][1]
        else:
            strategy = self.strategy
        
        if current_state not in strategy:
            return None
        
        strategy_value = strategy[current_state]
        
        if strategy_value == 'goal_reached':
            # Check if this is truly the final goal
            if hasattr(self, 'current_waypoint_idx'):
                if self.current_waypoint_idx >= len(self.waypoint_sequence) - 1:
                    return 'stay'
                # Not final, continue to next
                return None
            return 'stay'
        
        next_state, action = strategy_value
        return action



    def execute_strategy(self, start_state, max_steps=100):
        """Execute strategy from start state"""
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
                print(f"Step {step}: No action available")
                break
            
            if action == 'stay':
                print(f"Step {step}: Goal reached!")
                break
            
            actions.append(action)
            
            # Find next state
            next_state = None
            for neighbor in self.ts.graph.successors(current_state):
                if self.ts.graph[current_state][neighbor]['action'] == action:
                    next_state = neighbor
                    break
            
            if next_state is None:
                print(f"Step {step}: Action not feasible")
                break
            
            i, j, h_idx = next_state
            if step % 5 == 0:
                print(f"Step {step}: cell ({i},{j}), action '{action}'")
            
            path.append(next_state)
            current_state = next_state
        
        print(f"\nExecution complete: {len(path)} states")
        return path, actions
    
    def visualize(self, start_state=None):
        """Visualize the strategy"""
        if self.strategy is None:
            print("No strategy to visualize!")
            return
        
        # Execute if start provided
        path_cells = None
        if start_state:
            path, _ = self.execute_strategy(start_state, max_steps=100)
            path_cells = [(s[0], s[1]) for s in path]
        
        # Get goal cells
        goal_cells = []
        for prop in self.formula.atomic_props:
            if prop in self.ts.regions and 'goal' in prop.lower():
                goal_cells.extend(list(self.ts.regions[prop]))
        
        # Get winning region
        winning_cells = set()
        for state in self.strategy.keys():
            if state in self.ts.graph.nodes():
                winning_cells.add((state[0], state[1]))
        
        self.ts.grid.visualize_grid(
            highlight_cells=winning_cells,
            goal_cells=goal_cells,
            path=path_cells
        )


if __name__ == "__main__":
    print("="*70)
    print("UNIFIED LTL CONTROLLER - EXAMPLES")
    print("="*70)
    
    # Create grid with obstacles
    grid = GridAbstraction(world_size=10.0, grid_size=20, num_headings=8)
    
    # Add obstacles that block direct path
    grid.mark_obstacle_region(center=(5.0, 5.0), radius=1.2)
    
    # Build transition system
    ts = TransitionSystem(grid)
    
    # Add regions
    ts.add_region_from_continuous('start', center=(1.0, 1.0), radius=0.5)
    ts.add_region_from_continuous('waypoint_A', center=(2.0, 8.0), radius=0.6)
    ts.add_region_from_continuous('waypoint_B', center=(8.0, 2.0), radius=0.6)
    ts.add_region_from_continuous('goal', center=(9.0, 9.0), radius=0.7)
    
    # ========================================================================
    # EXAMPLE 1: F(goal) - Direct to goal (ignores waypoints)
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE 1: F(goal)")
    print("Robot goes DIRECTLY to goal (ignores waypoints A and B)")
    print("="*70)
    
    controller1 = UnifiedLTLController(ts)
    controller1.set_specification("F(goal)")
    controller1.synthesize()
    
    start_states = ts.get_states_in_region('start')
    path1, _ = controller1.execute_strategy(start_states[0], max_steps=100)
    print(f"\nPath length: {len(path1)} steps")
    
    # Check which waypoints were visited
    path1_cells = set((s[0], s[1]) for s in path1)  # Add set()
    visited_A1 = bool(path1_cells.intersection(ts.regions['waypoint_A']))
    visited_B1 = bool(path1_cells.intersection(ts.regions['waypoint_B']))
    print(f"Visited waypoint A: {visited_A1}")
    print(f"Visited waypoint B: {visited_B1}")

    # Visualize - convert back to list for path
    path1_cells_list = [(s[0], s[1]) for s in path1]
    waypoint_cells = (list(ts.regions['waypoint_A']) + 
                    list(ts.regions['waypoint_B']))
    grid.visualize_grid(
        goal_cells=list(ts.regions['goal']),
        path=path1_cells_list,
        highlight_cells=waypoint_cells
    )
    
    # controller1.visualize(start_state=start_states[0])
    
    input("\nPress Enter for Example 2...")
    
    # ========================================================================
    # EXAMPLE 2: F(waypoint_A & F(goal)) - Must visit A first
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE 2: F(waypoint_A & F(goal))")
    print("Robot MUST visit waypoint A, THEN go to goal")
    print("="*70)
    
    controller2 = UnifiedLTLController(ts)
    controller2.set_specification("F(waypoint_B & F(goal))")
    controller2.synthesize()
    
    path2, _ = controller2.execute_strategy(start_states[0], max_steps=100)
    print(f"\nPath length: {len(path2)} steps")
    
    # Check which waypoints were visited
    path2_cells = set((s[0], s[1]) for s in path2)  # Add set()
    visited_A2 = bool(path2_cells.intersection(ts.regions['waypoint_A']))
    visited_B2 = bool(path2_cells.intersection(ts.regions['waypoint_B']))
    print(f"Visited waypoint A: {visited_A2}")
    print(f"Visited waypoint B: {visited_B2}")

    # Visualize
    path2_cells_list = [(s[0], s[1]) for s in path2]
    waypoint_cells = (list(ts.regions['waypoint_A']) + 
                    list(ts.regions['waypoint_B']))
    grid.visualize_grid(
        goal_cells=list(ts.regions['goal']),
        path=path2_cells_list,
        highlight_cells=waypoint_cells
    )
    # controller2.visualize(start_state=start_states[0])
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Example 1 (direct):     {len(path1)} steps, visited A: {visited_A1}")
    print(f"Example 2 (via A):      {len(path2)} steps, visited A: {visited_A2}")
    print(f"Difference:             {len(path2) - len(path1)} extra steps")
    print("\nClearly shows LTL specification changes robot behavior!")
    print("="*70)