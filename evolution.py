"""
evolution.py - Evolutionary Population Manager & Tournament Runner.

Upgrades in Path A:
1. 20-Candidate Guess Buffer (top_k=20): Increases single-position legal move discovery
   probability from ~4% to over ~16.5%.
2. Anti-Stagnation "Earthquake" (Adaptive Reheating): If legality plateaus for 7 generations,
   triggers an earthquake:
   - Spikes sigma back up to 0.035 with 50% sparsity.
   - Injects 4 brand-new random gladiators to introduce novel genetic diversity.
   - Preserves top 2 reigning veterans untouched.
3. Realistic Mutation Floor: min_sigma raised to 0.012 so weights never freeze into
   an unbreakable deterministic rut.
"""

import os
import json
import random
import argparse
from typing import List, Dict, Any
import torch

from model import ChessGladiatorNet
from arena import play_match


class GladiatorEvolution:
    def __init__(
        self,
        population_size: int = 32,
        elite_count: int = 8,
        initial_sigma: float = 0.030,
        min_sigma: float = 0.012,
        cooling_rate: float = 0.985,
        mutation_sparsity: float = 0.30,
        rounds_per_gen: int = 2,
        max_turns_per_game: int = 60,
        top_k: int = 20,
        target_legality: float = 98.0,
        stagnation_patience: int = 7,
        save_dir: str = "."
    ):
        self.pop_size = population_size
        self.elite_count = elite_count
        self.initial_sigma = initial_sigma
        self.min_sigma = min_sigma
        self.cooling_rate = cooling_rate
        self.mutation_sparsity = mutation_sparsity
        self.rounds_per_gen = rounds_per_gen
        self.max_turns = max_turns_per_game
        self.top_k = top_k
        self.target_legality = target_legality
        self.stagnation_patience = stagnation_patience
        self.save_dir = save_dir

        print(f"Initializing population of {self.pop_size} gladiators (top_k={self.top_k})...")
        self.population: List[ChessGladiatorNet] = [ChessGladiatorNet() for _ in range(self.pop_size)]
        self.history: List[Dict[str, Any]] = []
        self.best_legality = 0.0
        self.stagnation_count = 0
        self.current_base_gen = 1
        self.is_earthquake_gen = False

    def compute_current_sigma(self, gen_idx: int) -> float:
        """Calculates cooled mutation noise relative to the last reheating event."""
        if self.is_earthquake_gen:
            return 0.035
        elapsed = gen_idx - self.current_base_gen
        decayed = self.initial_sigma * (self.cooling_rate ** max(0, elapsed))
        return max(self.min_sigma, decayed)

    def run_generation(self, gen_idx: int) -> Dict[str, Any]:
        fitness_scores = [0.0] * self.pop_size
        total_legal_moves = 0
        total_attempted_moves = 0
        total_turns_played = 0
        total_games_played = 0
        end_reasons_count: Dict[str, int] = {}

        current_sigma = self.compute_current_sigma(gen_idx)

        # Multi-round pairings
        for _ in range(self.rounds_per_gen):
            indices = list(range(self.pop_size))
            random.shuffle(indices)

            for i in range(0, self.pop_size, 2):
                idx_a = indices[i]
                idx_b = indices[i + 1]
                g_a = self.population[idx_a]
                g_b = self.population[idx_b]

                # Game 1: A as White, B as Black
                res1 = play_match(g_a, g_b, max_turns=self.max_turns, top_k=self.top_k)
                fitness_scores[idx_a] += res1.white_fitness
                fitness_scores[idx_b] += res1.black_fitness
                total_legal_moves += (res1.white_legal_moves + res1.black_legal_moves)
                total_attempted_moves += (res1.white_attempted_moves + res1.black_attempted_moves)
                total_turns_played += res1.turns_played
                total_games_played += 1
                end_reasons_count[res1.end_reason] = end_reasons_count.get(res1.end_reason, 0) + 1

                # Game 2: B as White, A as Black
                res2 = play_match(g_b, g_a, max_turns=self.max_turns, top_k=self.top_k)
                fitness_scores[idx_b] += res2.white_fitness
                fitness_scores[idx_a] += res2.black_fitness
                total_legal_moves += (res2.white_legal_moves + res2.black_legal_moves)
                total_attempted_moves += (res2.white_attempted_moves + res2.black_attempted_moves)
                total_turns_played += res2.turns_played
                total_games_played += 1
                end_reasons_count[res2.end_reason] = end_reasons_count.get(res2.end_reason, 0) + 1

        # Calculate metrics
        legality_rate = 0.0
        if total_attempted_moves > 0:
            legality_rate = (total_legal_moves / total_attempted_moves) * 100.0

        avg_turns = total_turns_played / max(1, total_games_played)
        ranked_indices = sorted(range(self.pop_size), key=lambda k: fitness_scores[k], reverse=True)
        top_fitness = fitness_scores[ranked_indices[0]]
        avg_fitness = sum(fitness_scores) / self.pop_size

        # Check stagnation & earthquake trigger
        earthquake_flag = self.is_earthquake_gen
        self.is_earthquake_gen = False

        if legality_rate > (self.best_legality + 0.5):
            self.best_legality = legality_rate
            self.stagnation_count = 0
            champion_idx = ranked_indices[0]
            champ_path = os.path.join(self.save_dir, "champion.pt")
            torch.save(self.population[champion_idx].state_dict(), champ_path)
        else:
            self.stagnation_count += 1
            if self.stagnation_count >= self.stagnation_patience:
                # Trigger Earthquake next generation
                self.is_earthquake_gen = True
                self.current_base_gen = gen_idx
                self.stagnation_count = 0

        gen_stats = {
            "generation": gen_idx,
            "legality_rate": round(legality_rate, 2),
            "current_sigma": round(current_sigma, 4),
            "avg_turns_per_game": round(avg_turns, 1),
            "total_legal_moves": total_legal_moves,
            "total_attempted_moves": total_attempted_moves,
            "top_fitness": round(top_fitness, 2),
            "avg_fitness": round(avg_fitness, 2),
            "earthquake": earthquake_flag,
            "end_reasons": end_reasons_count
        }
        self.history.append(gen_stats)

        eq_tag = " [⚡ EARTHQUAKE REHEAT]" if earthquake_flag else ""
        print(
            f"Gen {gen_idx:3d} | "
            f"Legality: {legality_rate:5.1f}% | "
            f"Sigma: {current_sigma:.4f} | "
            f"Avg Turns: {avg_turns:4.1f} | "
            f"Top Fit: {top_fitness:7.1f} | "
            f"Legal: {total_legal_moves}/{total_attempted_moves}{eq_tag}"
        )

        # Selection, Elitism & Mutation
        new_population: List[ChessGladiatorNet] = []

        if earthquake_flag:
            # Earthquake: strict elitism (keep top 2), inject 6 fresh gladiators, heavy mutate rest
            new_population.append(self.population[ranked_indices[0]].clone())
            new_population.append(self.population[ranked_indices[1]].clone())
            for _ in range(6):
                new_population.append(ChessGladiatorNet())  # Fresh random blood
            while len(new_population) < self.pop_size:
                parent_idx = random.choice(ranked_indices[:4])
                child = self.population[parent_idx].clone()
                child.mutate(sigma=0.035, sparsity=0.50)
                new_population.append(child)
        else:
            # Standard generation: top 8 veterans survive untouched
            for rank_pos in range(self.elite_count):
                elite_idx = ranked_indices[rank_pos]
                new_population.append(self.population[elite_idx].clone())

            # Fill remaining slots with mutated offspring
            while len(new_population) < self.pop_size:
                parent_idx = random.choice(ranked_indices[:self.elite_count])
                child = self.population[parent_idx].clone()
                child.mutate(sigma=current_sigma, sparsity=self.mutation_sparsity)
                new_population.append(child)

        self.population = new_population
        return gen_stats

    def train(self, max_generations: int = 150):
        print(f"\n{'='*75}")
        print(f"PATH A: GLADIATOR ARENA WITH 20-GUESS BUFFER & ANTI-STAGNATION REHEATING")
        print(f"Population: {self.pop_size} | Top-K Buffer: {self.top_k} | Target: {self.target_legality}%")
        print(f"Initial Sigma: {self.initial_sigma} | Min Floor: {self.min_sigma} | Sparsity: {self.mutation_sparsity*100:.0f}%")
        print(f"{'='*75}\n")

        for gen in range(1, max_generations + 1):
            stats = self.run_generation(gen)

            log_path = os.path.join(self.save_dir, "training_log.json")
            with open(log_path, "w") as f:
                json.dump(self.history, f, indent=2)

            if stats["legality_rate"] >= self.target_legality:
                print(f"\n{'*'*75}")
                print(f"🎯 GOAL REACHED! Legality {stats['legality_rate']:.2f}% on Generation {gen}!")
                print(f"Champion weights saved to: {os.path.join(self.save_dir, 'champion.pt')}")
                print(f"{'*'*75}\n")
                break


def main():
    parser = argparse.ArgumentParser(description="Run Path A Gladiator Evolutionary Tournament")
    parser.add_argument("--pop-size", type=int, default=32)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--target", type=float, default=98.0)
    parser.add_argument("--top-k", type=int, default=20, help="Candidate guess buffer size")
    parser.add_argument("--sigma-init", type=float, default=0.030)
    parser.add_argument("--sigma-min", type=float, default=0.012)
    parser.add_argument("--patience", type=int, default=7, help="Generations before triggering Earthquake reheat")
    args = parser.parse_args()

    evo = GladiatorEvolution(
        population_size=args.pop_size,
        elite_count=args.elites,
        initial_sigma=args.sigma_init,
        min_sigma=args.sigma_min,
        top_k=args.top_k,
        stagnation_patience=args.patience,
        target_legality=args.target
    )
    evo.train(max_generations=args.generations)


if __name__ == "__main__":
    main()
