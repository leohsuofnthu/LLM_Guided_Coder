"""
LLM-guided recursive net discovery implementation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import DiscoveryConfig, DiscoverySubnetConfig
from .embedder import Embedder
from .llm_client import LLMClient
from .utils import (
    cosine_normalize,
    get_logger,
    sample_texts,
)

@dataclass
class Net:
    id: str
    label: str
    seed_phrases: List[str]
    prototype: np.ndarray
    status: str = "active"
    members: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    iteration: int = 0
    parent_id: Optional[str] = None

    def to_payload(self) -> Dict:
        return {
            "id": self.id,
            "label": self.label,
            "seeds": self.seed_phrases,
            "status": self.status,
            "metrics": self.metrics,
            "parent": self.parent_id,
        }


@dataclass
class DiscoveryIterationRecord:
    iteration: int
    coverage: float
    unknown_count: int
    mean_confidence: float
    notes: str = ""


@dataclass
class RecursiveDiscoveryResult:
    nets: List[Net]
    assignments: pd.DataFrame
    topic_hierarchy: Dict
    label_lookup: Dict[str, Dict[str, str]]
    history: List[DiscoveryIterationRecord]
    llm_calls: int
    iteration_metrics: List[Dict[str, float]]
    llm_identifier: str


class RecursiveNetDiscovery:
    """Coordinates recursive net and subnet discovery."""

    def __init__(
        self,
        question: str,
        segments_df: pd.DataFrame,
        embeddings: np.ndarray,
        embedder: Embedder,
        config: DiscoveryConfig,
        llm_client: Optional[LLMClient] = None,
        global_registry: Optional[List[Dict[str, Any]]] = None,
        labeler_prompt: Optional[str] = None,
        random_state: int = 13,
        level: str = "net",
        logger=None,
    ) -> None:
        if "segment_id" not in segments_df or "segment_text" not in segments_df:
            raise KeyError("segments_df must contain 'segment_id' and 'segment_text' columns.")
        if segments_df.shape[0] != embeddings.shape[0]:
            raise ValueError("segments_df and embeddings must align by row.")
        self.question = question
        self.segments_df = segments_df.reset_index(drop=True)
        self.embeddings = embeddings.astype(np.float32)
        self.embedder = embedder
        self.config = config
        self.level = level
        self.logger = logger or get_logger(__name__)
        if llm_client is None:
            raise ValueError("llm_client is required for RecursiveNetDiscovery.")
        self.llm: LLMClient = llm_client
        self.labeler_prompt = labeler_prompt
        self.global_registry: List[Dict[str, Any]] = (
            global_registry if global_registry is not None else []
        )
        self._layer_registry: Dict[str, Dict[str, Any]] = {}
        seed = random_state if self.config.deterministic else random.randint(0, 2**31 - 1)
        self.random_state = seed
        self.history: List[DiscoveryIterationRecord] = []
        self.iteration_metrics: List[Dict[str, float]] = []
        self.llm_calls: int = 0
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        self._rng = random.Random(seed)
        self.segment_lookup = self.segments_df.set_index("segment_id")
        self.segment_index = {
            segment_id: idx for idx, segment_id in enumerate(self.segments_df["segment_id"])
        }
        self.synonym_lookup = {
            key.lower(): value for key, value in (self.config.synonym_map or {}).items()
        }
        if not np.allclose(np.linalg.norm(self.embeddings, axis=1), 1.0, atol=1e-3):
            cosine_normalize(self.embeddings)

    # ------------------------------------------------------------------
    def _register_net(self, net_id: str, label: str, embedding: np.ndarray) -> None:
        normalized_label = self._normalize_label(label).lower()
        vector = embedding.astype(np.float32, copy=True)
        entry = {"id": net_id, "label": normalized_label, "embedding": vector}
        self._layer_registry[net_id] = entry
        self.global_registry.append({"id": net_id, "label": normalized_label, "embedding": vector})

    def _update_registry_label(self, net_id: str, label: str) -> None:
        normalized_label = self._normalize_label(label).lower()
        if net_id in self._layer_registry:
            self._layer_registry[net_id]["label"] = normalized_label
        for entry in self.global_registry:
            if entry["id"] == net_id:
                entry["label"] = normalized_label
                break

    def _update_registry_embedding(self, net_id: str, embedding: np.ndarray) -> None:
        vector = embedding.astype(np.float32, copy=True)
        if net_id in self._layer_registry:
            self._layer_registry[net_id]["embedding"] = vector
        for entry in self.global_registry:
            if entry["id"] == net_id:
                entry["embedding"] = vector
                break

    def _create_fallback_net(self) -> Net:
        centroid = np.mean(self.embeddings, axis=0)
        if centroid.size == 0:
            raise RuntimeError("Cannot create fallback net: no embeddings available.")
        norm = np.linalg.norm(centroid)
        if norm == 0:
            centroid = self.embeddings[0]
            norm = np.linalg.norm(centroid)
        centroid = centroid / max(norm, 1e-6)
        label = self._normalize_label("General Feedback")
        net = Net(
            id="net_fallback",
            label=label,
            seed_phrases=[label],
            prototype=centroid.astype(np.float32),
        )
        self._layer_registry.clear()
        self.global_registry.clear()
        self._register_net(net.id, net.label, net.prototype)
        return net

    def _rename_registry_entry(self, old_id: str, new_id: str) -> None:
        if old_id == new_id:
            return
        if old_id in self._layer_registry:
            entry = self._layer_registry.pop(old_id)
            entry["id"] = new_id
            self._layer_registry[new_id] = entry
        for entry in self.global_registry:
            if entry["id"] == old_id:
                entry["id"] = new_id
                break

    def _is_duplicate(self, label: str, embedding: np.ndarray) -> bool:
        if not self.config.dedupe_enabled:
            return False
        normalized_label = self._normalize_label(label).lower()
        threshold = self.config.duplicate_similarity_threshold
        for entry in self._layer_registry.values():
            if normalized_label == entry["label"]:
                return True
            if float(np.dot(entry["embedding"], embedding)) >= threshold:
                return True
        for entry in self.global_registry:
            if normalized_label == entry["label"]:
                return True
            if float(np.dot(entry["embedding"], embedding)) >= threshold:
                return True
        return False

    # Public API -----------------------------------------------------------------

    def run(self) -> RecursiveDiscoveryResult:
        nets = self._bootstrap_nets()
        assignment_df = None
        previous_assignments = None

        for iteration in range(1, self.config.max_iterations + 1):
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("🔄 ITERATION %s/%s", iteration, self.config.max_iterations)
            self.logger.info("=" * 60)
            
            if self.config.max_llm_calls is not None and self.llm_calls >= self.config.max_llm_calls:
                self.logger.info("LLM call budget reached; stopping refinement loop.")
                break
            if not any(net.status == "active" for net in nets):
                self.logger.warning(
                    "All nets dropped before routing; injecting fallback general net."
                )
                fallback = self._create_fallback_net()
                self._rename_registry_entry(fallback.id, "net_0")
                fallback.id = "net_0"
                nets = [fallback]

            active_count = len([n for n in nets if n.status == "active"])
            self.logger.info("📊 Active nets: %d", active_count)
            
            assignments = self._route_segments(nets)
            assignment_df = assignments["frame"]

            nets = self._update_memberships(nets, assignments)
            self._refresh_nets(nets)
            nets = self._governance(nets, assignments)
            nets = self._enforce_global_cap(nets)

            coverage = 1.0 - assignments["unknown_mask"].mean()
            mean_confidence = assignment_df["confidence"].mean()
            self.history.append(
                DiscoveryIterationRecord(
                    iteration=iteration,
                    coverage=coverage,
                    unknown_count=int(assignments["unknown_mask"].sum()),
                    mean_confidence=float(np.nan_to_num(mean_confidence)),
                )
            )
            if iteration == 1 and coverage < 0.5:
                self.logger.warning(
                    "Initial net coverage below 50%% (%.2f%%). Consider revising bootstrap seeds.",
                    coverage * 100,
                )
            iteration_metrics = self._compute_iteration_metrics(assignment_df, nets)
            self.iteration_metrics.append(iteration_metrics)
            self.history[-1].notes = f"metrics={iteration_metrics}"
            self.logger.info("Iteration %s metrics: %s", iteration, iteration_metrics)

            if previous_assignments is not None:
                delta = self._assignment_delta(previous_assignments, assignment_df)
                self.logger.info("Assignment delta: %.4f", delta)
                if delta < self.config.improvement_tol:
                    self.logger.info("Converged after iteration %s", iteration)
                    break
            previous_assignments = assignment_df.copy()

        if assignment_df is None:
            raise RuntimeError("Discovery did not produce assignments.")

        nets = [net for net in nets if net.status == "active"]
        
        # Final routing pass: if splits occurred in last iteration, re-assign to new split nets
        self.logger.info("Performing final routing pass to ensure all segments are assigned to active nets.")
        final_assignments = self._route_segments(nets)
        assignment_df = final_assignments["frame"]
        nets = self._update_memberships(nets, final_assignments)
        
        assignment_df = self._finalize_assignments(assignment_df, nets)

        if self.config.subnet.enabled and self.level == "net":
            self.logger.info("Discovering subnets recursively.")
            nets, assignment_df = self._discover_subnets(nets, assignment_df)

        hierarchy, label_lookup = self._build_hierarchy(nets, assignment_df)
        return RecursiveDiscoveryResult(
            nets=nets,
            assignments=assignment_df,
            topic_hierarchy=hierarchy,
            label_lookup=label_lookup,
            history=self.history,
            llm_calls=self.llm_calls,
            iteration_metrics=self.iteration_metrics,
            llm_identifier=self._llm_identifier(),
        )

    def _invoke_llm(self, method_name: str, *args, **kwargs):
        """Call the LLM client with budget enforcement and logging."""
        if self.config.max_llm_calls is not None and self.llm_calls >= self.config.max_llm_calls:
            raise RuntimeError("LLM call budget exceeded during discovery.")

        # Rate limiting: ensure at least 2 seconds between calls (30 calls/min = 1 every 2s)
        import time
        wait_time = 0.0
        if hasattr(self, '_last_llm_call_time'):
            time_since_last = time.time() - self._last_llm_call_time
            if time_since_last < 2.0:
                wait_time = 2.0 - time_since_last
                time.sleep(wait_time)
        
        # Log the call details
        call_number = self.llm_calls + 1
        self.logger.info(
            "🔄 LLM Call #%d: %s (wait: %.1fs, total: %d/%s)",
            call_number,
            method_name,
            wait_time,
            call_number,
            self.config.max_llm_calls if self.config.max_llm_calls else "∞"
        )
        
        start_time = time.time()
        method = getattr(self.llm, method_name)
        result = method(*args, **kwargs)
        elapsed = time.time() - start_time
        
        self.llm_calls += 1
        self._last_llm_call_time = time.time()
        
        # Log completion
        self.logger.info(
            "✅ LLM Call #%d completed in %.2fs",
            call_number,
            elapsed
        )
        
        return result

    # Internal helpers -----------------------------------------------------------

    def _bootstrap_nets(self) -> List[Net]:
        self.logger.info("=" * 60)
        self.logger.info("🚀 BOOTSTRAP PHASE: Proposing initial nets")
        self.logger.info("=" * 60)
        
        # Treat all segments as initial "unknown" cluster
        all_segments = self.segments_df["segment_id"].tolist()
        self.logger.info("📊 Total segments for bootstrap: %d", len(all_segments))
        
        # Apply diversity sampling to bootstrap segment pool
        if self.config.bootstrap_sample_pct is not None:
            target_size = max(1, int(len(all_segments) * self.config.bootstrap_sample_pct))
        else:
            target_size = self.config.bootstrap_sample_size
        
        if len(all_segments) > target_size:
            sampled_segments = self._select_diverse_segment_ids(all_segments, target_size)
            self.logger.info(
                "  📊 Sampled %d diverse segments from %d total for bootstrap",
                len(sampled_segments),
                len(all_segments)
            )
        else:
            sampled_segments = all_segments
            self.logger.info(
                "  📊 Using all %d segments for bootstrap (below sample threshold)",
                len(all_segments)
            )
        
        # Get segment texts for LLM
        bootstrap_texts = self.segment_lookup.loc[sampled_segments, "segment_text"].tolist()
        
        aggregated: List[dict] = []
        rounds = max(1, self.config.bootstrap_rounds)
        for round_idx in range(rounds):
            self.logger.info("📋 Bootstrap round %d/%d", round_idx + 1, rounds)
            variant_question = (
                self.question
                if round_idx == 0
                else f"{self.question} (angle {round_idx + 1})"
            )
            proposals = self._invoke_llm(
                "propose_nets",
                variant_question,
                texts=bootstrap_texts,
                num_nets=self.config.max_bootstrap_nets,
                seeds_per_net=self.config.bootstrap_seeds_per_net,
            )
            aggregated.extend(proposals or [])
        self.logger.info(
            "Bootstrap proposals collected: rounds=%s total_raw=%s",
            rounds,
            len(aggregated),
        )
        if aggregated:
            raw_sample_labels = [str(p.get("label", "")).strip() for p in aggregated[: min(8, len(aggregated))]]
            self.logger.info("Sample raw bootstrap labels: %s", raw_sample_labels)

        # Within-batch deduplication of proposals before creating nets
        aggregated = self._dedupe_proposals(aggregated)
        self.logger.info(
            "Bootstrap proposals after within-batch dedupe: kept=%s",
            len(aggregated),
        )
        if aggregated:
            sample_labels = [str(p.get("label", "")).strip() for p in aggregated[: min(8, len(aggregated))]]
            self.logger.info("Sample deduped bootstrap labels: %s", sample_labels)

        nets: List[Net] = []
        for payload in aggregated:
            seeds = payload.get("seeds") or []
            if not seeds:
                continue
            prototype = self._embed_phrases(seeds)
            label = self._normalize_label(payload.get("label") or f"Net {len(nets) + 1}")
            if self._is_duplicate(label, prototype):
                self.logger.info("Skipping bootstrap net '%s' due to duplication.", label)
                continue
            net_id = f"net_{len(nets)}"
            net = Net(
                id=net_id,
                label=label,
                seed_phrases=seeds,
                prototype=prototype,
            )
            nets.append(net)
            self._register_net(net_id, label, prototype)

        if not nets:
            self.logger.warning(
                "LLM bootstrap produced no unique nets; creating fallback general net."
            )
            fallback = self._create_fallback_net()
            nets.append(fallback)
        else:
            self.logger.info("LLM bootstrap created %s unique nets.", len(nets))

        nets.sort(key=lambda n: len(n.seed_phrases), reverse=True)
        nets = nets[: self.config.max_bootstrap_nets]
        for idx, net in enumerate(nets):
            new_id = f"net_{idx}"
            self._rename_registry_entry(net.id, new_id)
            net.id = new_id
        return nets

    def _route_segments(self, nets: List[Net]) -> Dict:
        active_nets = [net for net in nets if net.status == "active"]
        if not active_nets:
            raise RuntimeError("No active nets available for routing.")

        proto_matrix = np.vstack([net.prototype for net in active_nets])
        cosine_normalize(proto_matrix)
        scores = self.embeddings @ proto_matrix.T

        if proto_matrix.shape[0] == 1:
            max_scores = scores[:, 0]
            second_scores = np.zeros_like(max_scores)
            margins = max_scores
            primary_idx = np.zeros(scores.shape[0], dtype=int)
        else:
            max_scores = scores.max(axis=1)
            second_scores = np.partition(scores, -2, axis=1)[:, -2]
            margins = max_scores - second_scores
            primary_idx = scores.argmax(axis=1)

        confidence = 0.6 * max_scores + 0.4 * margins

        primary_nets = [active_nets[i].id for i in primary_idx]

        alt_nets = []
        if proto_matrix.shape[0] == 1:
            alt_nets = [[] for _ in range(scores.shape[0])]
        else:
            alt_nets = []
            for row_idx, row in enumerate(scores):
                primary = row[primary_idx[row_idx]]
                alt_candidates = [
                    active_nets[col_idx].id
                    for col_idx, value in enumerate(row)
                    if col_idx != primary_idx[row_idx]
                    and primary - value <= self.config.multi_label_delta
                ]
                alt_nets.append(alt_candidates)

        unknown_mask = confidence < self.config.assignment_threshold
        primary_labels = [
            primary_nets[i] if not unknown_mask[i] else self.config.unknown_label
            for i in range(len(primary_nets))
        ]

        frame = pd.DataFrame(
            {
                "segment_id": self.segments_df["segment_id"],
                "net_id": primary_labels,
                "confidence": confidence,
                "margin": margins,
                "max_score": max_scores,
                "second_score": second_scores,
                "alternative_nets": alt_nets,
            }
        )
        return {
            "frame": frame,
            "scores": scores,
            "unknown_mask": unknown_mask,
            "active_nets": active_nets,
        }

    def _update_memberships(self, nets: List[Net], assignments: Dict) -> List[Net]:
        frame = assignments["frame"]
        for net in nets:
            members = frame.loc[frame["net_id"] == net.id, "segment_id"].tolist()
            net.members = members
            net.metrics["member_count"] = len(members)
            net.metrics["coverage"] = len(members) / max(len(self.segments_df), 1)
        return nets

    def _refresh_nets(self, nets: List[Net]) -> None:
        candidates: List[Net] = []
        for net in nets:
            if net.status != "active":
                continue
            if len(net.members) < self.config.min_net_size:
                continue
            last_delta = net.metrics.get("last_refresh_delta")
            stability_hits = net.metrics.get("stability_hits", 0)
            if (
                last_delta is not None
                and last_delta < self.config.skip_stability_delta
                and stability_hits >= self.config.skip_stability_patience
            ):
                self.logger.debug("Skipping stable net %s (delta=%.4f)", net.id, last_delta)
                continue
            candidates.append(net)

        if self.config.max_refresh_per_iteration is not None:
            candidates.sort(key=lambda n: n.metrics.get("mean_confidence", 0.0))
            candidates = candidates[: self.config.max_refresh_per_iteration]

        self.logger.info("🔄 Refreshing %d nets (skipped %d stable)", len(candidates), len(nets) - len(candidates) - len([n for n in nets if n.status != "active"]))
        
        for idx, net in enumerate(candidates, 1):
            if self.config.max_llm_calls is not None and self.llm_calls >= self.config.max_llm_calls:
                self.logger.info("LLM call budget reached; skipping remaining net refreshes.")
                break
            self.logger.info("  🔧 Refreshing net %d/%d: '%s' (%d members)", idx, len(candidates), net.label, len(net.members))
            self._refresh_single_net(net)

    def _refresh_single_net(self, net: Net) -> None:
        sample_ids = self._select_representative_ids(net, self.config.refresh_sample_size)
        if not sample_ids:
            return
        sample_texts_list = self.segment_lookup.loc[sample_ids, "segment_text"].tolist()
        net.metrics["last_sampled_segments"] = sample_ids[: min(10, len(sample_ids))]
        prev_conf = net.metrics.get("mean_confidence", 0.0)
        old_prototype = net.prototype.copy()
        old_label = net.label
        old_seed_embeddings = self._embed_texts(net.seed_phrases)
        update = self._invoke_llm(
            "summarize_net",
            sample_texts_list,
            max_words=5,
            seeds_per_net=self.config.bootstrap_seeds_per_net,
            fallback_label=net.label,
            system_prompt=self.labeler_prompt,
        )
        new_label = update.get("label", net.label)
        seeds = update.get("seeds", []) or net.seed_phrases
        net.prototype = self._embed_phrases(seeds)
        net.seed_phrases = seeds
        self._update_registry_embedding(net.id, net.prototype)
        label_similarity = self._label_similarity(old_label, new_label)
        seed_shift, seed_shift_fraction = self._seed_shift(old_seed_embeddings, net.prototype)
        normalized_old_label = self._normalize_label(old_label)
        normalized_new_label = self._normalize_label(new_label)
        if label_similarity >= 0.9 or seed_shift_fraction < 0.5:
            net.label = normalized_old_label
        else:
            net.label = normalized_new_label
        self._update_registry_label(net.id, net.label)

        sample_members = net.members[: min(200, len(net.members))]
        confidences = [self._segment_confidence(seg_id, net) for seg_id in sample_members]
        new_conf = float(np.mean(confidences)) if confidences else prev_conf
        delta = abs(new_conf - prev_conf)
        if delta < self.config.skip_stability_delta:
            net.metrics["stability_hits"] = net.metrics.get("stability_hits", 0) + 1
        else:
            net.metrics["stability_hits"] = 0
        prototype_similarity = float(np.dot(old_prototype, net.prototype))
        net.metrics["prototype_similarity"] = prototype_similarity
        net.metrics["seed_shift"] = seed_shift
        net.metrics["seed_shift_fraction"] = seed_shift_fraction
        net.metrics["last_label_similarity"] = label_similarity
        net.metrics["last_refresh_delta"] = delta
        net.metrics["mean_confidence"] = new_conf
        net.metrics["member_count"] = len(net.members)
        self.logger.info(
            "Net %s refresh: label='%s', proto_sim=%.3f, seed_shift=%.3f (%.0f%%), conf_delta=%.3f",
            net.id,
            net.label,
            prototype_similarity,
            seed_shift,
            seed_shift_fraction * 100,
            delta,
        )
        if prototype_similarity >= 0.9:
            net.metrics["stability_hits"] = max(net.metrics.get("stability_hits", 0), 2)
        if prototype_similarity >= 0.9 and seed_shift < 0.05:
            net.metrics["frozen"] = True
        else:
            net.metrics["frozen"] = False
        net.iteration += 1

    def _segment_confidence(self, segment_id: str, net: Net) -> float:
        idx = self.segments_df.index[self.segments_df["segment_id"] == segment_id]
        if len(idx) == 0:
            return 0.0
        vector = self.embeddings[idx[0]]
        return float(vector @ net.prototype)

    def _governance(self, nets: List[Net], assignments: Dict) -> List[Net]:
        nets = self._drop_small_nets(nets)
        nets = self._merge_similar_nets(nets)
        nets = self._add_from_unknown(nets, assignments)
        # Splitting is optional and depends on sample size
        nets = self._maybe_split_nets(nets)
        return nets

    def _enforce_global_cap(self, nets: List[Net]) -> List[Net]:
        cap = self.config.max_total_nets
        if not cap:
            return nets
        active = [net for net in nets if net.status == "active"]
        if len(active) <= cap:
            return nets
        surplus = len(active) - cap
        to_drop = sorted(active, key=lambda n: len(n.members))[:surplus]
        for net in to_drop:
            self.logger.info(
                "Dropping %s to respect global cap (%s active nets).",
                net.id,
                cap,
            )
            net.status = "dropped"
            net.metrics["dropped_reason"] = "global_cap"
        return nets

    def _drop_small_nets(self, nets: List[Net]) -> List[Net]:
        # Calculate threshold: percentage-based (if set) or absolute size
        if self.config.min_net_size_pct is not None:
            total_segments = len(self.segments_df)
            threshold = max(1, int(total_segments * self.config.min_net_size_pct))
        else:
            threshold = self.config.min_net_size // 2
        
        for net in nets:
            if net.status != "active":
                continue
            if len(net.members) < threshold:
                self.logger.info("Dropping %s due to low membership (%s, threshold: %s).", net.id, len(net.members), threshold)
                net.status = "dropped"
        return nets

    def _merge_similar_nets(self, nets: List[Net]) -> List[Net]:
        active = [net for net in nets if net.status == "active"]
        if len(active) < 2:
            return nets
        proto_matrix = np.vstack([net.prototype for net in active])
        cosine_normalize(proto_matrix)
        sim_matrix = proto_matrix @ proto_matrix.T
        merged_pairs: set[Tuple[str, str]] = set()
        
        # Find all similar pairs above threshold
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                similarity = sim_matrix[i, j]
                if similarity >= self.config.merge_similarity_threshold:
                    self.logger.info(
                        "  🔗 Merging similar nets: '%s' <-> '%s' (similarity: %.3f)",
                        active[i].label,
                        active[j].label,
                        similarity
                    )
                    merged_pairs.add((active[i].id, active[j].id))
        
        # Apply merges
        id_to_net = {net.id: net for net in nets}
        for left_id, right_id in merged_pairs:
            left = id_to_net.get(left_id)
            right = id_to_net.get(right_id)
            if left is None or right is None:
                continue
            if right.status == "dropped":  # Already merged
                continue
            self.logger.info("  ✅ Merged '%s' into '%s'", right.label, left.label)
            left.members.extend(right.members)
            left.seed_phrases = list(set(left.seed_phrases + right.seed_phrases))
            left.prototype = self._embed_phrases(left.seed_phrases)
            self._update_registry_embedding(left.id, left.prototype)
            right.status = "dropped"
        
        if merged_pairs:
            self.logger.info("  📊 Total merges: %d pairs", len(merged_pairs))
        
        return nets

    def _add_from_unknown(self, nets: List[Net], assignments: Dict) -> List[Net]:
        frame = assignments["frame"]
        unknown_segments = frame.loc[frame["net_id"] == self.config.unknown_label, "segment_id"].tolist()
        if not unknown_segments:
            return nets
        
        # Calculate min threshold: percentage-based (if set) or absolute size
        if self.config.min_net_size_pct is not None:
            total_segments = len(self.segments_df)
            min_threshold = max(1, int(total_segments * self.config.min_net_size_pct))
        else:
            min_threshold = self.config.min_net_size
        
        if len(unknown_segments) < min_threshold:
            self.logger.info("⏭️  Skipping unknown mining: only %d unknown segments (min: %d)", len(unknown_segments), min_threshold)
            return nets
        if self.config.max_llm_calls is not None and self.llm_calls >= self.config.max_llm_calls:
            self.logger.info("LLM call budget reached; skipping unknown bucket mining.")
            return nets

        self.logger.info("🔍 UNKNOWN MINING: %d unassigned segments", len(unknown_segments))
        
        # Apply diversity sampling to unknown segments
        if self.config.unknown_sample_pct is not None:
            target_size = max(1, int(len(unknown_segments) * self.config.unknown_sample_pct))
        else:
            target_size = self.config.unknown_sample_size
        
        if len(unknown_segments) > target_size:
            sampled_unknown = self._select_diverse_segment_ids(unknown_segments, target_size)
            self.logger.info(
                "  📊 Sampled %d diverse segments from %d unknown for LLM proposals",
                len(sampled_unknown),
                len(unknown_segments)
            )
        else:
            sampled_unknown = unknown_segments
            self.logger.info(
                "  📊 Using all %d unknown segments for LLM proposals",
                len(unknown_segments)
            )
        
        unknown_texts = self.segment_lookup.loc[sampled_unknown, "segment_text"].tolist()
        proposals = self._invoke_llm(
            "propose_new_nets",
            unknown_texts,
            max_nets=self.config.max_unknown_clusters,
            seeds_per_net=self.config.bootstrap_seeds_per_net,
            existing_labels=[net.label for net in nets if net.status == "active"],
        )
        self.logger.info(
            "Unknown mining proposals collected: raw=%s", len(proposals or [])
        )
        if proposals:
            raw_unknown_labels = [str(p.get("label", "")).strip() for p in proposals[: min(8, len(proposals))]]
            self.logger.info("Sample raw unknown labels: %s", raw_unknown_labels)
        # Within-batch deduplication of unknown proposals
        proposals = self._dedupe_proposals(proposals)
        self.logger.info(
            "Unknown mining proposals after dedupe: kept=%s", len(proposals or [])
        )
        if proposals:
            kept_unknown_labels = [str(p.get("label", "")).strip() for p in proposals[: min(8, len(proposals))]]
            self.logger.info("Sample deduped unknown labels: %s", kept_unknown_labels)
        
        limited_proposals = proposals[: self.config.max_unknown_clusters]
        for payload in limited_proposals:
            seeds = payload.get("seeds") or []
            if not seeds:
                continue
            proto = self._embed_phrases(seeds)
            label = self._normalize_label(payload.get("label", f"Candidate {len(nets)}"))
            if self._is_duplicate(label, proto):
                self.logger.info("Skipping candidate net '%s' from unknown bucket (duplicate).", label)
                continue
            net_id = f"net_{len(nets)}"
            net = Net(
                id=net_id,
                label=label,
                seed_phrases=seeds,
                prototype=proto,
                status="active",
            )
            nets.append(net)
            self._register_net(net_id, label, proto)
        if limited_proposals:
            self.logger.info(
                "Unknown mining added %s new nets (after dedupe).",
                sum(1 for p in limited_proposals if (p.get("seeds") or [])),
            )
        return nets

    def _maybe_split_nets(self, nets: List[Net]) -> List[Net]:
        for net in nets:
            if net.status != "active":
                continue
            if len(net.members) < self.config.min_net_size * 2:
                continue
            member_indices = self.segments_df.index[
                self.segments_df["segment_id"].isin(net.members)
            ].tolist()
            member_embeddings = self.embeddings[member_indices]
            if member_embeddings.shape[0] < 2:
                continue
            try:
                from sklearn.metrics import silhouette_score
                from sklearn.cluster import KMeans
            except ImportError:
                return nets

            kmeans = KMeans(n_clusters=2, random_state=self.random_state).fit(member_embeddings)
            labels = kmeans.labels_
            silhouette = silhouette_score(member_embeddings, labels)
            if silhouette < self.config.split_silhouette_threshold:
                continue
            self.logger.info("Splitting %s based on silhouette %.3f", net.id, silhouette)
            if self.config.max_llm_calls is not None and self.llm_calls >= self.config.max_llm_calls:
                self.logger.info("LLM call budget reached; skipping split proposal for %s.", net.id)
                continue
            sample_ids = self._rng.sample(net.members, min(len(net.members), 60))
            split_payload = self._invoke_llm(
                "propose_split",
                self.segment_lookup.loc[sample_ids, "segment_text"].tolist(),
            )
            split_payload = split_payload or []
            created_children = False
            for idx in range(2):
                seeds = split_payload[idx].get("seeds") if idx < len(split_payload) else None
                if not seeds:
                    continue
                new_proto = self._embed_phrases(seeds)
                label = self._normalize_label(
                    split_payload[idx].get("label", f"{net.label} {idx+1}")
                )
                if self._is_duplicate(label, new_proto):
                    self.logger.info("Skipping split candidate '%s' due to duplication.", label)
                    continue
                new_net = Net(
                    id=f"{net.id}_split_{idx}",
                    label=label,
                    seed_phrases=seeds,
                    prototype=new_proto,
                    status="active",
                )
                nets.append(new_net)
                self._register_net(new_net.id, label, new_proto)
                created_children = True
            if created_children:
                net.status = "dropped"
        return nets

    def _assignment_delta(self, prev: pd.DataFrame, current: pd.DataFrame) -> float:
        merged = prev.merge(current, on="segment_id", suffixes=("_prev", "_curr"))
        changes = (merged["net_id_prev"] != merged["net_id_curr"]).mean()
        return float(changes)

    def _finalize_assignments(self, assignments: pd.DataFrame, nets: List[Net]) -> pd.DataFrame:
        active_ids = {net.id for net in nets}
        assignments = assignments.copy()
        assignments.loc[~assignments["net_id"].isin(active_ids), "net_id"] = self.config.unknown_label
        assignments["subnet_id"] = self.config.unknown_label
        return assignments

    def _discover_subnets(self, nets: List[Net], assignments: pd.DataFrame):
        self.logger.info("=" * 60)
        self.logger.info("🌳 SUBNET DISCOVERY: Finding sub-topics")
        self.logger.info("=" * 60)
        
        updated_assignments = assignments.copy()
        all_nets = nets.copy()
        eligible_nets = [n for n in nets if len(updated_assignments[updated_assignments["net_id"] == n.id]) >= self.config.subnet.min_size]
        self.logger.info("📌 Processing %d nets (of %d total) that meet size threshold", len(eligible_nets), len(nets))
        
        for idx, net in enumerate(nets, 1):
            if self.config.max_llm_calls is not None and self.llm_calls >= self.config.max_llm_calls:
                self.logger.info("LLM call budget reached; skipping remaining subnet discovery.")
                break
            member_mask = updated_assignments["net_id"] == net.id
            members = updated_assignments.loc[member_mask, "segment_id"].tolist()
            if len(members) < self.config.subnet.min_size:
                continue

            self.logger.info("🔍 Subnet %d/%d: Analyzing '%s' (%d members)", idx, len(eligible_nets), net.label, len(members))
            
            # Apply diversity sampling for subnet discovery if members are too large
            if self.config.subnet_sample_pct is not None:
                target_size = max(1, int(len(members) * self.config.subnet_sample_pct))
            else:
                target_size = self.config.subnet_sample_size
            
            if len(members) > target_size:
                sampled_members = self._select_representative_ids(net, target_size)
                self.logger.info(
                    "  📊 Sampled %d diverse members from %d total",
                    len(sampled_members),
                    len(members)
                )
                members = sampled_members
            
            subset_mask = self.segments_df["segment_id"].isin(members)
            subset_embeddings = self.embeddings[subset_mask]
            if subset_embeddings.shape[0] < self.config.subnet.min_size:
                continue
            variance = float(np.var(subset_embeddings))
            if variance < self.config.subnet.variance_threshold:
                self.logger.debug(
                    "Skipping subnet discovery for %s due to low variance (%.6f).",
                    net.id,
                    variance,
                )
                continue
            subset_df = self.segments_df[subset_mask].reset_index(drop=True)
            sub_config = DiscoveryConfig(
                enabled=True,
                max_iterations=self.config.subnet.max_iterations,
                assignment_threshold=self.config.subnet.assignment_threshold,
                multi_label_delta=self.config.subnet.multi_label_delta,
                max_bootstrap_nets=min(self.config.subnet.max_subnets, len(members) // max(1, self.config.subnet.min_size)),
                bootstrap_rounds=self.config.bootstrap_rounds,
                bootstrap_seeds_per_net=self.config.bootstrap_seeds_per_net,
                refresh_sample_size=self.config.refresh_sample_size,
                min_net_size=max(10, self.config.subnet.min_size // 2),
                merge_similarity_threshold=self.config.merge_similarity_threshold,
                reassign_margin_quantile=self.config.reassign_margin_quantile,
                unknown_label=self.config.unknown_label,
                max_unknown_clusters=max(2, self.config.subnet.max_subnets // 2),
                improvement_tol=self.config.improvement_tol,
                min_parent_child_similarity=self.config.min_parent_child_similarity,
                min_subnet_fraction=self.config.min_subnet_fraction,
                deterministic=self.config.deterministic,
                dedupe_enabled=self.config.dedupe_enabled,
                # Propagate diversity sampling limits to child subnet discovery
                bootstrap_sample_size=self.config.bootstrap_sample_size,
                unknown_sample_size=self.config.unknown_sample_size,
                subnet_sample_size=self.config.subnet_sample_size,
                subnet=DiscoverySubnetConfig(enabled=False),
            )
            if self.config.max_subnet_refresh_per_iteration is not None:
                sub_config.max_refresh_per_iteration = self.config.max_subnet_refresh_per_iteration
            if self.config.max_llm_calls is not None:
                remaining = max(0, self.config.max_llm_calls - self.llm_calls)
                if remaining <= 0:
                    self.logger.info("No remaining LLM budget for subnet discovery; stopping.")
                    break
                sub_config.max_llm_calls = remaining
            child = RecursiveNetDiscovery(
                question=f"Subtopics of {net.label}",
                segments_df=subset_df,
                embeddings=subset_embeddings,
                embedder=self.embedder,
                config=sub_config,
                llm_client=self.llm,
                global_registry=self.global_registry,
                labeler_prompt=self.labeler_prompt,
                random_state=self.random_state + 7,
                level="subnet",
                logger=self.logger,
            )
            child_result = child.run()
            self.llm_calls += child.llm_calls
            parent_size = len(members)
            min_required = max(
                self.config.subnet.min_size,
                int(math.ceil(parent_size * self.config.min_subnet_fraction)),
            )
            parent_proto = net.prototype
            filtered_assignments = child_result.assignments.copy()
            allowed_ids: set[str] = set()
            for child_net in child_result.nets:
                child_size = len(child_net.members)
                similarity = float(np.dot(parent_proto, child_net.prototype))
                if child_size < min_required:
                    self.logger.debug(
                        "Dropping subnet %s under %s due to size (%s < %s).",
                        child_net.label,
                        net.label,
                        child_size,
                        min_required,
                    )
                    filtered_assignments.loc[
                        filtered_assignments["net_id"] == child_net.id, "net_id"
                    ] = self.config.unknown_label
                    continue
                if similarity < self.config.min_parent_child_similarity:
                    self.logger.debug(
                        "Dropping subnet %s under %s due to low similarity (%.3f).",
                        child_net.label,
                        net.label,
                        similarity,
                    )
                    filtered_assignments.loc[
                        filtered_assignments["net_id"] == child_net.id, "net_id"
                    ] = self.config.unknown_label
                    continue
                child_net.parent_id = net.id
                child_net.metrics["parent_similarity"] = similarity
                allowed_ids.add(child_net.id)
                all_nets.append(child_net)

            subnet_assignment = filtered_assignments[["segment_id", "net_id"]].rename(
                columns={"net_id": "subnet_id"}
            )
            updated_assignments = updated_assignments.merge(
                subnet_assignment, on="segment_id", how="left", suffixes=("", "_child")
            )
            updated_assignments["subnet_id"] = updated_assignments["subnet_id_child"].fillna(
                updated_assignments["subnet_id"]
            )
            updated_assignments = updated_assignments.drop(columns=["subnet_id_child"])
        return all_nets, updated_assignments

    def _build_hierarchy(self, nets: List[Net], assignments: pd.DataFrame):
        hierarchy = {"nets": {}}
        label_lookup = {"nets": {}, "subnets": {}}
        net_lookup = {net.id: net for net in nets}
        segment_frame = self.segment_lookup
        text_lookup = segment_frame["segment_text"].to_dict()
        response_lookup = (
            segment_frame["response_text"].to_dict() if "response_text" in segment_frame else {}
        )
        response_id_lookup = (
            segment_frame["response_id"].to_dict() if "response_id" in segment_frame else {}
        )

        def build_examples(member_ids: List[str], limit: int) -> List[Dict[str, Any]]:
            available = [mid for mid in member_ids if mid in text_lookup]
            if not available:
                return []
            if len(available) > limit:
                sample_ids = self._rng.sample(available, limit)
            else:
                sample_ids = available
            examples = []
            for seg_id in sample_ids:
                examples.append(
                    {
                        "segment_id": seg_id,
                        "segment_text": text_lookup.get(seg_id, ""),
                        "response_id": response_id_lookup.get(seg_id),
                        "response_text": response_lookup.get(seg_id, text_lookup.get(seg_id, "")),
                    }
                )
            return examples

        for net in nets:
            if net.parent_id:
                continue
            members = assignments.loc[assignments["net_id"] == net.id, "segment_id"].tolist()
            examples = build_examples(members, 10)
            hierarchy["nets"][net.id] = {
                "label": net.label,
                "seeds": net.seed_phrases,
                "metrics": net.metrics,
                "examples": examples,
                "subnets": {},
            }
            label_lookup["nets"][net.id] = net.label

        for net in nets:
            if not net.parent_id:
                continue
            parent_entry = hierarchy["nets"].setdefault(net.parent_id, {"subnets": {}})
            members = assignments.loc[assignments["subnet_id"] == net.id, "segment_id"].tolist()
            examples = build_examples(members, 8)
            parent_entry["subnets"][net.id] = {
                "label": net.label,
                "seeds": net.seed_phrases,
                "metrics": net.metrics,
                "examples": examples,
            }
            label_lookup["subnets"][f"{net.parent_id}|{net.id}"] = net.label

        return hierarchy, label_lookup

    def _compute_iteration_metrics(self, assignments: pd.DataFrame, nets: List[Net]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        coverage = 1.0 - float((assignments["net_id"] == self.config.unknown_label).mean())
        metrics["coverage"] = coverage
        metrics["mean_confidence"] = float(np.nan_to_num(assignments["confidence"].mean()))
        if "margin" in assignments:
            metrics["mean_margin"] = float(np.nan_to_num(assignments["margin"].mean()))
        else:
            metrics["mean_margin"] = 0.0
        metrics["active_nets"] = float(sum(1 for net in nets if net.status == "active"))
        silhouette = self._estimate_silhouette(assignments)
        if silhouette is not None:
            metrics["silhouette"] = silhouette
        return metrics

    def _estimate_silhouette(self, assignments: pd.DataFrame) -> Optional[float]:
        try:
            from sklearn.metrics import silhouette_score
        except ImportError:
            return None
        mask = assignments["net_id"] != self.config.unknown_label
        if mask.sum() < 10:
            return None
        candidate_indices = np.where(mask)[0]
        if candidate_indices.size <= 1:
            return None
        sample_size = min(2000, candidate_indices.size)
        sample_indices = np.random.choice(candidate_indices, size=sample_size, replace=False)
        sample_labels = assignments.iloc[sample_indices]["net_id"].tolist()
        if len(set(sample_labels)) < 2:
            return None
        sample_embeddings = self.embeddings[sample_indices]
        try:
            value = float(silhouette_score(sample_embeddings, sample_labels, metric="cosine"))
        except Exception:
            return None
        return value

    def _llm_identifier(self) -> str:
        return getattr(self.llm, "model_name", self.llm.__class__.__name__)

    def _normalize_label(self, label: str) -> str:
        if not label:
            return label
        normalized = label.strip()
        canonical = self.synonym_lookup.get(normalized.lower())
        if canonical:
            normalized = canonical
        return normalized.title()

    def _select_representative_ids(self, net: Net, max_count: int) -> List[str]:
        """Select diverse representatives from a net using maximin sampling."""
        ids = [sid for sid in net.members if sid in self.segment_index]
        if not ids:
            return []
        if max_count <= 0 or len(ids) <= max_count:
            return ids

        indices = np.array([self.segment_index[sid] for sid in ids])
        vectors = self.embeddings[indices]

        proto_sims = vectors @ net.prototype
        first_idx = int(np.argmax(proto_sims))
        selected_mask = np.zeros(len(ids), dtype=bool)
        selected_mask[first_idx] = True

        current_max_sim = vectors @ vectors[first_idx]
        current_max_sim[selected_mask] = 1.0

        while selected_mask.sum() < max_count:
            distances = 1.0 - current_max_sim
            distances[selected_mask] = -1.0
            next_idx = int(np.argmax(distances))
            if distances[next_idx] <= 0:
                break
            selected_mask[next_idx] = True
            new_sim = vectors @ vectors[next_idx]
            current_max_sim = np.maximum(current_max_sim, new_sim)

        selected_indices = np.where(selected_mask)[0]
        if selected_indices.size < max_count:
            # pad deterministically with highest-confidence remaining segments
            remaining = [
                idx for idx in np.argsort(proto_sims)[::-1] if idx not in selected_indices
            ]
            needed = max_count - selected_indices.size
            selected_indices = np.concatenate(
                [selected_indices, np.array(remaining[:needed], dtype=int)]
            )
        selected_indices = np.unique(selected_indices[:max_count])
        return [ids[int(idx)] for idx in selected_indices]

    def _select_diverse_segment_ids(self, segment_ids: List[str], max_count: int, center_vector: Optional[np.ndarray] = None) -> List[str]:
        """Select diverse segments from a list using maximin sampling.
        
        Args:
            segment_ids: List of segment IDs to sample from
            max_count: Maximum number of segments to select
            center_vector: Optional center vector to start from (if None, uses centroid)
        
        Returns:
            List of selected segment IDs with maximal diversity
        """
        valid_ids = [sid for sid in segment_ids if sid in self.segment_index]
        if not valid_ids:
            return []
        if max_count <= 0 or len(valid_ids) <= max_count:
            return valid_ids

        indices = np.array([self.segment_index[sid] for sid in valid_ids])
        vectors = self.embeddings[indices]

        # Start with segment closest to center
        if center_vector is not None:
            center_sims = vectors @ center_vector
            first_idx = int(np.argmax(center_sims))
        else:
            # Use centroid
            centroid = vectors.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            centroid_sims = vectors @ centroid
            first_idx = int(np.argmax(centroid_sims))
        
        selected_mask = np.zeros(len(valid_ids), dtype=bool)
        selected_mask[first_idx] = True

        current_max_sim = vectors @ vectors[first_idx]
        current_max_sim[selected_mask] = 1.0

        # Greedily select maximally distant segments
        while selected_mask.sum() < max_count:
            distances = 1.0 - current_max_sim
            distances[selected_mask] = -1.0
            next_idx = int(np.argmax(distances))
            if distances[next_idx] <= 0:
                break
            selected_mask[next_idx] = True
            new_sim = vectors @ vectors[next_idx]
            current_max_sim = np.maximum(current_max_sim, new_sim)

        selected_indices = np.where(selected_mask)[0]
        return [valid_ids[int(idx)] for idx in selected_indices]

    def _embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embeddings.shape[1]), dtype=np.float32)
        vectors = self.embedder.embed(list(texts))
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float32)
        return vectors.astype(np.float32)

    def _label_similarity(self, old_label: str, new_label: str) -> float:
        normalized_old = self._normalize_label(old_label)
        normalized_new = self._normalize_label(new_label)
        if normalized_old == normalized_new:
            return 1.0
        vectors = self._embed_texts([normalized_old, normalized_new])
        if vectors.shape[0] < 2:
            return 0.0
        return float(np.dot(vectors[0], vectors[1]))

    @staticmethod
    def _seed_shift(seed_embeddings: np.ndarray, new_proto: np.ndarray) -> Tuple[float, float]:
        if seed_embeddings.size == 0:
            return 0.0, 0.0
        sims = seed_embeddings @ new_proto
        movement = 1.0 - sims
        fraction = float(np.mean(movement > 0.15))
        return float(np.mean(movement)), fraction

    def _embed_phrases(self, phrases: Sequence[str]) -> np.ndarray:
        vectors = self._embed_texts(list(phrases))
        if vectors.size == 0:
            raise RuntimeError("Failed to embed seed phrases.")
        centroid = vectors.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            raise RuntimeError("Seed embeddings produced zero vector.")
        return centroid / norm

    def _dedupe_proposals(self, proposals: List[dict]) -> List[dict]:
        """Remove duplicate proposals within a batch based on label similarity and seed embeddings."""
        if not proposals or not self.config.dedupe_enabled:
            return proposals
        
        unique_proposals: List[dict] = []
        seen_labels: set = set()
        seen_embeddings: List[np.ndarray] = []
        
        for proposal in proposals:
            label = proposal.get("label", "")
            seeds = proposal.get("seeds") or []
            if not label or not seeds:
                self.logger.debug("Dropping proposal with missing label/seeds: %s", proposal)
                continue
            
            normalized_label = self._normalize_label(label).lower()
            
            # Check for exact label match
            if normalized_label in seen_labels:
                self.logger.debug("Skipping duplicate label in batch: '%s'", label)
                continue
            
            # Check for semantic similarity via seeds
            try:
                proto = self._embed_phrases(seeds)
            except RuntimeError:
                self.logger.debug("Failed to embed seeds for proposal: %s", proposal)
                continue
            
            is_duplicate = False
            for idx, seen_emb in enumerate(seen_embeddings):
                similarity = float(np.dot(seen_emb, proto))
                if similarity >= self.config.duplicate_similarity_threshold:
                    # Auto-deduplicate based on similarity threshold
                    existing_label = unique_proposals[idx].get("label", "Unknown")
                    self.logger.debug(
                        "Skipping duplicate proposal '%s' (similar to '%s', similarity: %.3f)",
                        label,
                        existing_label,
                        similarity
                    )
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_proposals.append(proposal)
                seen_labels.add(normalized_label)
                seen_embeddings.append(proto)
        
        return unique_proposals
