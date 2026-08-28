# Timing-Based Overlap Features for HMM Analysis

**Generated**: 2026-07-15T14:38:53.712570

## New Feature Columns

### Pure Timing Subtypes (counts per window)
- `tr_ovl_simultaneous`: start_diff < 200ms
- `tr_ovl_backchannel`: overlap < 250ms
- `tr_ovl_smooth`: overlap 250-500ms
- `tr_ovl_competitive_timing`: overlap 500-1000ms (TIMING, not semantic)
- `tr_ovl_floor_fight`: overlap >= 1000ms

### Context Labels (lexical signal)
- `tr_ovl_context_collaborative`: overlaps with lexical support cues (e.g., "yes", "and", "right")
- `tr_ovl_context_competitive`: overlaps with lexical conflict cues (e.g., "no", "but", "wait")

### Engineered Collaboration Features
- `tr_ovl_long_ratio`: (competitive_timing + floor_fight) / total_timing -> long-overlap prevalence
- `tr_ovl_short_ratio`: (simultaneous + backchannel) / total_timing -> short-overlap prevalence
- `tr_ovl_collaboration_index`: context_collaborative / total_overlaps -> lexical collaboration signal strength

### Totals
- `tr_ovl_total`: Total overlap count (all types)
- `tr_ovl_timing_only`: Sum of all timing subtypes (should equal tr_ovl_total if fully classified)

## Rationale

The previous feature set mixed **timing-based counts** with **context-label categorization**, conflating two distinct signals:
- **Timing subtype**: Objective, duration-based (when overlaps occur and for how long)
- **Context label**: Subjective, lexical hint (what speakers said suggests collaboration/conflict despite timing)

This reconstruction separates them:
1. **Timing subtypes** capture the mechanics of simultaneous speech (short bursts, long floor fights, etc.)
2. **Context labels** capture the semantic/collaborative intent (support words on long overlaps suggest working together)
3. **Engineered ratios** combine both signals in interpretable ways (e.g., collaboration index = collaboration signal strength)

This allows the HMM to learn the relationship between **interaction dynamics** and **emotional state** with cleaner feature semantics.
