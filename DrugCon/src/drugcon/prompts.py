"""Prompt templates for contradiction detection and explanation.

The template text below is kept byte-for-byte identical to the version used to
produce the released runs in ``results/runs``. Every template has a stable
``version`` string that is written into each run's filename, so outputs from
different prompts never collide.

Detection prompts
-----------------
``counterfactual``         -> ``counterfactual_appendix_v2``  (CF, paper Appendix B.1)
``direct``                 -> ``drug_disease_claim_only_v2``  (DC, paper Appendix B.2)
``counterfactual_strict``  -> ``counterfactual_appendix_v1``  (earlier strict-logic CF prompt)

Explanation prompts
-------------------
``taxonomy``     -> ``taxonomy_v1``   (factor names only; the 10 benchmark factors)
``descriptive``  -> ``appendix_v1``   (factor names + descriptions; 9-factor variant)

See ``docs/prompts.md`` for which prompt backs each number in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

SYSTEM_PROMPT = "You are a helpful biomedical reasoning assistant."


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def build_detection_prompt_direct(claim1: str, claim2: str) -> str:
    """Direct-comparison (valence-discordance) prompt."""
    return f"""Task: Determine whether two biomedical claims contain a potential contradiction about the same drug-disease association.
For this task, "contradiction" means a potential conflict in the overall assessment of the drug for the disease, not strict logical impossibility.
Focus only on:
- the drug, treatment, or intervention
- the disease, condition, or indication
- the overall direction/valence of the claim about that drug-disease association
Classify as "Contradiction exists" if the claims concern the same drug or drug class and the same disease/condition, and their overall assessments are discordant.
Do NOT reject a contradiction merely because the claims differ in population, outcome, endpoint, biomarker, comparator, dose, duration, study design, or statistical method.
Classify as "Claims can coexist" only if the claims are about different drugs/diseases, share the same valence, or make no clear assessment.
Keep the reasoning short and direct.
Claim 1: {claim1}
Claim 2: {claim2}
Output Format - Return your answer in the exact structure below:
<result>
<reasoning> [reasoning] </reasoning>
<conclusion> [One of: "Contradiction exists", "Claims can coexist"] </conclusion>
</result>
Output:"""


def build_detection_prompt_counterfactual_strict(claim1: str, claim2: str) -> str:
    """Earlier bidirectional counterfactual prompt (strict logical coexistence)."""
    return f"""Task:
(1) Assume Claim 1 is true. Under this assumption, evaluate whether Claim 2 could still be true.
(2) Assume Claim 2 is true. Under this assumption, evaluate whether Claim 1 could still be true.

Task Notes:
Identify any components or subpoints in the two claims that cannot logically coexist under these assumptions.
The two claims are considered contradictory if either (1) or (2) identifies a contradiction.

Claim 1: {claim1}
Claim 2: {claim2}

Output Format - Return your answer in the exact structure below:
<result>
<reasoning> [Your reasoning steps here] </reasoning>
<conclusion> [One of: "Contradiction exists", "Claims can coexist"] </conclusion>
</result>
Output:"""


def build_detection_prompt_counterfactual(claim1: str, claim2: str) -> str:
    """Counterfactual prompt targeting *apparent* contradictions.

    Keeps the bidirectional "assume-true" structure but (a) targets the overall
    assessment of the drug for the disease rather than strict logical
    impossibility, and (b) states that differing outcomes, subgroups, doses, or
    conditions do NOT excuse a contradiction, and that partial/subclaim
    conflicts count.
    """
    return f"""Task: Decide whether two biomedical claims about the same drug--disease association present a contradiction, using counterfactual reasoning.
(1) Assume Claim 1 is true. Under this assumption, would Claim 2's overall assessment of the drug for the disease still hold?
(2) Assume Claim 2 is true. Under this assumption, would Claim 1's overall assessment of the drug for the disease still hold?

Definition:
A "contradiction" here is an *apparent* contradiction: the two claims give opposing overall assessments of the same drug for the same disease (e.g., effective vs. ineffective, beneficial vs. harmful, safe vs. unsafe). It need NOT be a strict logical impossibility.

Task Notes:
a. The two claims are contradictory if either (1) or (2) reveals that the overall assessments oppose each other.
b. A contradiction may be partial: if one subclaim or subpoint in a claim opposes the other claim, treat the pair as contradictory, even if other parts agree.
c. Do NOT dismiss a contradiction merely because the two studies differ in outcome measured, patient subgroup, dose, duration, timing, comparator, or study design. Opposing conclusions still count as a contradiction even if they could be reconciled by such differences.
d. Classify as "Claims can coexist" only if the claims concern different drugs or diseases, express the same overall direction, or make no clear assessment of the drug--disease association.

Claim 1: {claim1}
Claim 2: {claim2}

Output Format - Return your answer in the exact structure below:
<result>
<reasoning> [Your reasoning steps here] </reasoning>
<conclusion> [One of: "Contradiction exists", "Claims can coexist"] </conclusion>
</result>
Output:"""


# --------------------------------------------------------------------------- #
# Explanation
# --------------------------------------------------------------------------- #
TAXONOMY_BLOCK = """- Species
- Population
- Dosage or exposure duration
- Route or mode of administration
- Combined drug effects
- Evolving scientific evidence
- Known controversy or self-qualified claims
- Study design
- Outcome measures
- Ambiguous referent"""


def build_explanation_prompt_taxonomy(claim1: str, abs1: str, claim2: str, abs2: str) -> str:
    """Compact prompt listing the 10 benchmark factor names."""
    return f"""Task: Two biomedical claims report opposing outcomes for the same drug-disease association.
Determine which contextual factor(s) from the taxonomy explain the apparent contradiction.
Base your analysis strictly on the provided claims and abstracts. Do not use external knowledge.

Contextual Factor Taxonomy:
{TAXONOMY_BLOCK}

Claim 1: {claim1}
Claim 1 context: {abs1}
Claim 2: {claim2}
Claim 2 context: {abs2}

Output Format - Return your answer in the exact structure below:
<result>
<conclusion>
contextual_factor: [comma-separated factor names | none]
</conclusion>
<explanation> [brief explanation] </explanation>
</result>
Output:"""


def build_explanation_prompt_descriptive(claim1: str, abs1: str, claim2: str, abs2: str) -> str:
    """Prompt with per-factor descriptions (9-factor variant)."""
    return f"""Task:
Given two biomedical claims that appear to report opposing outcomes for the same
drug--disease association, analyze whether the apparent contradiction can be
explained by differences in contextual conditions.

Specifically:
(1) Examine Claim 1 together with its source abstract.
(2) Examine Claim 2 together with its source abstract.
(3) Determine whether the opposing outcomes can be explained by at least one contextual factor from the taxonomy below, and provide an appropriate explanation of how this factor accounts for the contradiction.

Contextual Factor Taxonomy:
- Species. The claims are based on different biological species (e.g., animal models vs. humans).
- Population. The claims target different human subpopulations, such as differences in age, sex, genetic background, comorbidities, ethnicity, or risk profiles.
- Physiological context. The intervention is evaluated under different transient physiological or environmental conditions (e.g., exertion state, hypoxia, fasting status, or acute stress), even within the same species and population.
- Dosage or exposure duration. The same intervention is administered at different doses, frequencies, or durations.
- Route or mode of administration. The intervention is delivered via different routes (e.g., oral, intravenous, topical, sublingual, localized).
- Combined drug effects. The reported effect of a drug depends on its use in combination with other drugs or therapies.
- Evolving scientific evidence. The claims reflect different stages of scientific understanding, such as early hypotheses, preliminary findings, or later large-scale confirmatory studies.
- Known controversy or self-qualified claims. One or both claims explicitly acknowledge uncertainty, limitations, debate, or conditional validity.
- Ambiguous expression. One or both claims contain grammatical errors, unclear referents, vague quantification, or incomplete statements that obscure the meaning of a claim.

Task Notes:
Base your analysis strictly on the provided claims and abstracts.
Do not use external biomedical knowledge.
Do not speculate beyond what is explicitly stated in the text.

Claim 1: {claim1}
Claim 1 context: {abs1}
Claim 2: {claim2}
Claim 2 context: {abs2}

Output Format - Return your answer in the exact structure below:
<result>
<conclusion>
contextual_factor: [factor name | none]
</conclusion>
<explanation>
[Brief explanation describing how the identified contextual factor accounts for the apparent contradiction, with references to specific textual evidence.]
</explanation>
</result>
Output:"""


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PromptSpec:
    alias: str
    version: str
    builder: Callable[..., str]
    description: str


DETECTION_PROMPTS: dict[str, PromptSpec] = {
    "counterfactual": PromptSpec(
        "counterfactual", "counterfactual_appendix_v2", build_detection_prompt_counterfactual,
        "Counterfactual reasoning, apparent-contradiction definition (CF)."),
    "direct": PromptSpec(
        "direct", "drug_disease_claim_only_v2", build_detection_prompt_direct,
        "Direct comparison of drug-disease valence (DC)."),
    "counterfactual_strict": PromptSpec(
        "counterfactual_strict", "counterfactual_appendix_v1", build_detection_prompt_counterfactual_strict,
        "Earlier counterfactual prompt using strict logical coexistence."),
}

EXPLANATION_PROMPTS: dict[str, PromptSpec] = {
    "taxonomy": PromptSpec(
        "taxonomy", "taxonomy_v1", build_explanation_prompt_taxonomy,
        "10-factor taxonomy, factor names only."),
    "descriptive": PromptSpec(
        "descriptive", "appendix_v1", build_explanation_prompt_descriptive,
        "9-factor taxonomy with per-factor descriptions."),
}

DEFAULT_DETECTION_PROMPT = "counterfactual"
DEFAULT_EXPLANATION_PROMPT = "taxonomy"


def get_prompt(task: str, alias: str) -> PromptSpec:
    registry = DETECTION_PROMPTS if task == "detection" else EXPLANATION_PROMPTS
    if alias not in registry:
        raise KeyError(f"Unknown {task} prompt '{alias}'. Choose from {sorted(registry)}.")
    return registry[alias]
