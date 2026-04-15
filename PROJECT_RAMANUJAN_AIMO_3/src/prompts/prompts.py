from dataclasses import dataclass
from typing import Dict, Optional

import day17_ablation as day17


@dataclass(frozen=True)
class ProtocolVariant:
    name: str
    instruction_placement: str
    final_answer_style: str
    strict_post_tool_finalization: bool


_BASE_SYSTEM_PROMPT = """You are an elite mathematical reasoning assistant solving olympiad-style integer-answer problems.
Work carefully and logically, but keep visible output terse and machine-parseable.
Do not write conversational filler such as "Okay, so", "Let me think", or similar tutor-style preambles.

Use this grading contract:
1. If no Python tool is needed, respond with exactly one line: FINAL_ANSWER: <integer>
2. If Python is needed, respond with exactly:
   THOUGHT: <one short sentence>
   TOOL_REQUEST:
   <valid Python code only>
   FINAL_ANSWER: PENDING
3. Only one FINAL_ANSWER line counts.
4. Repeated FINAL_ANSWER lines are invalid.
5. Any text after the final answer line is invalid.
6. Do not use markdown on the final answer line.
"""

# Backward-compatible default prompt.
SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT

_USER_ONLY_SYSTEM_PROMPT = """You are an elite mathematical reasoning assistant solving olympiad-style problems.
Work carefully and logically.
Do not write conversational filler such as "Okay, so", "Let me think", or similar tutor-style preambles.
"""

_VARIANTS = {
    "baseline": ProtocolVariant(
        name="baseline",
        instruction_placement="system_user",
        final_answer_style="plain",
        strict_post_tool_finalization=False,
    ),
    "user_only": ProtocolVariant(
        name="user_only",
        instruction_placement="user_only",
        final_answer_style="plain",
        strict_post_tool_finalization=False,
    ),
    "strict_post_tool": ProtocolVariant(
        name="strict_post_tool",
        instruction_placement="system_user",
        final_answer_style="plain",
        strict_post_tool_finalization=True,
    ),
    "boxed_directive": ProtocolVariant(
        name="boxed_directive",
        instruction_placement="system_user",
        final_answer_style="boxed",
        strict_post_tool_finalization=True,
    ),
}

_VARIANT_ALIASES = {
    "baseline": "baseline",
    "user_only": "user_only",
    "user-only": "user_only",
    "strict_post_tool": "strict_post_tool",
    "strict-post-tool": "strict_post_tool",
    "boxed_directive": "boxed_directive",
    "boxed-directive": "boxed_directive",
}

LONG_PROBLEM_LENGTH_THRESHOLD = 220


V3_DIRECT_FINAL_EASY = """Solve tersely.
Do not announce a plan. Do not use Python. Do not add commentary before the answer.
Your final output must be exactly one line:
FINAL_ANSWER: [integer]
That line must be the LAST line. Do not write anything after it."""

V3_DIRECT_FINAL_LOCKED = """Solve tersely and stay on one line of attack.
Do not announce a plan or narrate your process.
If a tiny native Python verification is truly needed, execute it directly instead of describing it.
When you have the answer, write exactly one line:
FINAL_ANSWER: [integer]
That line must be the LAST line. Do not write anything after it."""

V3_DIRECT_FINAL_COMMENTARY = """Commit to one solution approach immediately and execute it without narration.
Do not hedge, tutor, or discuss alternative ideas.
Do not mention Python or code.
End with exactly one line:
FINAL_ANSWER: [integer]
That line must be the LAST line. Do not write anything after it."""

# Backward-compatible alias retained for tests and legacy references.
V3_DIRECT_FINAL_SOFT = V3_DIRECT_FINAL_LOCKED

V3_COMMIT_THEN_EXECUTE = """Commit to ONE solution approach immediately and execute it completely.
Do not narrate alternatives. Do not mention Python or code.
If the chosen route fails, restart cleanly once rather than drifting.
Final answer format: FINAL_ANSWER: [integer]"""

V3_TOOL_FIRST = """You are solving a combinatorics or geometry search problem.
Rules:
1. You MUST write and execute Python code to search or verify your answer.
2. Do NOT describe what code would do. Write the actual code.
3. After code execution confirms your answer, write: FINAL_ANSWER: [integer]
4. Code execution is required. Pure reasoning is not acceptable for this problem type."""

V3_TOOL_FIRST_DIRECT_WITNESS = """You are solving a witness-counting search problem.
Rules:
1. Your FIRST response must be exactly one native python_exec tool call.
2. That first code block must directly test actual candidate objects against the ORIGINAL witness condition on tiny supports.
3. Do NOT begin with factorization, cyclotomic, divisor, generating-function, or representation counting arguments.
4. Count validated final objects only after code has demonstrated how witness existence is checked.
5. After code execution confirms your answer, write: FINAL_ANSWER: [integer]
6. Code execution is required. Pure reasoning is not acceptable for this problem type."""

V3_HYBRID_CLASSIFY_FIRST = """Before solving, classify this problem on your first line:
- DERIVATION: solvable by closed-form algebraic reasoning
- COMPUTATION: requires programmatic search or enumeration
- HYBRID: requires both derivation and computational verification
Then commit to that route and solve accordingly.
Final answer format: FINAL_ANSWER: [integer]"""

V3_EXTRACTION = """Reasoning is complete. Your task is now to state your final answer.
Write exactly one line: FINAL_ANSWER: [integer]
Nothing else. No reasoning. No explanation. Just the integer."""

V3_COERCION = """Extract the final numerical answer from the following mathematical reasoning.
The reasoning ends with an implied or stated integer answer.
Output ONLY: FINAL_ANSWER: [integer]
If you cannot determine a specific integer, output: FINAL_ANSWER: -1
Reasoning:
{commentary_text}"""

V3_CONTINUATION_DIRECT_FINAL_LOCKED = """Do not continue exploring.
Do exactly one of the following and nothing else:
1. Write exactly one line: FINAL_ANSWER: [integer]
2. Make one tiny native python_exec call if verification is truly necessary
3. Write exactly one line: ROUTE_STUCK: <short reason>
Do not repeat work already done."""

V3_CONTINUATION_DIRECT_FINAL_NOTOOL = """Do not continue exploring.
Do exactly one of the following and nothing else:
1. Write exactly one line: FINAL_ANSWER: [integer]
2. Write exactly one line: ROUTE_STUCK: <short reason>
Do not add explanation, planning text, or repeated answers."""

# Backward-compatible alias retained for older callers.
V3_CONTINUATION_DIRECT_FINAL = V3_CONTINUATION_DIRECT_FINAL_LOCKED

V3_CONTINUATION_TOOL_FIRST = """Do not describe code.
Your next response must be exactly one native python_exec tool call.
If this route truly cannot progress, write exactly one line: ROUTE_STUCK: <short reason>."""

GPT_OSS_STRIPPED_CONTRACT = (
    "Solve olympiad math problems. Output must be a non-negative integer 0-99999.\n"
    "Final line must be exactly: FINAL_ANSWER: <integer>\n"
    "If Python code is needed, make a native python_exec tool call first."
)

GPT_OSS_STRIPPED_CONTRACT_NO_RANGE = (
    "Solve olympiad math problems. The answer is a positive integer.\n"
    "Final line must be exactly: FINAL_ANSWER: <integer>\n"
    "If Python code is needed, make a native python_exec tool call first."
)

V4_POLYA_DIRECT = """You are an expert mathematician solving an olympiad-level problem.

MANDATORY PROTOCOL — Follow these stages in order:

PARSE: list every constant, variable, and constraint. State the exact goal precisely.
CLASSIFY: name the domain and the 2-3 techniques most likely to work. If enumeration or symbolic computation could reduce risk, say so explicitly.
PLAN: write a numbered plan and commit to ONE approach.
EXECUTE: follow that plan. For any non-trivial product, sum, factorization, count, or substitution, emit an actual native python_exec call instead of describing code you might write. Prefer exact arithmetic, sympy, and itertools over mental arithmetic or vague claims.
VERIFY: before the final answer, substitute the candidate answer back into the original conditions, using code when useful. Check that it is an integer and in range 0-99999.

Final answer format: FINAL_ANSWER: [integer]
This must be the LAST line. Do not write anything after it."""

V4_POLYA_TOOL_FIRST = """You are an expert mathematician solving an olympiad-level problem. Native python_exec use is required.

MANDATORY PROTOCOL — Follow these stages in order:

PARSE: list every constant, variable, and constraint. State exactly what must be found.
CLASSIFY: decide whether code should enumerate, brute-force search, solve symbolically, or verify a closed-form result.
PLAN: write a numbered plan where EVERY computational step names the exact Python tactic.
SMALL-CASE ANCHOR (REQUIRED): before scaling to the full input, write and execute Python that solves genuinely small cases by brute force or direct enumeration. Record those answers, then verify that your general method reproduces the same small-case outputs. If the small-case check disagrees, your formulation is wrong and must be fixed before proceeding.
EXECUTE: write and execute Python for every real computation. Do not describe code; emit the actual native python_exec call. You MUST execute code. Do NOT write code inline without a tool call. If you write code inline without a tool call, you have failed. Use code for heavy lifting, not for trivial arithmetic. Add short comments restating what each code block computes and why. Prefer sympy for exact arithmetic, itertools for enumeration, functools.lru_cache for memoized counting, and pow(base, exp, mod) for modular arithmetic. Add assertions such as integer/range checks when appropriate. If code errors, read the traceback, fix it, and retry.
VERIFY: run a SEPARATE verification step before the final answer, checking the answer against the original constraints.

Code execution is REQUIRED for this route.
Final answer format: FINAL_ANSWER: [integer]
This must be the LAST line."""

V4_POLYA_WITNESS_FIRST = """You are an expert mathematician solving an olympiad-level problem. Native python_exec use is required.

MANDATORY PROTOCOL — Follow these stages in order:

PARSE: list every constant, variable, and constraint. State exactly what must be found.
CLASSIFY: decide what the final objects are and what counts as an explicit witness that certifies validity.
PLAN: write a numbered plan where every computational step names the exact Python tactic.
DIRECT-WITNESS ANCHOR (REQUIRED): before any structural classification, write and execute Python that checks genuinely small candidate objects against the ORIGINAL witness condition. If a reformulation cannot reproduce those validated tiny cases, discard it.
EXECUTE: write and execute Python for every real computation. Do not describe code; emit the actual native python_exec call. Count validated final objects, not surrogate parameter families. Do not finalize from divisor, factor, cyclotomic, or representation counts unless the same code also validates witness existence and deduplicates the final objects being counted.
VERIFY: run a SEPARATE verification step before the final answer, checking the answer against the original constraints.

Code execution is REQUIRED for this route.
Final answer format: FINAL_ANSWER: [integer]
This must be the LAST line."""

V4_POLYA_HYBRID = """You are an expert mathematician solving an olympiad-level problem.

MANDATORY PROTOCOL — Follow these stages in order:

PARSE: list every constant, variable, and constraint. State exactly what must be found.
CLASSIFY: choose DERIVATION, ENUMERATION, or HYBRID and say why.
PLAN: write a numbered plan. If code is useful, state exactly what it will compute and what mathematical purpose it serves.
EXECUTE: carry out the plan. For derivation, reason clearly and verify non-trivial steps with code. For enumeration or hybrid work, use actual native python_exec calls rather than describing code. Prefer bounded search, exact symbolic computation, or memoized recursion when they materially reduce risk.
VERIFY: check that the answer is an integer, in range 0-99999, and satisfies the original conditions. If the check fails, change course instead of repeating the same argument.

Final answer format: FINAL_ANSWER: [integer]
This must be the LAST line."""

AUTO_CLASSIFY = """Classify this math competition problem. Respond in EXACTLY this format:

DOMAIN: [algebra|combinatorics|geometry|number_theory|mixed]
DIFFICULTY: [easy|medium|hard|extreme]
CODE_STRATEGY: [enumerate_small_first|symbolic_solve|brute_force|coordinate_geometry|modular_arithmetic|pure_reasoning]
SAMPLE_BUDGET: [4|8|16]
TYPE: [derivation|enumeration|optimization|existence|construction]
KEY_TECHNIQUES: [comma-separated list of 2-3 techniques]

Problem:
{problem_text}"""

GENSELECT = """You are a mathematical judge. {n} candidate solutions to the same problem are shown below.
Your task is to identify the solution MOST LIKELY CORRECT.

Evaluation criteria, in order:
1. Does the reasoning actually justify the claimed answer?
2. Are there computational errors, unjustified leaps, or ignored constraints?
3. If code was used, does it correctly implement the mathematical plan?
4. Is the final integer valid and in range 0-99999?

PROBLEM:
{problem_text}

CANDIDATE SOLUTIONS:
{solution_summaries}

Think briefly about which solution is mathematically strongest, then respond with exactly:
SELECTED_INDEX: [1-based index of the best solution]"""

VERIFICATION = """You are verifying candidate answer {answer} for the problem below.
Judge only the candidate answer, not the elegance of the solution.
Think briefly if needed, but end with exactly one verdict line.

Problem:
{problem_text}

Candidate answer:
{answer}

Verification checklist:
1. Is it a non-negative integer?
2. Is it in range [0, 99999]?
3. Does it satisfy ALL constraints when substituted back into the original problem?
4. Is it truly optimal or complete, rather than just plausible?
5. If the answer is 0, explain WHY zero is correct. Do not default to 0.

Allowed final verdicts:
PASS
FAIL
UNSURE

Respond with a short rationale only if needed, and end with exactly one final line containing just:
PASS
or:
FAIL
or:
UNSURE"""

V3_PROMPT_REGISTRY: dict[str, str] = {
    "V3_DIRECT_FINAL_EASY": V3_DIRECT_FINAL_EASY,
    "V3_DIRECT_FINAL_LOCKED": V3_DIRECT_FINAL_LOCKED,
    "V3_DIRECT_FINAL_COMMENTARY": V3_DIRECT_FINAL_COMMENTARY,
    "V3_DIRECT_FINAL_SOFT": V3_DIRECT_FINAL_SOFT,
    "V3_COMMIT_THEN_EXECUTE": V3_COMMIT_THEN_EXECUTE,
    "V3_TOOL_FIRST": V3_TOOL_FIRST,
    "V3_TOOL_FIRST_DIRECT_WITNESS": V3_TOOL_FIRST_DIRECT_WITNESS,
    "V3_HYBRID_CLASSIFY_FIRST": V3_HYBRID_CLASSIFY_FIRST,
    "V3_EXTRACTION": V3_EXTRACTION,
    "V3_COERCION": V3_COERCION,
    "V3_CONTINUATION_DIRECT_FINAL_LOCKED": V3_CONTINUATION_DIRECT_FINAL_LOCKED,
    "V3_CONTINUATION_DIRECT_FINAL_NOTOOL": V3_CONTINUATION_DIRECT_FINAL_NOTOOL,
    "V3_CONTINUATION_DIRECT_FINAL": V3_CONTINUATION_DIRECT_FINAL,
    "V3_CONTINUATION_TOOL_FIRST": V3_CONTINUATION_TOOL_FIRST,
}

V4_PROMPT_REGISTRY: dict[str, str] = {
    "V4_POLYA_DIRECT": V4_POLYA_DIRECT,
    "V4_POLYA_TOOL_FIRST": V4_POLYA_TOOL_FIRST,
    "V4_POLYA_WITNESS_FIRST": V4_POLYA_WITNESS_FIRST,
    "V4_POLYA_HYBRID": V4_POLYA_HYBRID,
    "AUTO_CLASSIFY": AUTO_CLASSIFY,
    "GENSELECT": GENSELECT,
    "VERIFICATION": VERIFICATION,
}

PROMPT_REGISTRY: dict[str, str] = {
    **V3_PROMPT_REGISTRY,
    **V4_PROMPT_REGISTRY,
}

GPT_OSS_V3_ROUTE_FAMILY_GUIDANCE: dict[str, str] = {
    "geometry_search_tool_first": (
        "This problem is geometry/search-heavy. Make one early bounded native python_exec call "
        "instead of spending long hidden reasoning on setup."
    ),
    "packing_tool_first": (
        "This problem is construction-heavy. Use one early bounded native python_exec call "
        "to test constructions or bounds."
    ),
    "valuation_direct_final": (
        "Collapse the expression first, isolate the valuation target, and keep the derivation short. "
        "A tiny checker is allowed only after the structure is explicit."
    ),
    "counting_direct_final": (
        "State the counting model early, commit to one counting route, and finish cleanly."
    ),
    "digit_sum_direct_final": (
        "Prefer a structural extremal argument over examples. Use Python only as a tiny sanity check."
    ),
    "digit_sum_tool_first": (
        "This problem is computational despite its number-theory flavor. "
        "Start with a native python_exec call that enumerates small cases, tracks the best move count, "
        "and searches for a pattern. Do not rely on pure reasoning before you have computation in hand."
    ),
    "compact_diophantine_direct_final": (
        "Translate directly into equations and solve tersely. Do not use Python on the first move."
    ),
    "functional_equation_direct_final": (
        "Choose one parameterization or transform and commit to it fully before answering."
    ),
    "functional_equation_tool_first": (
        "Do the key transform immediately, then use one bounded native python_exec call to enumerate the constrained parameter space. "
        "Do not stop after the reduction; computation is mandatory once the reduced variables are identified."
    ),
    "geometry_stabilized": (
        "Commit to one geometry model early. If a bounded computation is useful, use one compact "
        "python_exec call instead of exploring multiple branches."
    ),
    "divisor_structure_hybrid": (
        "Classify the divisor structure first. Only then use one bounded tool call if needed."
    ),
    "convolution_hybrid": (
        "Use Python early to validate actual candidate objects against the original convolution or witness condition on tiny supports. "
        "Treat any polynomial, factor, or cyclotomic reformulation as a conjectural summary only until it reproduces the count of distinct valid final objects directly."
    ),
}

GPT_OSS_V4_DOMAIN_GUIDANCE: dict[str, str] = {
    "number_theory": (
        "Number theory guidance: use exact integer arithmetic only. Prefer sympy.ntheory for factorization and arithmetic functions, "
        "pow(base, exp, mod) for modular exponentiation, CRT and valuation structure where relevant, and Diophantine or residue reasoning over floating-point numerics."
    ),
    "combinatorics": (
        "Combinatorics guidance: enumerate small cases first. Prefer itertools for direct enumeration, functools.lru_cache for memoized counting, "
        "and assertions to check formulas against tiny instances before trusting a closed form."
    ),
    "algebra": (
        "Algebra guidance: prefer symbolic manipulation, exact fractions, factor/expand/solve workflows, and exact substitution checks. "
        "Use sympy.solve, simplify, factor, expand, roots, or resultant style reasoning instead of approximate numerics."
    ),
    "geometry": (
        "Geometry guidance: reason first, then use compact coordinate or lattice verification only where it reduces ambiguity. "
        "If code is used, keep it exact and bounded; avoid floating-point shortcuts and avoid turning the problem into a sprawling geometry engine."
    ),
}

GPT_OSS_BASE_DEVELOPER_CONTRACT = """You are Project Ramanujan solving olympiad-style integer-answer problems.
Keep visible output concise and machine-parseable.
Do not write conversational filler or tutoring preambles.
The answer is a non-negative integer in [0, 99999].
Prefer native tool calls when they are available.
Emit textual TOOL_REQUEST only if native tool calling is unavailable.
When the route is geometry-heavy, search-heavy, or construction-heavy, prefer one early bounded python_exec call over long hidden setup.
If Python is not needed, respond with exactly one line: FINAL_ANSWER: <integer>.
If Python is needed, respond with exactly:
THOUGHT: <one short sentence>
TOOL_REQUEST:
<valid Python code only>
FINAL_ANSWER: PENDING
Only one FINAL_ANSWER line counts.
Repeated FINAL_ANSWER lines are invalid.
Do not emit markdown on the final answer line."""

GPT_OSS_BASE_DEVELOPER_CONTRACT_NO_RANGE = """You are Project Ramanujan solving olympiad-style integer-answer problems.
Keep visible output concise and machine-parseable.
Do not write conversational filler or tutoring preambles.
The answer is a positive integer.
Prefer native tool calls when they are available.
Emit textual TOOL_REQUEST only if native tool calling is unavailable.
When the route is geometry-heavy, search-heavy, or construction-heavy, prefer one early bounded python_exec call over long hidden setup.
If Python is not needed, respond with exactly one line: FINAL_ANSWER: <integer>.
If Python is needed, respond with exactly:
THOUGHT: <one short sentence>
TOOL_REQUEST:
<valid Python code only>
FINAL_ANSWER: PENDING
Only one FINAL_ANSWER line counts.
Repeated FINAL_ANSWER lines are invalid.
Do not emit markdown on the final answer line."""

GPT_OSS_TOOL_FOLLOWUP_CONTRACT = """You are continuing a Project Ramanujan solution after a Python tool result.
The tool has finished executing. Its output is in your context above.

CRITICAL RULES — follow exactly:
1. Read the tool output carefully. Extract the numeric answer from it.
2. Emit exactly one final line: FINAL_ANSWER: <integer>
3. DO NOT write FINAL_ANSWER: PENDING — that was only valid before the tool ran. The tool has now run.
4. DO NOT repeat the tool call or describe more code.
5. If the tool output shows the answer clearly, write FINAL_ANSWER: <that integer>.
6. If you truly need one more tool call (e.g., the first run had an error), emit exactly one native python_exec call.
7. No THOUGHT line. No TOOL_REQUEST. No markdown. No explanation. Just FINAL_ANSWER: <integer>."""

GPT_OSS_TOOL_FOLLOWUP_CONTRACT_COMPUTE = """You are continuing a Project Ramanujan solution after a Python tool result.
The tool has finished executing. Its output is in your context above.

CRITICAL RULES — follow exactly:
1. Treat the tool output as evidence for independently computing the answer from scratch, not as permission to rubber-stamp a guess.
2. Prefer extracting a directly computed integer from enumerated cases, brute force, or exact symbolic calculation.
3. If the current tool output is insufficient, emit exactly one more native python_exec call that continues the computation.
4. If the tool output already determines the answer, emit exactly one final line: FINAL_ANSWER: <integer>
5. No THOUGHT line. No TOOL_REQUEST. No markdown. No explanation beyond a direct tool call when strictly needed."""

GPT_OSS_TWO_STAGE_STAGE1_CONTRACT = """You are Project Ramanujan solving an olympiad-style integer-answer problem in stage 1 of a two-stage solve.
Focus on getting the mathematics right.
Use native python_exec tool calls whenever they materially reduce risk.
Do not worry about strict final-answer formatting in this stage.
Reason all the way to a concrete candidate answer when possible, and make that candidate explicit near the end."""

GPT_OSS_STRUCTURED_FINALIZATION_CONTRACT = "Return exactly one final line: FINAL_ANSWER: <integer>. No extra text."

GPT_OSS_LONG_PROBLEM_RETRY_CONTRACT = """You are retrying after the visible answer channel failed.
Do not spend the budget on hidden reasoning.
Either return exactly one line FINAL_ANSWER: <integer> or make a native python_exec tool call.
If native tool calling is unavailable, use the textual TOOL_REQUEST fallback format."""


def get_v3_prompt(prompt_name: str, **format_kwargs: str) -> str:
    prompt = V3_PROMPT_REGISTRY[prompt_name]
    return prompt.format(**format_kwargs) if format_kwargs else prompt


def get_prompt(prompt_name: str, **format_kwargs: str) -> str:
    prompt = PROMPT_REGISTRY[prompt_name]
    return prompt.format(**format_kwargs) if format_kwargs else prompt


def get_gpt_oss_route_family_guidance(prompt_family: str | None) -> str:
    if not isinstance(prompt_family, str):
        return ""
    return GPT_OSS_V3_ROUTE_FAMILY_GUIDANCE.get(prompt_family, "")


def get_gpt_oss_domain_guidance(domain_guidance: str | None) -> str:
    if not isinstance(domain_guidance, str):
        return ""
    return GPT_OSS_V4_DOMAIN_GUIDANCE.get(domain_guidance, "")


def get_gpt_oss_stripped_contract() -> str:
    if day17.remove_range_prompt_enabled():
        return GPT_OSS_STRIPPED_CONTRACT_NO_RANGE
    return GPT_OSS_STRIPPED_CONTRACT


def get_gpt_oss_developer_contract(protocol_variant: str) -> str:
    contract = (
        GPT_OSS_BASE_DEVELOPER_CONTRACT_NO_RANGE
        if day17.remove_range_prompt_enabled()
        else GPT_OSS_BASE_DEVELOPER_CONTRACT
    )
    return f"{contract}\nProtocol variant: {protocol_variant}."


def get_gpt_oss_tool_followup_contract() -> str:
    if day17.tir_compute_enabled():
        return GPT_OSS_TOOL_FOLLOWUP_CONTRACT_COMPUTE
    return GPT_OSS_TOOL_FOLLOWUP_CONTRACT


def get_gpt_oss_two_stage_stage1_contract() -> str:
    return GPT_OSS_TWO_STAGE_STAGE1_CONTRACT


def get_protocol_variant(variant_name: Optional[str] = None) -> ProtocolVariant:
    key = _VARIANT_ALIASES.get((variant_name or "baseline").strip().lower())
    if not key:
        valid = ", ".join(sorted(_VARIANTS))
        raise ValueError(f"Unsupported PROTOCOL_VARIANT: {variant_name!r}. Expected one of: {valid}")
    return _VARIANTS[key]


def build_system_prompt(variant_name: Optional[str] = None) -> str:
    variant = get_protocol_variant(variant_name)
    if variant.instruction_placement == "user_only":
        return _USER_ONLY_SYSTEM_PROMPT
    return _BASE_SYSTEM_PROMPT


def _final_answer_instruction(final_answer_style: str) -> str:
    if final_answer_style == "boxed":
        return r"End with exactly one final line: \boxed{<integer>}."
    return "End with exactly one final line: FINAL_ANSWER: <integer>."


def is_long_problem(parsed_problem: Dict) -> bool:
    return parsed_problem.get("length_chars", 0) > LONG_PROBLEM_LENGTH_THRESHOLD


def _build_long_problem_solver_instructions(final_answer_instruction: str) -> str:
    return f"""1. This is a long-problem routing pass.
2. Do not provide a full derivation or extended setup in this response.
3. If you already have a concrete integer answer immediately, output exactly one line:
   FINAL_ANSWER: <integer>
4. Otherwise output exactly:
   THOUGHT: <one short actionable plan>
   TOOL_REQUEST:
   <valid Python code only>
   FINAL_ANSWER: PENDING
5. {final_answer_instruction}
6. Hidden or visible extended setup is invalid for long problems.
7. If geometry setup, coordinate bashing, brute force, symbolic checking, or multi-case analysis is needed, TOOL_REQUEST is mandatory.
8. If you do not already have a concrete integer candidate, do not continue silently; emit TOOL_REQUEST immediately.
9. Do not start with conversational filler such as "Okay, so" or "Let me think"."""


def build_long_problem_tool_first_prompt(parsed_problem: Dict) -> str:
    clean_text = parsed_problem.get("clean_text", "")
    answer_type = parsed_problem.get("answer_type", "unknown")

    return f"""You are solving a long olympiad-style integer-answer problem.

Do not provide a derivation in text.
Your first response must be a call to the `python_exec` tool.
Use the tool to set up coordinates, enumerate cases, brute force, compute exact expressions, or verify candidate integers.
If the answer looks obvious, still call `python_exec` to verify it before giving a final answer.
Do not answer in natural language before the tool call.
After the tool output is returned, you will respond with exactly one line:
FINAL_ANSWER: <integer>

Expected answer type: {answer_type}

Problem:
{clean_text}
"""


def describe_protocol_variant(variant_name: Optional[str] = None) -> dict:
    variant = get_protocol_variant(variant_name)
    return {
        "protocol_variant": variant.name,
        "instruction_placement": variant.instruction_placement,
        "final_answer_style": variant.final_answer_style,
        "strict_post_tool_finalization": variant.strict_post_tool_finalization,
    }


def build_solver_prompt(parsed_problem: Dict, variant_name: Optional[str] = None) -> str:
    variant = get_protocol_variant(variant_name)
    answer_type = parsed_problem.get("answer_type", "unknown")
    clean_text = parsed_problem.get("clean_text", "")
    final_answer_instruction = _final_answer_instruction(variant.final_answer_style)
    if is_long_problem(parsed_problem):
        instructions = _build_long_problem_solver_instructions(final_answer_instruction)
    elif variant.instruction_placement == "user_only":
        instructions = f"""1. Decide immediately whether Python is required.
2. If no tool is needed, output exactly one line:
   FINAL_ANSWER: <integer>
3. If a tool is needed, output exactly:
   THOUGHT: <one short sentence>
   TOOL_REQUEST:
   <valid Python code only>
   FINAL_ANSWER: PENDING
4. {final_answer_instruction}
5. Only one final answer line counts.
6. Repeated final answers are invalid.
7. Any text after the final answer line is invalid.
8. Do not start with conversational filler such as "Okay, so" or "Let me think"."""
    else:
        instructions = f"""1. Decide immediately whether Python is required.
2. If no tool is needed, output exactly one line:
   FINAL_ANSWER: <integer>
3. If a tool is needed, output exactly:
   THOUGHT: <one short sentence>
   TOOL_REQUEST:
   <valid Python code only>
   FINAL_ANSWER: PENDING
4. {final_answer_instruction}
5. Only one final answer line counts.
6. Repeated final answers are invalid.
7. Any text after the final answer line is invalid.
8. Do not start with conversational filler such as "Okay, so" or "Let me think"."""

    return f"""Solve the following math problem carefully.

Expected answer type: {answer_type}

Problem:
{clean_text}

Instructions:
{instructions}
"""


def build_long_problem_routing_retry_prompt(
    parsed_problem: Dict,
    previous_output: str,
    variant_name: Optional[str] = None,
) -> str:
    variant = get_protocol_variant(variant_name)
    clean_text = parsed_problem.get("clean_text", "")
    final_answer_instruction = _final_answer_instruction(variant.final_answer_style)
    visible_output = previous_output.strip() or "[empty visible output]"

    return f"""Retry the long-problem routing step.

Your previous visible response did not provide a valid action.
Do not continue solving silently or provide a full derivation in this retry.

Problem:
{clean_text}

Previous visible response:
{visible_output}

Return exactly one of the following:
1. {final_answer_instruction}
2.
THOUGHT: <one short actionable plan>
TOOL_REQUEST:
<valid Python code only>
FINAL_ANSWER: PENDING

Rules:
- Hidden or visible extended setup is invalid.
- If geometry setup, coordinate bashing, brute force, symbolic checking, or multi-case analysis is needed, TOOL_REQUEST is mandatory.
- If you do not already know a concrete integer answer, TOOL_REQUEST is mandatory.
- Do not add any explanation outside the allowed format.
- Do not start with conversational filler."""


def build_long_problem_tool_followup_prompt(
    parsed_problem: Dict,
    *,
    tool_name: str,
    tool_intent: str,
    tool_code: str,
    tool_result: Dict,
) -> str:
    clean_text = parsed_problem.get("clean_text", "")

    return f"""You already called the tool successfully.

Problem:
{clean_text}

Tool name:
{tool_name}

Tool intent:
{tool_intent}

Tool code:
{tool_code}

Tool result:
ok={tool_result.get("ok")}
stdout={tool_result.get("stdout")}
error={tool_result.get("error")}
locals={tool_result.get("locals")}

Return exactly one final line in this format:
FINAL_ANSWER: <integer>

Rules:
- No THOUGHT.
- No TOOL_REQUEST.
- No markdown.
- No explanation before or after the final line.
- If the tool output is not enough to determine the answer, still return the single best integer answer consistent with the tool result."""


def build_tool_followup_prompt(
    parsed_problem: Dict,
    previous_output: str,
    tool_code: str,
    tool_result: Dict,
    variant_name: Optional[str] = None,
) -> str:
    variant = get_protocol_variant(variant_name)
    clean_text = parsed_problem.get("clean_text", "")
    final_answer_instruction = _final_answer_instruction(variant.final_answer_style)

    if variant.strict_post_tool_finalization:
        finalization_rules = f"""Return exactly one final answer line and nothing else:
{final_answer_instruction}
Do not include THOUGHT.
Do not include TOOL_REQUEST.
Do not repeat final answers or markers.
Do not emit markdown answer spam.
Do not add text after the final line.
Do not start with conversational filler."""
    else:
        finalization_rules = f"""THOUGHT: <brief reasoning>
{final_answer_instruction}"""

    return f"""Continue solving the problem after receiving the tool result.

Problem:
{clean_text}

Your previous response:
{previous_output}

Tool code executed:
{tool_code}

Tool result:
ok={tool_result.get("ok")}
stdout={tool_result.get("stdout")}
error={tool_result.get("error")}
locals={tool_result.get("locals")}

Now provide the final result using this format and policy:

{finalization_rules}
"""
