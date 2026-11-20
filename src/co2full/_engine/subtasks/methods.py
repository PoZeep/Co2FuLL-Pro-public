COMPILE_V2 = """Compare two decompiled functions for binary code similarity: could these be builds of the same original source function?
Work through three subtasks using the supplied code:
1. Describe each function's core operation, its inputs, outputs and state changes in one sentence. Do not guess an algorithm from the examples or anonymous names.
2. Map the distinctive operations between A and B. Account for inlining of helper functions, loop unrolling/vectorization, 32/64-bit word size and compiler specialization of constant parameters. A specialized build may have fewer parameters or branches. Explain only transformations supported by the remaining shared computation.
3. Decide whether remaining differences change the core source operation or are plausibly compilation/decompilation effects. Similar generic purpose alone is insufficient: functions with different input element semantics, transformations or essential side effects can be different source functions. Conversely, do not reject a strong algorithmic correspondence only because a helper is opaque, intermediate types differ, offsets differ with pointer size, or one version expands more implementation detail.
This is source-origin classification, not a demand to prove instruction-level equivalence. Use the most plausible source relationship supported by the code.
Write at most 250 words covering the three subtasks, then exactly <same_source>yes</same_source> or <same_source>no</same_source>.

Code A:
{code_a}

Code B:
{code_b}"""
