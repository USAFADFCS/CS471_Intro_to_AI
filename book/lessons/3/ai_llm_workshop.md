# L3: LLM Workshop

:::{admonition} Lesson Objectives
:class: note

* Understand how LLMs are also AI systems with strengths and weaknesses.

* Evaluate how changes in prompts affect LLM AI outputs.

* Identify inconsistencies and failure cases in LLM AI systems.

* Explain how input design influences output quality for AI systems.
:::

## LLMs as AI Systems (The Pipeline)


In Lesson 2, we established that AI systems are not magic; they are pipelines that map inputs to outputs. A {term}`Large Language Model (LLM)` is no different. It operates as an {term}`Agent` processing unstructured data.

**Input (Sensor):** The text prompt you type into the interface.

**Preprocessing:** The system breaks your text down into sub-word numbers via {term}`Tokenization`.

**Modeling (Inference):** The neural network mathematically calculates the most probable next token based on its training.

**Output (Actuator):** The generated text string provided back to the user.


### Strengths and Weaknesses

* Strengths: LLMs excel at processing massive amounts of unstructured data. They can summarize a 50-page intelligence report into a 3-bullet brief in seconds, or translate intercepted communications instantaneously.

* Weaknesses: LLMs are statistical prediction engines, not knowledge databases. They do not "know" facts; they predict what word mathematically looks best next. This makes them highly non-deterministic (they might give you two different answers to the exact same question) and prone to severe errors if not handled correctly.

## Input Design: The Science of Prompting

In a traditional software agent, if you click a button, a specific function runs. In an LLM, the input text is the programming language. How you design your input directly controls the quality of the {term}`Inference`. This is known as {term}`Prompt Engineering`. We cover prompt engineering in more detail in Lesson 34: Prompt Engineering.

If you provide a weak input, the LLM will fall back on its generalized, broad training data, resulting in vague or unhelpful outputs.

{term}`Zero-Shot Prompting`: Asking the AI to perform a task without giving it any examples. (e.g., "Classify this intercepted radio message.")

{term}`Few-Shot Prompting`: Providing the AI with examples of the exact input-output format you want before asking it to process the new data. This drastically reduces errors and forces the model into a strict operational format.

The more constraints and context you place in the input, the smaller the mathematical search space becomes for the model, resulting in a safer, more accurate output. 

## Failure Cases and Inconsistencies

Because LLMs are statistical text generators, they exhibit unique failure modes that demand strict human-on-the-loop evaluation:

{term}`Hallucinations`: The model confidently generates false, fabricated, or nonsensical information. For example, an LLM might invent a completely fictitious enemy weapons system simply because the words "stealth," "drone," and "laser" are mathematically highly associated in its training data.

{term}`Prompt Injection`: A vulnerability where an adversary embeds malicious instructions within a piece of data (like a translated document) that causes the LLM to ignore its original system instructions and output dangerous or compromised data.

**Context Forgetting:** If an input prompt exceeds the model's {term}`Context Window`, the model will "forget" the instructions at the beginning of the prompt, leading to inconsistent outputs that fail to follow the Commander's intent.


## Summary Infographic
![LLM Workshop](../../figures/llm_workshop.png "LLM Workshop")

<br>
<hr width="100%" size="4" color="black">

## Knowledge Check & Practice Questions

1. You paste a 10-page enemy doctrine manual into an LLM and ask it to summarize the key points. It provides a great summary, but includes a paragraph about naval tactics that wasn't in the original text. What failure mode is this?

2. Describe "Few-Shot Prompting"?

3. Why do LLMs sometimes generate entirely different answers to the exact same prompt?
