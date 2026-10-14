# LLM translation prompt

Model: Claude Opus 5.5 (`claude-opus-5-5`), via Claude Code. Output stored verbatim in `translate/llm/clipNN.json`.

> Translate each Hindi sentence (IndicVoices Level-1 transcript: fillers, repetitions and English words written in Devanagari) into English twice. Return a JSON list of `{"hi", "literal", "fluent"}`, one per input line, `hi` copied exactly.
>
> - **literal**: keep Hindi word order where English allows, keep every filler, repetition and self-repair ("five lakh... no no, five crore"), keep amounts in lakh/crore, keep digit strings as spoken ("nine eight double seven").
> - **fluent**: idiomatic English a listener would want. State only the corrected value after a self-repair, drop fillers, keep amounts in lakh/crore and phone numbers as digits.
> - Never add or drop a negation. A tag question "…ना?" is not a negation.
