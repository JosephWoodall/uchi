---
name: grounded-answer
description: Answers factual questions only from sources it can check, cites each claim, and abstains honestly when a claim cannot be verified. Use when the user asks for a factual answer, a summary of their own documents or codebase, a number, a date, or a how-to that depends on facts, and wants the answer to be trustworthy rather than fluent. Do NOT use for creative writing, brainstorming, opinions, or when the user explicitly wants a guess with no checking.
---

# Grounded Answer

Give an answer the reader can check. If a claim cannot be traced to a
source you actually read, or to a calculation you actually ran, do not
present it as fact. Say so, and say what would verify it.

This skill is standalone. It does not require uchi, or any other tool,
to be installed. It uses only the agent's normal tools: reading files
the user points to, searching the user's documents or codebase, and,
when available, running a calculation or a web search.

## Workflow

Follow these steps in order, for every factual answer.

### 1. Split the question into checkable claims

Before answering, list the separate facts the answer would need:

- names, dates, prices, quantities, and other numbers
- what something does, what happened, and in what order
- any comparison, ranking, or "best" claim

A claim that cannot be checked is not a fact yet. Keep it in the
"Not verified" section at the end, or leave it out.

### 2. Find a source for each claim, in this order

1. Files, documents, or code the user supplied or pointed to.
2. The user's own project, notes, or data files, searched directly.
3. A trusted public source, when the question is about the outside
   world and the user has not supplied a source.

Do not use your own memory, training, or a plausible-sounding detail
as a source. If the only "source" is that an answer sounds right,
treat the claim as unverified.

If two sources disagree, say so. Quote or summarise both, name which
one you are following for the answer, and explain why, for example
because it is newer, primary, or directly from the project.

### 3. Verify numbers and code instead of restating them

- For a calculation, work it out step by step and, when a calculator
  or code execution is available, run it. Show the inputs, not just
  the result.
- For a claim about code, read the code. Do not summarise a function
  from its name alone.
- For a total, average, or count, state what was counted and over
  what range. "50 invoices" is not a source; the file and the rows
  counted are.

If you cannot run the check, say "not run" next to that claim. Do not
upgrade an unrun calculation to a verified result.

### 4. Write the answer in this format

**Answer**
The direct answer, in plain language, with no claim left unsourced.

**Sources**
One line per claim, in this form:

- Claim: source, with enough detail to find it again, for example
  a file path and section, a document title and page, or a URL.

**Not verified / abstained**
Anything you could not check, in plain language, plus what would
check it, for example "the 2024 invoice total, which needs the
2024 export file."

Keep the answer short. A long answer with weak sourcing is worse
than a short answer with strong sourcing.

## Examples

### Good: grounded, with a clear limit

User: "What was our Q3 revenue growth?"

Answer: "Revenue grew 12% in Q3, from $100,000 in Q2 to $112,000
in Q3."

Sources:
- Q2 and Q3 totals: `finance/q3-report.csv`, rows for Q2 and Q3,
  checked by adding the monthly figures: 30,000 + 35,000 + 35,000
  = 100,000 for Q2, and 36,000 + 37,000 + 39,000 = 112,000 for Q3.

Not verified: nothing in this answer.

### Good: honest abstention

User: "Why did sales drop in October?"

Answer: "I cannot tell you why sales dropped in October from the
files supplied. October sales were $41,000, down from $52,000 in
September."

Sources:
- October and September totals: `finance/sales.csv`.

Not verified: the cause of the drop. That would need sales notes,
a campaign log, or a comparable period, none of which were supplied.

### Bad: do not do this

"Sales dropped in October because of the holiday slowdown and a
failed marketing campaign." That invents a cause with no source.
Even if it sounds plausible, it is a guess presented as a fact.

## Rules that override style

- Never soften an abstention into a guess. "Probably", "likely",
  and "it seems" do not turn an unchecked claim into a fact.
  If it is unchecked, put it in "Not verified".
- Never cite a source you did not read. A search result title is
  not a source; the page, file, or passage behind it is.
- Never invent a citation, file path, URL, number, or quote to
  make an answer look sourced.
- If the user says "just guess", you may give a clearly labelled
  guess, in its own section, after the grounded answer, and never
  mixed into the grounded claims.
- Do not change a user's files, run network requests to unknown
  destinations, or read credentials or secrets to answer a
  question. Read only the sources the question needs.

## Before you finish: self-check

Run through this list and fix anything that fails before replying:

1. Every number, date, name, and factual claim in "Answer" has a
   matching line in "Sources".
2. Every calculation was worked out, not restated, and its inputs
   are shown.
3. Anything without a source is in "Not verified / abstained",
   not in "Answer".
4. No source is cited that was not actually read.
5. Conflicting sources, if any, are named, not hidden.

If any check fails and cannot be fixed from the sources available,
shrink the answer until it passes, or abstain on that part.
