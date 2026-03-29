# Architecture

## Overview

Scaffoldly transforms a URL (blog post, paper, repo) into a hands-on course through a 4-phase pipeline. Phase 1 uses structured output to produce a Blueprint. Phase 2 uses multi-turn conversations to produce lesson documents and exercises. Phase 3 validates and reviews.

```mermaid
flowchart TB
    URL["User pastes URL + describes background"]
    URL --> P0

    subgraph P0["Phase 0: Preprocessing"]
        direction LR
        Detect["detect_source_type()"] --> Handler
        Handler --> |arxiv| TeX["Download TeX source"]
        Handler --> |blog| Jina["Jina Reader → markdown + images"]
        Handler --> |pdf| PDF["Download + Jina text"]
        Handler --> |github| Clone["git clone --depth 1"]
    end

    P0 --> Sources["_sources/manifest.json + local files"]
    Sources --> Budget["sources.py: budget management"]
    Budget --> SourceContent["source_content (full text, no truncation)"]

    SourceContent --> P1a

    subgraph P1["Phase 1: Blueprint Design (structured output)"]
        direction TB
        P1a["Phase 1a: Analyze\n1 API call (design model)\n→ Analysis (Pydantic)"]
        P1a --> P1b["Phase 1b: Blueprint\n1 API call (design model)\n→ CurriculumDesign (Pydantic)"]
    end

    P1b --> Blueprint["Blueprint JSON"]
    P1b --> RootFiles["README.md + requirements.txt"]

    Blueprint --> P2

    subgraph P2["Phase 2: Module Generation (parallel multi-turn conversations)"]
        direction LR
        M0["Module 0\n~11 turns"]
        M1["Module 1\n~11 turns"]
        M2["Module 2\n~11 turns"]
        MN["Module N\n~11 turns"]
    end

    P2 --> Files["Lesson docs + exercise files + solutions"]
    Files --> P3

    subgraph P3["Phase 3: Review"]
        direction TB
        PreFlight["3a: Pre-flight\n(Python, no LLM)\nsyntax, TODOs, length, patterns"]
        PreFlight --> LLMReview["3b: LLM Review\n(1 call per module)\ncontract compliance, quality"]
        LLMReview --> |REVISE| Regen["Re-generate\nfailed modules"]
        LLMReview --> |PASS| Done["Done"]
    end
```

## Phase 1: Blueprint Design

Two structured output calls using the design model. The Blueprint is the sole coordination mechanism between phases — Phase 2 modules never see each other's output.

```mermaid
sequenceDiagram
    participant Python
    participant LLM as Design Model
    participant Disk

    Note over Python,Disk: Phase 1a: Analyze
    Python->>LLM: system: ANALYSIS_PROMPT<br/>user: full source_content + URL
    LLM-->>Python: Analysis (Pydantic)<br/>concepts, prerequisites, content_type
    Python->>Disk: _analysis.json

    Note over Python,Disk: Phase 1b: Blueprint
    Python->>LLM: system: CURRICULUM_DESIGN_PROMPT<br/>user: Analysis + full source_content + student_level
    LLM-->>Python: CurriculumDesign (Pydantic)<br/>curriculum + scaffold_contracts + key_excerpts + root_readme
    Python->>Disk: _curriculum.json, README.md, requirements.txt
    Python-->>Python: coverage check (essential concepts → exercises)
    Python-->>Python: emit curriculum event → Web UI renders DAG
```

### Blueprint Schema

```
CurriculumDesign
├── curriculum
│   ├── course_title, course_description
│   └── modules[]
│       ├── title, description, learning_objectives
│       ├── depends_on[]                          ← module dependency graph
│       ├── key_excerpts[]                        ← verbatim formulas from source
│       ├── exercises[]
│       │   ├── title, type, scaffolding_level
│       │   ├── what_is_provided                  ← "class Node with __init__, import block, __main__ harness"
│       │   ├── what_student_writes               ← "Node.backward() ~8-12 lines; compute_loss() ~5-8 lines"
│       │   ├── key_insight                       ← "backward() must accumulate gradients at fan-out nodes"
│       │   ├── common_mistakes                   ← "forgetting to zero gradients between batches"
│       │   ├── milestone                         ← "prints gradient table, all errors < 1e-5"
│       │   └── expected_output_pattern           ← "relative error"
│       └── inline_questions[]
├── shared_definitions
│   ├── language, dependencies[], naming_convention
├── root_readme                                   ← full course README.md content
└── requirements                                  ← requirements.txt content
```

## Phase 2: Module Generation

Each module is generated through a **multi-turn conversation**. Modules run in parallel across `anyio.create_task_group()`, but turns within a module are sequential — each turn sees all prior turns.

```mermaid
sequenceDiagram
    participant Python
    participant LLM as Generate Model
    participant Exec as Python Executor
    participant Disk

    Note over Python,Disk: Turn 1: Lesson Document
    Python->>LLM: system: MODULE_CONVERSATION_PROMPT<br/>user: Blueprint slice + full source material<br/>"Write the lesson for this module"
    LLM-->>Python: Raw markdown (3,000-10,000 words)<br/>objectives, running example, inline checks,<br/>formula translation, comprehension questions
    Python->>Disk: module_XX/README.md

    Note over Python,Disk: Turn 2: Exercise 1 Scaffold
    Python->>LLM: [conversation history]<br/>"Write scaffold for exercise 1: {spec}"
    LLM-->>Python: Raw Python code<br/>(~65% provided, ~35% TODO markers)
    Python->>Disk: module_XX/ex01_foo.py
    Python->>Python: ast.parse() — syntax check

    Note over Python,Disk: Turn 3: Exercise 1 Solution
    Python->>LLM: [conversation history]<br/>"Write the solution version"
    LLM-->>Python: Raw Python code<br/>(TODOs filled in)
    Python->>Disk: module_XX/_solutions/ex01_foo.py
    Python->>Exec: python ex01_foo.py (10s timeout)
    Exec-->>Python: stdout: "k=1: dp=12, brute=12 [OK]"

    Note over Python,Disk: Turn 4: Exercise 2 Scaffold
    Python->>LLM: [conversation history + execution output]<br/>"Ex1 output: dp=12, brute=12.<br/>Now write scaffold for exercise 2: {spec}"
    LLM-->>Python: Raw Python code<br/>(references ex1's actual results)
    Python->>Disk: module_XX/ex02_bar.py

    Note over Python,Disk: Turn 5: Exercise 2 Solution
    Python->>LLM: [conversation history]<br/>"Write the solution version"
    LLM-->>Python: Raw Python code
    Python->>Disk: module_XX/_solutions/ex02_bar.py
    Python->>Exec: python ex02_bar.py
    Exec-->>Python: stdout: "lambda=5.0 → best_k=3"

    Note over Python,Disk: ...repeat for exercises 3-5...

    Note over Python,Disk: Fix Turn (if syntax errors)
    Python->>LLM: "Syntax error in ex03: line 42. Fix it."
    LLM-->>Python: Corrected file
    Python->>Disk: overwrite ex03
```

### Why conversations, not single-shot

```
SINGLE-SHOT (old, broken):
  1 API call → JSON blob with ALL files → hope it works

  Result: 200-word READMEs, empty exercise shells, param*42

CONVERSATIONAL (current):
  Turn 1: Write 5,000-word lesson (deep source engagement)
  Turn 2: Write ex1 scaffold (full attention on one file)
  Turn 3: Write ex1 solution → EXECUTE → see output
  Turn 4: Write ex2 scaffold (references ex1's real output)
  ...

  Result: MIT-quality lessons, working exercises, real numbers
```

### Parallel execution model

```mermaid
gantt
    title Phase 2: Module Generation (parallel)
    dateFormat X
    axisFormat %s

    section Module 0
    Lesson (Turn 1)        :m0t1, 0, 15
    Ex1 scaffold (Turn 2)  :m0t2, after m0t1, 5
    Ex1 solution (Turn 3)  :m0t3, after m0t2, 5
    Ex2 scaffold (Turn 4)  :m0t4, after m0t3, 5
    Ex2 solution (Turn 5)  :m0t5, after m0t4, 5
    Ex3 scaffold (Turn 6)  :m0t6, after m0t5, 5
    Ex3 solution (Turn 7)  :m0t7, after m0t6, 5

    section Module 1
    Lesson (Turn 1)        :m1t1, 0, 15
    Ex1 scaffold (Turn 2)  :m1t2, after m1t1, 5
    Ex1 solution (Turn 3)  :m1t3, after m1t2, 5
    Ex2 scaffold (Turn 4)  :m1t4, after m1t3, 5
    Ex2 solution (Turn 5)  :m1t5, after m1t4, 5
    Ex3 scaffold (Turn 6)  :m1t6, after m1t5, 5
    Ex3 solution (Turn 7)  :m1t7, after m1t6, 5

    section Module 2
    Lesson (Turn 1)        :m2t1, 0, 15
    Ex1 scaffold (Turn 2)  :m2t2, after m2t1, 5
    Ex1 solution (Turn 3)  :m2t3, after m2t2, 5
    Ex2 scaffold (Turn 4)  :m2t4, after m2t3, 5
    Ex2 solution (Turn 5)  :m2t5, after m2t4, 5
    Ex3 scaffold (Turn 6)  :m2t6, after m2t5, 5
    Ex3 solution (Turn 7)  :m2t7, after m2t6, 5
```

All modules start simultaneously. Within each module, turns are sequential (each turn sees prior context). Total wall-clock time ≈ time for the slowest module, not the sum.

## Phase 3: Review

```mermaid
flowchart TB
    subgraph PreFlight["3a: Pre-flight (Python, zero LLM cost)"]
        Syntax["ast.parse() on all .py files"]
        TODO["Check TODO/YOUR CODE HERE markers"]
        Length["File length > 15 non-comment lines"]
        Main["__main__ block present"]
        Pattern["Solution contains expected_output_pattern"]
    end

    PreFlight --> |FAIL| Regen["Re-generate module\n(full conversational flow)"]
    PreFlight --> |PASS| LLMReview

    subgraph LLMReview["3b: LLM Review (1 call per module, parallel)"]
        Contract["Contract compliance:\nscaffold matches what_is_provided?"]
        Excerpt["Key excerpt fidelity:\nsolution implements actual formulas?"]
        Scaffold["Scaffolding quality:\nTODOs, docstrings, line counts"]
        Milestone["Milestone quality:\n__main__ prints expected output?"]
        Difficulty["Progressive difficulty"]
        Realism["Realistic data"]
        Questions["Analytical questions Level 3+?"]
    end

    LLMReview --> |PASS| Done["✓ Module passes"]
    LLMReview --> |REVISE| Regen
    Regen --> PreFlight
```

## Data Flow

What each phase sees:

```mermaid
flowchart LR
    Source["Full source\nmaterial"]
    Analysis["Analysis\n(concepts, type)"]
    Blueprint["Blueprint\n(contracts, excerpts)"]
    Lesson["Lesson\n(5,000 words)"]
    Ex1Out["Ex1 output\n(execution result)"]

    Source --> |Phase 1a| Analysis
    Source --> |Phase 1b| Blueprint
    Analysis --> |Phase 1b| Blueprint

    Source --> |Phase 2, Turn 1| Lesson
    Blueprint --> |Phase 2, Turn 1| Lesson

    Source -.-> |in conversation context| Ex1
    Blueprint --> |exercise spec| Ex1["Ex1 scaffold\n+ solution"]
    Lesson -.-> |in conversation context| Ex1

    Ex1 --> |executed| Ex1Out
    Ex1Out --> |in conversation context| Ex2["Ex2 scaffold\n+ solution"]
    Blueprint --> |exercise spec| Ex2
```

Key property: **the full source material is available at every turn**. No truncation, no summarization. The model reads the actual blog/paper formulas while writing code.

## Cost Model

For a 5-module course with 3 exercises each, using GPT-5.4:

| Phase | Calls | Input tokens | Output tokens | Cost |
|---|---|---|---|---|
| 1a: Analyze | 1 | ~15K | ~3K | ~$0.08 |
| 1b: Blueprint | 1 | ~25K | ~8K | ~$0.18 |
| 2: Generate | 5 modules × ~7 turns | ~200K/module | ~25K/module | ~$6.50 |
| 3: Review | 5 | ~30K/module | ~2K/module | ~$0.55 |
| **Total** | **~40** | **~1.2M** | **~140K** | **~$7.30** |

Phase 2 dominates cost because conversations accumulate context. Each module's later turns re-read the lesson + prior exercises. This is the price of quality — the model needs that context to produce coherent, progressive exercises.

## Stack

```
┌─────────────────────────────────────────┐
│  Web UI (vanilla JS, SSE, DAG viz)      │
├─────────────────────────────────────────┤
│  Starlette + Uvicorn (web server)       │
├─────────────────────────────────────────┤
│  pipeline.py (orchestration)            │
│  ├── Phase 1: Instructor (structured)   │
│  ├── Phase 2: raw completions (conv.)   │
│  └── Phase 3: Instructor (structured)   │
├─────────────────────────────────────────┤
│  LLMClient (llm.py)                     │
│  ├── LiteLLM (provider routing)         │
│  ├── Instructor (Pydantic validation)   │
│  └── Cost tracking                      │
├─────────────────────────────────────────┤
│  Provider APIs                          │
│  OpenAI / Anthropic / Google / Ollama   │
└─────────────────────────────────────────┘
```

No LangChain, no LangGraph, no agent frameworks. Direct API calls through LiteLLM, with Instructor for structured output in Phases 1 and 3, and raw chat completions for Phase 2 conversations.
