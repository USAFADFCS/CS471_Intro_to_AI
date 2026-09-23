# CS471 Artificial Intelligence: Interactive Course Curriculum

This repository contains the complete Teachbooks-powered curriculum for CS471. The materials are designed to bridge core AI theory with tactical military applications, moving students from deterministic search algorithms to reinforcement learning and probabilistic reasoning under the "fog of war."

The content is written in MyST Markdown and compiled via Jupyter Book/Teachbooks to provide an interactive, accessible, and highly structured learning environment.

## Curriculum Components

The repository is structured to support both independent study and instructor-led classroom engagement.

* **Textbook Chapters (`/lessons/`)**: The core instructional material. Each lesson integrates foundational AI math with military scenarios (e.g., UAV navigation, cyber-defense, electronic warfare). Chapters include Mermaid.js diagrams for state spaces and transition models.
* **Knowledge Checks & Practice Exercises**: Embedded directly at the end of each lesson. These feature step-by-step mathematical breakdowns hidden behind interactive HTML `<details>` tags, allowing students to attempt the problem before revealing the solution.
* **Master Glossary (`master_glossary.md`)**: A centralized MyST `{glossary}` defining all conceptual and tactical terminology. Terms throughout the textbook are hyperlinked back to this file using the `{term}` role.
* **Notation Reference (`notations.md`)**: A dedicated cheat sheet for the mathematical symbology used across the course, ensuring consistency when transitioning between Markov Decision Processes, Q-Learning, and Hidden Markov Models.
* **Python Labs (`/labs/`)**: Standalone Markdown-based Jupyter Notebooks. These bridge the theoretical math with computational implementation, walking students through building AI engines (e.g., online HMM filtering, Q-learning loops) using standard Python libraries.
* **Printable Worksheets (`/worksheets/`)**: LaTeX-formatted handouts designed for in-class guided practice.
* **Instructor Solution Keys (`/solutions/`)**: Companion LaTeX files for all worksheets. These feature fully worked mathematical proofs and tactical interpretations formatted in shaded `tcolorbox` environments for easy grading and review.

## How to Use This Textbook (Student Guide)

To achieve conceptual mastery rather than rote memorization, engage with the lesson materials in the following sequence:

1. **Read the Lesson Chapter:** Focus on the tactical scenarios and the step-by-step breakdown of the core equations. Click on highlighted `{term}` links if you need to refresh a definition from previous blocks.
2. **Attempt the Embedded Exercises:** At the bottom of each chapter, complete the practice exercises on scratch paper *before* clicking "Show solution." The step-by-step breakdowns will help you identify exactly where your math or logic deviated.
3. **Execute the Python Lab:** Review the lesson's corresponding Python lab to see how the mathematical formulas translate into code. Focus heavily on the "Interpreting the Results" sections to understand the tactical "why" behind the code's output.
4. **Complete the Independent Practice Worksheets:** Use the provided LaTeX worksheets to simulate the Graded Review (GR) environment. The formatting, notation, and problem structures on these worksheets strictly mirror what you will encounter on exams.

## Build Instructions

This repository requires the Teachbooks environment to compile the interactive web version of the text.

1. Install the required dependencies:
```bash
pip install -r requirements.txt

```


2. Build the book locally:
```bash
jb build .

```


3. Open `_build/html/index.html` in your browser to view the compiled textbook.

If you are modifying the LaTeX worksheets or solution keys, use standard `pdflatex` or an editor like Overleaf to compile the PDFs for classroom distribution.