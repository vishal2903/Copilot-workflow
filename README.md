# Copilot-workflow

________________________________________
1) Exact phrase to start the flow
Copy and send one of these (pick the one you like):
Short / direct:
Generate a lesson plan on [TOPIC]. Start HITL discovery for a lesson_plan and ask 6–7 clarifying questions. After every answer, show Objective / Assumptions / Paraphrase of my input / Next step. At the end ask “Are you done?” and then “Should I make the lesson plan now?”
(Replace [TOPIC] with the real topic, e.g. “Prompt Engineering for Designers”.)
________________________________________
[Most Imp]
7) Extra tips to guarantee good output
●	Always include numbers (minutes, counts, versions).

●	Give at least one sample deliverable.

●	If you want the lesson plan in a particular format, add at start: “Output format: Markdown lesson plan with sections — Objectives, Prereqs, Timeline, Activities, Assessment (max marks 20), Resources.”

●	To force exactly 6 questions: add “Ask exactly 6 clarifying questions.”

●	To force interactive questions after every answer: add “Do not advance until you show the 4-point block.”




2) How to answer each HITL question (so you always get the 4-point block)
When the assistant asks a question, reply with answers that include concrete constraints/examples. Use this mini-template:
Answer template (copyable):
Short answer (1–2 sentences).
 Examples / numbers / time: e.g. “Target audience: junior designers (0–2 yrs). Session length: 90 minutes. Expected pre-req: basic Figma skills.”
 Deliverable: e.g. “Students will submit a one-page prompt notebook and 3 generated images.”
Why this works: the code checks for specificity and length — giving numbers, time, deliverables and examples prevents “too-short” early returns and forces the 4-point summary to appear.
________________________________________
3) Exact phrasing for common clarifications the HITL will ask
If asked about audience:
Target audience: [role and experience]. Example: “graphic designers, 0–2 years experience.”
If asked about duration:
Total session time: [minutes/hours]. Example: “90 minutes total (60 min lecture + 30 min hands-on).”
If asked about assessment:
Assessment & deliverables: [what to submit + rubric]. Example: “Submit 3 images + 200-word writeup. Rubric: creativity 8, prompt clarity 6, reproducibility 6 (out of 20).”
If asked about tools:
Tools & environment: [list]. Example: “ComfyUI + SDXL or Runway Gen4; provide links to starter files.”
________________________________________
4) What to say when the HITL finishes
When it asks “Are you done?” answer:
yes
When it then asks “Should I make the lesson plan now?” answer:
yes, generate the lesson plan now
Or if you want changes:
not yet — please add X / ask 2 more questions
________________________________________
5) Troubleshooting quick-lines (if the flow doesn’t start)
If nothing happens, Speak exactly:
“Start HITL discovery for a lesson_plan now.”
If the assistant starts asking questions but skips the 4-point block, speak:
“After my answer, please show Objective / Assumptions / Paraphrase / Next step before asking the next question.”
________________________________________
6) Example short interaction (one round) — for your reference
You (initial):
Generate a lesson plan on Prompt Engineering for Designers. Start HITL discovery for a lesson_plan and ask 6 questions. After each answer show Objective / Assumptions / Paraphrase / Next step.
Assistant (Q1):
Who is the target audience?
You (good answer):
Junior UI/UX designers, 0–2 years experience. Session length: 90 minutes. Pre-req: basic Figma and Photoshop. Deliverable: 3 prompt-driven mockups and a 200-word reflection.
Assistant (immediately after your answer — this is the 4-point block you should see):
Objective: Produce a 90-minute hands-on session that teaches junior designers to translate design briefs into high-quality image prompts you can reuse.
 Assumptions: Participants know Figma basics and can open a provided starter file; they have access to SDXL or Runway.
 Your input (paraphrase): Junior UI/UX designers (0–2 yrs); 90 min; deliverable = 3 mockups + 200-word reflection.
 Next step: Confirm tool choice and desired assessment rubric.
Then assistant asks Q2 (tool choice), and flow repeats.
________________________________________







