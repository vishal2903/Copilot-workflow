User Voice Input
       ↓
   ORCHESTRATOR (Zeno)
       ↓
   ┌───┴───┐
   │ HITL  │ ←── Discovery questions (one-by-one)
   │ Agent │ ←── Parses answers, confirms each
   └───┬───┘
       ↓ (when discovery complete)
   ┌───┴───────┐
   │ RETRIEVAL │ ←── Index Doc → Data Doc lookup
   │   Agent   │ ←── Drive search
   └───┬───────┘     Cross-references related topics
       ↓
   ┌───┴───┐
   │  WEB  │ ←── Community sentiment
   │ Agent │ ←── Recent developments
   └───┬───┘
       ↓
   ┌───┴──────┐
   │ COMPOSER │ ←── 8 priority sections
   │  Agent   │ ←── Grounded in retrieved content
   └───┬──────┘
       ↓
   ┌───┴───┐
   │  QA   │ ←── Validates all sections present
   │ Agent │ ←── Checks grounding in Data Doc
   └───┬───┘     Detects generic content
       ↓
   Upload to Drive + Voice Confirmation
