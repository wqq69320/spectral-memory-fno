# M12 Decision Log

## Decisions Made

- Create a lightweight presentation package under `docs/presentation/` instead
  of generating a slide deck in the repository.
- Reference existing generated figure and table paths rather than copying
  generated artifacts into git.
- Make claim boundaries explicit in both the talk materials and final caveats.
- Present cost-efficiency outputs as reporting-mechanism artifacts, not final
  efficiency evidence.
- Treat SM-FNO2D v2 as an implementation and protocol extension, not as a
  finalized architecture with established benchmark behavior.

## Alternatives Considered

- A full slide deck was deferred because the immediate goal is durable written
  material that students can turn into slides.
- Embedding generated figures in docs was rejected because generated outputs
  should remain ignored and reproducible from configs.
- Including metric tables as main presentation evidence was rejected because
  smoke metrics can be misread as rankings.
- Writing a final-results report with claims was rejected because current runs
  are protocol-scale sanity checks only.

## Rationale

- Markdown documents are easy to review, diff, and revise before the forum.
- Referencing artifact paths keeps the presentation reproducible while
  preserving the repository rule that generated artifacts stay out of git.
- Separating figure guidance, Q&A, and caveats reduces the risk of overstating
  the evidence during a live presentation.
- The package gives each student a concrete speaking role while keeping the
  narrative aligned with the fair-comparison protocol.

## Known Limitations

- The package is not a rendered slide deck.
- Some referenced artifacts are local generated files and must be regenerated
  from configs if absent in another checkout.
- The materials intentionally avoid final claims; additional experiments are
  required before a results-focused paper or benchmark presentation.

## Follow-Up Work

- Convert the outline into slides once speaker assignments are final.
- Add any advisor-requested figures after confirming the corresponding configs
  and commands.
- Rerun selected figures from clean configs before public presentation.
- Create a final benchmark report only after larger validated experiments are
  complete.
