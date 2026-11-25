# Writing English

An English page reads as if one engineer explained RPent to another: direct, concrete, and in standard technical terms. These rules apply to docs pages and all other English project writing: PR descriptions, review comments, commit messages, and issues.

## Voice

- Write in plain, readable English understood on the first read. Prefer concrete verbs over stacked noun phrases or stiff spec-sheet wording.
- Address the reader as "you" and give instructions in the imperative ("Run `rpent --robot libero`", "Set `PI05_CHECKPOINT_PATH`"). Use "we" sparingly, only when walking through an example on a development page.
- Use present tense and active voice ("The planner calls `move_to`", not "`move_to` is invoked by the planner").
- Lead with the outcome, then the step.
- Write "Use X" instead of "Consider using X" or "You may want to use X"; reserve "may" and "might" for real uncertainty.

## Words to cut

| Cut | Why | Write instead |
|---|---|---|
| simply, just, easily, obviously, basically, of course | the step may not be simple for the reader | nothing |
| seamlessly, powerful, robust, comprehensive, cutting-edge, state-of-the-art, leverage, crucial, pivotal, delve, showcase, unlock, empower | sales or machine register | the concrete fact, or nothing |
| it is worth noting that, note that, please note, importantly | filler | the fact itself, or a `.. note::` |
| in order to, allows you to, enables you to, is able to | wordy | to, you can, can |
| currently, new, now, no longer, recently | goes stale | the version or current behavior, or nothing |
| a wide range of, various, a variety of | vague | the actual items or count |

## Constructions to avoid

- Negative-contrast framing ("this is not A, not B, but C", "not just X, but Y") and restating what something is not after saying what it is. State directly what it does unless clearing up a real misconception.
- Trailing "-ing" benefit clauses ("…, ensuring reliable execution"). State the concrete consequence or cut it.
- Lists padded to three items; list the real count.
- Closing sentences that praise the design or restate the paragraph, and closing summary sections.
- "serves as", "acts as", "boasts", "features" where "is" or "has" fits.
- Announcements instead of content ("In this section, we will…", "As mentioned above", "Notice that…").
- Rotating synonyms for one concept; follow [glossary.md](glossary.md).

## Calques from Chinese

| Avoid | Write |
|---|---|
| video memory | GPU memory |
| realize the function, realize support | implement, support |
| carry out evaluation, perform the run | evaluate, run |
| open or close the switch | enable or disable |
| the relevant configurations | the flags or config keys (name them) |
| in the case of | if, for |
| based on (for 基于, everywhere) | with, using, on, or restructure |
| it is necessary to | you must, or the imperative |

## Headings

- Title Case (lowercase only a, an, the, and, or, of, to, in, on, with, vs).
- Task headings start with a verb (`Run a Task`, `View Results`, `Configure Planners`); concept headings are noun phrases (`Memory Design`, `System Architecture`).

## Other project writing

PR descriptions, review comments, commit messages, and issues follow the same voice, cut list, and code-name rule ([base.md](base.md#write-about-what-the-reader-does-not-about-the-code)): state what changed or what fails in user/behavior terms first, and cite files or symbols only where needed to locate the change. Keep each Markdown paragraph on one line for GitHub rendering.
