# Driftbox corpus (fictional)

These documents describe **Driftbox**, a made-up self-hosted file sync and backup product from
a made-up company, **Quillfeather Labs**. Every name, number, price and policy here is invented
for the lab. Any resemblance to a real product is accidental.

The corpus is deliberately small (17 docs) so experiments run in minutes on a laptop, but it has
a few properties that make retrieval interesting:

- Facts are specific (ports, prices, durations), so wrong retrieval produces visibly wrong answers.
- Some facts appear in more than one document (for example ARM64 support), and the FAQ overlaps
  with other pages, which creates realistic distractors.
- Some questions in the golden set need two documents to answer fully.
- Two golden questions are unanswerable from the corpus, to test whether the generator admits it.

The loader skips this README; each other `*.md` file is one document and its file stem is its ID.
