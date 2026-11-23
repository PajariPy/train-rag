Think through how the RAG should handle the way real people write at work. The benchmark's
synthetic documents don't show that, so our results may not carry over to a real company's
data.

## Why
EnterpriseRAG-Bench was written by an LLM, mostly through a cheap pipeline where each
document saw only a company overview and related file paths. The result is unusually clean:
every Slack message is a full paragraph, facts and numbers are restated in full, each thread
stays on one topic and ends with a decision. Real work chats look different:

- fragments and filler: "+1", "lgtm", "on it", reactions, typos, side jokes;
- pointers instead of facts: "see the dashboard", a screenshot, a link to JIRA-1234, "numbers
  in the doc";
- assumed context: "same as last time", "the thing Maya fixed", nicknames and abbreviations;
- threads that drift, move to a call, or never write the answer down;
- low information density: a few facts buried in chatter.

Some of our problems are easier here than in reality (the answer is always written in the
thread), and some are artificial (template families of near-twins, brainstorm 006).

## How
Questions to think about, before any build:
- **Pointers:** follow links and ticket ids between documents (Slack → Jira, Gmail →
  Drive), so a thread that says "see INC-123" leads to the answer.
- **Context:** resolve "same as last time" from the thread's neighbours, the channel, or the
  same participants' earlier threads.
- **Noise:** drop or down-weight filler messages before embedding, without losing short but
  decisive ones ("approved", "rolled back to 1.21.4").
- **Missing answers:** when the decision happened in a call or never got written, say so
  rather than guess (the clarification and "info not found" work).
- **People:** who said it matters (the on-call engineer, the security lead); speaker and role
  are already parsed for Slack, Jira, Linear and Fireflies.
- **Testing:** how to check any of this without real company data. Options: a small,
  hand-made "messy" rewrite of some benchmark threads, or a public dataset of real chats or
  tickets.

Which of our results lean on the synthetic style (the fine-tuned embedder and reranker
learn the generator's way of writing) and which should transfer (channel and source
filters, links, metadata).

Related: [[Try HyDE and query rewriting]], the data preprocessing problem in the semantic search map
(`docs/research/research_map.canvas`), brainstorm 006.
