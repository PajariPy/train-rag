# Detecting data drift for the SLM

> [!note] Status
> Template. Theoretical only, no build. Follows article 008.

<!-- Intro: the SLM is cheap because it is trained once and then runs on every new document.
That only holds while new documents look like the training data. -->

## Why drift matters

<!-- What changes in real company data (new tools, templates, topics, writing styles) and
why nothing in the pipeline would notice. -->

## Detecting it with labels

<!-- Accuracy on new data vs old data: re-label a small fresh sample with the LLM, score the
SLM on it, compare with the calibration-set score. -->

## Detecting it without labels

<!-- Cheap signals to research: input statistics, embedding distribution shift, SLM
confidence from the logits, share of empty or malformed outputs. -->

## When to act

<!-- How often to check, sample size, what threshold triggers re-labelling and retraining. -->

## Open questions
