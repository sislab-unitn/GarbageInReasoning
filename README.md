# Garbage In, Competence Out? Why Reasoning Scores are Unreliable & What to Do About It

Repository containing the code for the paper *Garbage In, Competence Out? Why Reasoning Scores are Unreliable & What to Do About It*.

## Introduction

We conduct a systematic audit of three widely used reasoning benchmarks, SocialIQa, FauxPas-EAI, and ToMi, and uncover pervasive flaws in both benchmark items and evaluation methodology. Using five LLMs (GPT-3, 3.5, 4, o1, and LLaMA 3.1) as diagnostic tools, we identify structural, semantic, and pragmatic issues in benchmark design (e.g., duplicated items, ambiguous wording, and implausible answers), as well as scoring procedures that prioritize output form over reasoning process. Through structured human annotation and re-evaluation on cleaned benchmark subsets, we find that model scores often improve not due to better reasoning, but due to the removal of input-level noise, including ambiguous wording, annotation inconsistencies, and scoring artifacts. Further analyses show that model performance is highly sensitive to minor input variations such as context availability and phrasing, revealing that high scores may reflect alignment with format-specific cues rather than consistent inference based on the input. These findings challenge the validity of current benchmark-based claims about reasoning in LLMs, and highlight the need for evaluation protocols that assess reasoning as a process of drawing inference from available information, rather than as static output selection. We release audited data and evaluation tools to support more interpretable and diagnostic assessments of model reasoning.

## Benchmarks

We worked on these social reasoning benchmarks and we share a cleaned version of them.

* [SocialIQA](https://aclanthology.org/D19-1454/) is a crowd-sourced multiple-choice benchmark for evaluating social and emotional commonsense reasoning. Each example consists of a short context, a social inference question, and three answer options, only one of which is correct.
* [ToMi](https://aclanthology.org/D19-1598/) is a synthetic benchmark designed to evaluate Theory of Mind (ToM) reasoning in language models. Each story involves multiple agents and object relocations, followed by six questions probing memory, reality, first-order beliefs, and second-order beliefs. ToMi includes randomized story generation, distractor agents, and varied event sequences to reduce exploitable patterns and test belief reasoning more robustly.
* [FauxPas-EAI](https://aclanthology.org/2023.findings-acl.663/) includes 44 short narrative stories, 22 containing a social faux pas and 22 matched control stories, each followed by four multiple-choice questions: a detection question, a statement identification question, a comprehension check, and a false-belief probe. The benchmark is adapted from a clinical Theory of Mind test (Baron-Cohen et al., 1999) designed to assess social reasoning and false-belief understanding.

## Repository Structure

The repository contains three folders, each one specific for every benchmark evaluated. Inside each folder there are:
* the scripts to reproduce the paper experiments
* documents related to specific issues analysis and human annotation guidelines
* cleaned data version of the original benchmarks
* a `README.md` file explaining the folder contents and how to run experiments

## License

## How To Cite
