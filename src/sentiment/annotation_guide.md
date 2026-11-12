# SocialPulse AI — Sentiment Annotation Guide

## 1. Project Overview

**Project Name:** SocialPulse AI

**Research Topic:** Public Opinion Analysis of Generative AI Assistants: ChatGPT vs Gemini vs Claude.

**Objective:** The objective of this annotation process is to classify public comments according to the sentiment expressed by their authors toward generative AI assistants.

The comments may come from permitted datasets and YouTube comments.

The annotation language for this phase is English.

## 2. Sentiment Categories

Each comment must receive one of the following five labels.

### 2.1 Positive

**Definition:** The author expresses satisfaction, appreciation, trust, or a favorable opinion about an AI assistant.

Examples:
- "ChatGPT is amazing for programming."
- "Claude gives very detailed explanations."
- "Gemini is really helpful for studying."

### 2.2 Negative

**Definition:** The author expresses dissatisfaction, criticism, frustration, or an unfavorable opinion.

Examples:
- "Gemini constantly gives incorrect answers."
- "ChatGPT is very slow today."
- "Claude is not useful for my work."

### 2.3 Neutral

**Definition:** The comment provides information, asks a question, or describes an experience without expressing a clear positive or negative opinion.

Examples:
- "I started using Claude yesterday."
- "Does Gemini support PDF files?"
- "ChatGPT was released in 2022."

### 2.4 Mixed

**Definition:** The comment contains both positive and negative opinions about an AI assistant.

Examples:
- "ChatGPT is useful, but sometimes it gives inaccurate answers."
- "Claude writes well, although it can be slow."
- "Gemini has interesting features, but its responses are inconsistent."

### 2.5 Unclear

**Definition:** The meaning or sentiment cannot be reliably interpreted because the comment is incomplete, unintelligible, or too ambiguous.

Examples:
- "That thing... well, you know."
- "Hmmmmmm..."
- "It is what it is."

Unclear comments should be excluded from supervised model training unless they are later clarified.

## 3. Assistant Identification

Each comment must also be associated with an assistant when possible.

Allowed values:

- ChatGPT
- Gemini
- Claude
- Multiple
- Unknown

Use Multiple when a comment discusses two or more assistants.

Use Unknown when the assistant cannot be identified reliably.

## 4. Annotation Rules

1. Read the complete comment before assigning a label.
2. Focus on the author's expressed opinion, not your own opinion.
3. Do not classify every question as negative.
4. Do not assume that mentioning an assistant means the author likes or dislikes it.
5. Use Mixed when both positive and negative opinions are clearly expressed.
6. Be careful with sarcasm, irony, jokes, and informal language.
7. Use Unclear when the meaning cannot be interpreted reliably.
8. Do not invent missing context.
9. Preserve the original comment text.
10. Apply the same rules consistently to every comment.

## 5. Special Cases

### Questions
A question without an expressed opinion is generally Neutral.

Example: "Can ChatGPT generate images?" → Neutral.

### Comparisons
When a comment compares several assistants, identify all assistants mentioned.

Example: "Claude is better than Gemini for writing." → Positive, Multiple.

### Sarcasm
Do not guess the intended meaning when sarcasm is unclear. Use Unclear if it cannot be interpreted reliably.

### Spam and Irrelevant Comments
Record irrelevant or spam comments separately if the project schema supports it. Do not automatically assign them a sentiment.

## 6. Dataset Annotation Format

The labeled dataset must contain the following columns:

| Column | Description |
|---|---|
| comment_id | Unique comment identifier |
| comment_text | Original comment text |
| assistant | ChatGPT, Gemini, Claude, Multiple, or Unknown |
| sentiment | Positive, Negative, Neutral, Mixed, or Unclear |

## 7. Quality Control

- All annotators must follow this guide.
- A shared subset of comments should be independently annotated by both students.
- Disagreements should be reviewed and resolved through discussion.
- Annotation consistency should be measured using Cohen's Kappa when two annotators label the same comments.
- Any changes to annotation rules must be documented.

## 8. Final Objective

The annotated dataset will be used to train and evaluate supervised sentiment classification models, including Logistic Regression and Linear SVM with TF-IDF features.

The results will support the comparative analysis of public opinions about ChatGPT, Gemini, and Claude.